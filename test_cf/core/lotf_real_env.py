"""Frontière « radio cflib + firmware » du VRAI Crazyflie — miroir de `LotfHoverEnv`.

Ce backend remplace Genesis par le drone réel : il expose EXACTEMENT la même API
que `lotf_genesis_env.LotfHoverEnv` (`get_obs`, `step`, `set_goal`), pilotée par le
même cerveau LOTF (`lotf_online`). Au lieu de pousser l'action SI dans la physique
Genesis, il l'envoie par radio (`cflib`) au firmware embarqué.

Comme la démo Genesis, il réutilise les objets de `RL-real/` (single source of truth) :
    * `DroneState`  → obs 27-dim + FIFO des dernières actions (alimenté EN ASYNCHRONE
      par les callbacks de log cflib, comme sur le vrai vol).
    * `cf_params`   → constantes + map `thrust_N_to_pwm_pct` (calibration cflib).

Les conventions de `setup_logs` et `_send_action` sont reprises telles quelles de
`RL-real/drone_controller.py` (la partie « interface drone » qui marchait), mais SANS
réimporter l'ancien finetune in-process (`finetune_from_log`) qu'on remplace par le
worker JAX. Boucle de contrôle temps réel à 50 Hz (DT=0.02, comme LOTF).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

THIS_DIR = Path(__file__).resolve().parent              # core/
REPO = next(p for p in THIS_DIR.parents if (p / "lotf").is_dir())   # racine du dépôt
REAL_DIR = REPO / "test_cf" / "RL-real"
if str(REAL_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DIR))

import cf_params as P                                   # noqa: E402  (single source of truth)
from cf_params import (DT, ACTION_LOW, ACTION_HIGH, RAD_TO_DEG,   # noqa: E402
                       SEND_OMEGA_MAX, thrust_N_to_pwm_pct)
from drone_state import DroneState                      # noqa: E402  (obs 27-dim + FIFO, code réel)


class LotfRealEnv:
    """Env vrai Crazyflie piloté par la pipeline LOTF — même API que `LotfHoverEnv`.

    L'observation 27-dim et le FIFO d'actions viennent du VRAI `DroneState`, alimenté
    par les callbacks de log cflib (`setup_logs`) ; la conversion action→radio
    reproduit `drone_controller._send_action`.
    """

    def __init__(self, scf, goal=None):
        self.scf = scf
        self.dt = P.DT                                  # pas de décision (50 Hz)
        self.goal = np.asarray(goal if goal is not None else P.HOVER_GOAL, dtype=np.float64)

        # ── pipeline real : obs 27-dim + FIFO des dernières actions ──
        self.state = DroneState(goal=self.goal)
        self.num_obs = P.NUM_OBS
        self.num_actions = P.NUM_ACTIONS

        # Décollage/atterrissage via MotionCommander (comme drone_controller.run).
        self.mc = MotionCommander(self.scf, default_height=float(self.goal[2]))
        self.episode_length = 0

        # Historique de poussée (pour le plot post-vol, cf utils.plot_thrust).
        self.thrust_history = {'time': [], 'thrust_N': []}

    # ───────────────────────────── cflib log setup ─────────────────────────────
    def setup_logs(self):
        """Branche les callbacks de log cflib → DroneState (cf drone_controller).

        Position/Velocity/Gyro/Quat à la période de contrôle ; chaque trame met à
        jour `state.latest`, d'où `get_obs()` reconstruit l'obs 27-dim."""
        period_ms = int(1000 * self.dt)
        configs = [
            ('Position', period_ms, ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z']),
            ('Velocity', period_ms, ['stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz']),
            ('Gyro',     period_ms, ['gyro.x', 'gyro.y', 'gyro.z']),
            ('Quat',     period_ms, ['stateEstimate.qx', 'stateEstimate.qy',
                                     'stateEstimate.qz', 'stateEstimate.qw']),
        ]
        for name, period, variables in configs:
            cfg = LogConfig(name=name, period_in_ms=period)
            for v in variables:
                cfg.add_variable(v, 'float')
            self.scf.cf.log.add_config(cfg)
            cfg.data_received_cb.add_callback(lambda t, d, c: self.state.update_from_log(d))
            cfg.start()

    # ───────────────────────────── obs ─────────────────────────────
    def get_obs(self):
        """Obs 27-dim construite par le VRAI `DroneState.get_obs()` (1-D, float32)."""
        return self.state.get_obs()

    # ───────────────────────── action SI → radio ─────────────────────────
    def _send_action(self, action_np):
        """Reproduit `drone_controller._send_action` : action SI → setpoint radio.

        Safety clamp sur le setpoint radio uniquement (ω → SEND_OMEGA_MAX) ; la FIFO
        et le log gardent l'action pleine autorité (obs/résidu intacts)."""
        T_N = float(action_np[0])
        wx, wy, wz = np.clip(action_np[1:], -SEND_OMEGA_MAX, SEND_OMEGA_MAX)
        roll_dps = wx * RAD_TO_DEG
        pitch_dps = wy * RAD_TO_DEG
        yaw_dps = wz * RAD_TO_DEG
        pwm_pct = thrust_N_to_pwm_pct(T_N)
        # send_setpoint_manual en mode rate : rate_in_deg_s=True
        self.scf.cf.commander.send_setpoint_manual(
            roll_dps, pitch_dps, yaw_dps, int(pwm_pct), True)
        return T_N, pwm_pct

    # ───────────────────────────── step ─────────────────────────────
    def step(self, action):
        """action (4,) SI = [thrust_N, ωx, ωy, ωz]. -> (obs, info).

        Reproduit `drone_controller._rl_step` + `_send_action` :
          1. clip aux bornes physiques, push dans le FIFO (obs des pas suivants) ;
          2. action SI → setpoint radio (clamp de sécurité ω) ;
          3. cadence temps réel : sleep(dt) pour tenir ~50 Hz.
        Les capteurs arrivent en asynchrone (callbacks cflib) ; `get_obs()` lit le
        dernier état reçu.
        """
        if hasattr(action, "detach"):
            action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        # même clip que controller._rl_step (RLPolicy clippe déjà, ceinture+bretelles)
        action_np = np.clip(action_np, ACTION_LOW, ACTION_HIGH)

        # FIFO du vrai DroneState (l'action pleine autorité entre dans l'obs)
        self.state.push_action(action_np)

        T_N, pwm_pct = self._send_action(action_np)
        self.thrust_history['thrust_N'].append(T_N)

        time.sleep(self.dt)                             # cadence temps réel ~50 Hz
        self.episode_length += 1

        pos = self.state.latest['pos']
        info = {
            "pos": pos,
            "pos_error": float(np.linalg.norm(pos - self.goal)),
            "thrust_N": T_N,
            "pwm_pct": pwm_pct,
        }
        return self.get_obs(), info

    # ───────────────────────── cycle de vol (cf drone_controller.run) ─────────
    def takeoff(self):
        """Décolle via MotionCommander (monte à default_height = goal z)."""
        self.mc.take_off()

    def hover_idle(self, z=None):
        """Hover firmware stabilisé (avant/après la fenêtre RL) — pas de policy."""
        z = float(self.goal[2]) if z is None else float(z)
        self.scf.cf.commander.send_hover_setpoint(0, 0, 0, z)
        time.sleep(self.dt)

    def emergency_stop(self):
        """KILL SWITCH — coupe les moteurs et NE LES RALLUME PAS (le drone CHUTE).

        Point CRITIQUE : après `mc.take_off()`, cflib lance un thread de fond
        (`_SetPointThread`) qui renvoie `send_hover_setpoint(...)` toutes les ~200 ms
        en boucle. Un simple `send_stop_setpoint()` ne coupe qu'un instant : 200 ms
        plus tard ce thread re-stabilise le vol. Il faut donc ARRÊTER ce thread, sinon
        les moteurs se rallument (bug observé en vol). On NE désarme PAS (inutile, et
        ça empêche de re-décoller proprement) — couper + tuer le thread suffit à chuter.

        Idempotent et sûr à appeler depuis le thread clavier."""
        cmd = self.scf.cf.commander
        try:
            cmd.send_stop_setpoint()                 # coupe tout de suite
        except Exception:
            pass
        # Arrête le thread de setpoint du MotionCommander (sinon il renvoie du hover).
        th = getattr(self.mc, "_thread", None)
        if th is not None:
            try:
                th.stop()                            # met TERMINATE + join (le thread meurt)
            except Exception:
                pass
            self.mc._thread = None
            self.mc._is_flying = False
        try:
            cmd.send_stop_setpoint()                 # dernier mot, une fois le thread mort
        except Exception:
            pass

    # Alias historique (ancien nom appelé ailleurs) → même coupe d'urgence.
    kill = emergency_stop

    def land(self):
        """Stoppe les setpoints RL puis atterrit proprement (descente contrôlée)."""
        self.scf.cf.commander.send_stop_setpoint()
        self.mc.land()

    # ───────────────────────────── housekeeping ─────────────────────────────
    def set_goal(self, goal):
        """Change la cible en vol (comme `DroneState.set_goal` côté réel)."""
        self.goal = np.asarray(goal, dtype=np.float64)
        self.state.set_goal(self.goal)
