"""RL flight controller — LOTF-aligned, rate-control cascade only.

Runs at 50 Hz (matches LOTF env dt=0.02). Each step:
    1. read obs from DroneState (27-dim, normalised)
    2. policy → action in SI: [thrust_N, ωx, ωy, ωz]
    3. clip to physical bounds, push into FIFO
    4. send to Crazyflie via send_setpoint_manual (rate control)
    5. log a raw SI row for the offline LOTF residual fit
"""
import threading
import time

import numpy as np
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

from cf_params import (DT, ACTION_LOW, ACTION_HIGH, RAD_TO_DEG,
                       SEND_OMEGA_MAX, GOAL_PHASE1, GOAL_PHASE2, GOAL_SWITCH_T,
                       thrust_N_to_pwm_pct)
from drone_state import _quat_wxyz_to_R
from finetune_lotf import finetune_from_log
from RL_policy import RLPolicy


# ────────────────────────── LOTF raw log (SI units) ──────────────────────────
_LOTF_LOG = {'t': [], 'p': [], 'R': [], 'v': [], 'T_N': [], 'omega': []}

# ───────────────────────── in-flight finetune settings ───────────────────────
# Lighter than the offline CLI: a 2 s window holds ~100 samples, and the work
# runs in a background thread while the 50 Hz loop keeps flying, so we keep it
# short to limit CPU contention / control jitter.
ONLINE_FT_WINDOW_SEC = 2.0
ONLINE_FT_RES_EPOCHS = 80
ONLINE_FT_BPTT_EPOCHS = 15
ONLINE_FT_NUM_ENVS = 16
ONLINE_FT_HORIZON = 60


def save_lotf_log(path='lotf_log.npz'):
    if not _LOTF_LOG['t']:
        print('[lotf] empty log, nothing to save.')
        return
    np.savez_compressed(
        path,
        t=np.asarray(_LOTF_LOG['t']),
        p=np.asarray(_LOTF_LOG['p']),
        R=np.asarray(_LOTF_LOG['R']),
        v=np.asarray(_LOTF_LOG['v']),
        T_N=np.asarray(_LOTF_LOG['T_N']),
        omega=np.asarray(_LOTF_LOG['omega']),
    )
    print(f'[lotf] saved {len(_LOTF_LOG["t"])} rows -> {path}')


class RLDroneController:
    def __init__(self, scf, drone_state, rl_agent):
        self.scf = scf
        self.state = drone_state
        self.rl_agent = rl_agent
        self.is_flying = False
        self.fail_safe = False
        self.land_fail_safe = False
        self.exit_program = False
        self.history_cmd = {'time': [], 'x_pos': [], 'y_pos': [], 'z_pos': []}
        self.thrust_history = {'time': [], 'thrust_N': []}
        self.dt = DT                        # 50 Hz
        self.period_ms = int(1000 * self.dt)
        # In-flight learning-on-the-fly (enabled by test_main via --online-finetune)
        self.online_finetune = False
        self.finetuning = False             # True while a background fit is running
        self._finetune_requested = False    # set by the keyboard listener thread
        self.start_time = 0.0               # takeoff wall-clock (for thrust-plot axis)
        self.hotswap_times = []             # wall-clock of each completed hot-swap

    # ───────────────────────────── cflib log setup ─────────────────────────────
    def setup_logs(self):
        configs = [
            ('Position', self.period_ms, ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z']),
            ('Velocity', self.period_ms, ['stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz']),
            ('Gyro',     self.period_ms, ['gyro.x', 'gyro.y', 'gyro.z']),
            ('Quat',     self.period_ms, ['stateEstimate.qx', 'stateEstimate.qy',
                                          'stateEstimate.qz', 'stateEstimate.qw']),
        ]
        for name, period, variables in configs:
            cfg = LogConfig(name=name, period_in_ms=period)
            for v in variables:
                cfg.add_variable(v, 'float')
            self.scf.cf.log.add_config(cfg)
            cfg.data_received_cb.add_callback(lambda t, d, c: self.state.update_from_log(d))
            cfg.start()

    # ───────────────────────────── inference + logging ─────────────────────────
    def _rl_step(self):
        obs = self.state.get_obs()
        action = self.rl_agent.get_action(obs).cpu().numpy().astype(np.float32)
        action = np.clip(action, ACTION_LOW, ACTION_HIGH)

        self.state.push_action(action)
        self._log_lotf_row(action)
        return action

    def _log_lotf_row(self, action):
        T_N = float(action[0])
        omega = action[1:]                                # rad/s, command
        q = self.state.latest['quat_wxyz']
        R_w = _quat_wxyz_to_R(q)
        _LOTF_LOG['t'].append(time.time())
        _LOTF_LOG['p'].append(self.state.latest['pos'].copy())
        _LOTF_LOG['R'].append(R_w.flatten())
        _LOTF_LOG['v'].append(self.state.latest['vel'].copy())
        _LOTF_LOG['T_N'].append(T_N)
        _LOTF_LOG['omega'].append(omega.copy())

    # ───────────────────────────── send to Crazyflie ─────────────────────────
    def _send_action(self, action):
        T_N = float(action[0])
        # Safety clamp on the radio setpoint only — the FIFO/log already stored
        # the full-authority action in _rl_step, so the policy's observation and
        # the residual dataset are untouched.
        wx, wy, wz = np.clip(action[1:], -SEND_OMEGA_MAX, SEND_OMEGA_MAX)
        roll_dps = wx * RAD_TO_DEG
        pitch_dps = wy * RAD_TO_DEG
        yaw_dps = wz * RAD_TO_DEG
        pwm_pct = thrust_N_to_pwm_pct(T_N)
        # send_setpoint_manual in rate mode: rate_in_deg_s=True
        self.scf.cf.commander.send_setpoint_manual(
            roll_dps, pitch_dps, yaw_dps, int(pwm_pct), True
        )
        self.thrust_history['thrust_N'].append(T_N)

    # ──────────────────────── in-flight finetune + hot-swap ────────────────────
    def request_finetune(self):
        """Flag a finetune. Called from the keyboard listener thread (key 'f')."""
        if not self.online_finetune:
            print("[online-ft] disabled — relancer avec --online-finetune"); return
        if self.finetuning:
            print("[online-ft] déjà en cours, ignoré"); return
        if not self.state.RL:
            print("[online-ft] fenêtre RL pas encore active, ignoré"); return
        self._finetune_requested = True

    def _start_finetune(self):
        """Snapshot data + base weights (main thread), then spawn the worker."""
        self.finetuning = True
        self._finetune_requested = False
        log_snapshot = {k: list(v) for k, v in _LOTF_LOG.items()}
        base_state = {k: v.detach().clone()
                      for k, v in self.rl_agent.policy.state_dict().items()}
        print(f"[online-ft] start — résiduel + BPTT sur les {ONLINE_FT_WINDOW_SEC:.0f} "
              f"dernières s ; vol maintenu avec la policy actuelle")
        threading.Thread(target=self._finetune_worker,
                         args=(log_snapshot, base_state), daemon=True).start()

    def _finetune_worker(self, log_snapshot, base_state):
        """Heavy work off the control loop; atomic policy hot-swap on success."""
        try:
            new_state, _ = finetune_from_log(
                base_state, log_snapshot, window_sec=ONLINE_FT_WINDOW_SEC,
                res_epochs=ONLINE_FT_RES_EPOCHS, bptt_epochs=ONLINE_FT_BPTT_EPOCHS,
                num_envs=ONLINE_FT_NUM_ENVS, horizon=ONLINE_FT_HORIZON, verbose=False)
            self.rl_agent = RLPolicy.from_state_dict(new_state)   # atomic ref swap
            self.hotswap_times.append(time.time())
            print("[online-ft] terminé → policy hot-swap effectué")
        except Exception as e:
            print(f"[online-ft] abandonné, policy inchangée : {e}")
        finally:
            self.finetuning = False

    # ───────────────────────────── main loop ─────────────────────────────────
    def run(self):
        mc = MotionCommander(self.scf, default_height=float(self.state.goal[2]))
        take_off = False
        start_time = 0.0

        print("\nPRÊT. Maintenez ENTRÉE pour voler, ESPACE pour fail-safe.")

        while not self.exit_program:
            if not self.is_flying:
                time.sleep(self.dt); continue

            if not take_off:
                mc.take_off()
                start_time = time.time()
                self.start_time = start_time
                take_off = True

            t = time.time() - start_time
            if self.fail_safe:
                self.scf.cf.commander.send_hover_setpoint(0, 0, 0, 0.3)

            elif self.land_fail_safe:
                self.scf.cf.commander.send_stop_setpoint()
                mc.land(); self.exit_program = True

            elif t > 35:
                self.state.RL = False
                self.scf.cf.commander.send_stop_setpoint()
                mc.land(); self.exit_program = True

            elif t > 5:
                # ───── RL active window ─────
                self.state.RL = True
                # Immediate two-phase goal switch (t_rl = time since RL took over).
                t_rl = t - 5.0
                self.state.set_goal(GOAL_PHASE2 if t_rl >= GOAL_SWITCH_T else GOAL_PHASE1)
                if self._finetune_requested and not self.finetuning:
                    self._start_finetune()
                action = self._rl_step()
                self._send_action(action)
                self.history_cmd['time'].append(t - 5)
                self.history_cmd['x_pos'].append(float(self.state.goal[0]))
                self.history_cmd['y_pos'].append(float(self.state.goal[1]))
                self.history_cmd['z_pos'].append(float(self.state.goal[2]))
                self.thrust_history['time'].append(t)

            elif t > 1:
                self.scf.cf.commander.send_hover_setpoint(0, 0, 0, float(self.state.goal[2]))
            else:
                self.scf.cf.commander.send_hover_setpoint(0, 0, 0, float(self.state.goal[2]) + 0.1)

            time.sleep(self.dt)

        self.scf.cf.commander.send_stop_setpoint()
