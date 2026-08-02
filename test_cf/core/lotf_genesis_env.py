"""Pont Genesis qui PILOTE directement la pipeline real LOTF (`RL-real/`).

But : tester dans Genesis (physique rigide + URDF cf2x) la VRAIE pipeline de
contrôle LOTF (Pan et al.), AVANT de risquer le Crazyflie. Au lieu de recopier
les conventions obs/action, ce pont réutilise les objets de `RL-real/` :

    * `DroneState`  (RL-real/drone_state.py)  → construit l'obs 27-dim ET gère le
      FIFO des dernières actions ; même code que sur le vrai drone.
    * `RLPolicy`    (RL-real/RL_policy.py)    → réseau 27→512→512→4 + biais hover.
    * `cf_params`   (RL-real/cf_params.py)    → TOUTES les constantes + la map
      `thrust_N_to_pwm_pct` (single source of truth, partagée avec le vrai vol).

Genesis ne remplace QUE la frontière « radio cflib + firmware embarqué », qui
n'a pas d'équivalent réutilisable :

    action SI [T_N, ωx, ωy, ωz]                      (sortie politique)
      └─ mêmes conversions que `drone_controller._send_action` :
         thrust_N → %PWM (cf_params), ω clampé à SEND_OMEGA_MAX (rad/s)
      └─ PID de rate Genesis (genesis_pid.CrazyfliePIDTorch, cascade=1) → RPM moteurs
      └─ physique Genesis (cf2x.urdf)

Boucle : politique à 50 Hz (DT=0.02, comme LOTF), physique Genesis à 100 Hz
(2 sous-pas / décision) pour réutiliser le PID validé à dt=0.01 de RL-sim.

Mono-environnement : on réutilise tel quel `DroneState` (numpy, 1 drone), ce qui
est exactement le régime d'un test de hover avant le vrai vol.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

THIS_DIR = Path(__file__).resolve().parent
# Racine du dépôt = 1er parent contenant le package `lotf/` (robuste au rangement).
REPO = next(p for p in THIS_DIR.parents if (p / "lotf").is_dir())
REAL_DIR = REPO / "test_cf" / "RL-real"  # pipeline real LOTF (cf_params, drone_state, ...)
for p in (str(REPO), str(REAL_DIR)):     # REPO -> package lotf
    if p not in sys.path:
        sys.path.insert(0, p)

from genesis_pid import CrazyfliePIDTorch               # noqa: E402  (PID Genesis validé)
import cf_params as P                                   # noqa: E402  (single source of truth)
from drone_state import DroneState                      # noqa: E402  (obs 27-dim + FIFO, code réel)

URDF_PATH = REPO / "modele_drones" / "cf2x.urdf"

# Physique Genesis : 100 Hz, 2 sous-pas → 50 Hz côté politique (= LOTF DT)
PHYSICS_DT = 0.01
SUBSTEPS_PER_STEP = max(1, round(P.DT / PHYSICS_DT))    # = 2

# Calibration actionneur Genesis (mesurée par calibrate_thrust.py, masse réelle
# simulée 0.042 kg) : la force verticale RÉELLE (Newtons honnêtes) produite par la
# chaîne PWM%→get_rpm_from_pwm→set_propellers_rpm→physique Genesis vaut
# F(pwm) = A·pwm² + B·pwm + C. On INVERSE cette courbe pour que la poussée commandée
# T_N (force SI de la politique) soit RÉELLEMENT délivrée en Newtons par Genesis.
# Actionneur HONNÊTE : la politique réglée pour 0.027 kg commande ~0.265 N alors que
# le drone (0.042 kg) en exige 0.412 N pour planer → il sagge → résidu de masse réel
# que le learning-on-the-fly (résidu + BPTT) doit corriger (démo LOTF dans Genesis).
# La map cf_params.thrust_N_to_pwm_pct (calibration du VRAI drone cflib) reste inchangée.
GENESIS_THRUST_FIT = (2.592776e-05, 3.981357e-03, 1.240423e-02)


def genesis_force_to_pwm_pct(T_N: float) -> float:
    """Inverse de F(pwm) Genesis : %PWM tel que Genesis produise la force T_N (N)."""
    A, B, C = GENESIS_THRUST_FIT
    disc = B * B - 4.0 * A * (C - T_N)
    if disc <= 0.0:
        return 0.0
    pwm = (-B + np.sqrt(disc)) / (2.0 * A)
    return float(np.clip(pwm, 0.0, 100.0))


class LotfHoverEnv:
    """Env Genesis mono-drone piloté par la pipeline real LOTF (`RL-real/`).

    L'observation 27-dim et le FIFO d'actions viennent du VRAI `DroneState` ;
    la conversion action→moteurs reproduit `drone_controller._send_action`.
    """

    def __init__(self, goal=None, show_viewer: bool = False,
                 visualize_target: bool = True, visualize_camera: bool = False,
                 max_visualize_FPS: int = 60, freeze_inertia: bool = False):
        self.num_envs = 1
        self.device = gs.device
        self.dt = P.DT                                  # pas de décision (50 Hz)
        self.physics_dt = PHYSICS_DT
        self.substeps = SUBSTEPS_PER_STEP
        # Diagnostic : si True, set_total_mass ne change QUE la masse (translation),
        # en gardant l'inertie d'origine -> isole l'effet de l'inertie sur la boucle
        # de rate (cf. set_total_mass). Défaut False = comportement physique normal
        # (l'inertie suit la masse, comme pour la démo de finetuning au bouton 'm').
        self.freeze_inertia = freeze_inertia

        self.goal = np.asarray(goal if goal is not None else P.HOVER_GOAL, dtype=np.float64)
        self._goal_t = torch.tensor(self.goal, device=self.device, dtype=gs.tc_float).unsqueeze(0)

        # ── pipeline real : obs 27-dim + FIFO des dernières actions ──
        self.state = DroneState(goal=self.goal)
        self.num_obs = P.NUM_OBS
        self.num_actions = P.NUM_ACTIONS

        self.visualize_target = visualize_target

        # ── scène Genesis ──
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=max_visualize_FPS,
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                dt=self.physics_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())

        if self.visualize_target:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(file="meshes/sphere.obj", scale=0.05,
                                     fixed=False, collision=False),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5))),
            )
        else:
            self.target = None

        self.cam = None
        if visualize_camera:
            self.cam = self.scene.add_camera(res=(640, 480), pos=(3.5, 0.0, 2.5),
                                             lookat=(0, 0, 0.5), fov=30, GUI=True)

        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                           device=self.device, dtype=gs.tc_float)
        self.drone = self.scene.add_entity(gs.morphs.Drone(file=str(URDF_PATH)))
        self.scene.build(n_envs=self.num_envs)

        # PID de rate Genesis (cascade=1 : consignes ω rad/s + thrust %PWM → RPM)
        self.pid = CrazyfliePIDTorch(num_envs=self.num_envs, device=self.device)

        # masses par lien d'origine (URDF cf2x) -> permet de poser une masse TOTALE
        # cible en vol sans dérive (on repart toujours des valeurs initiales).
        self._link_masses0 = [float(l.get_mass()) for l in self.drone.links]
        self.mass = float(sum(self._link_masses0))   # masse Genesis simulée (kg)

        self.episode_length = 0
        self.reset()

    def set_total_mass(self, total_mass: float) -> float:
        """Pose la masse TOTALE simulée par Genesis à `total_mass` kg (en vol).

        On répartit la cible sur les liens NON fixes au prorata de leurs masses
        d'origine (set_mass est un no-op sur un lien fixe). `set_mass` rescale aussi
        l'inertie/invweight du lien ∝ masse → effet immédiat sur la dynamique. Sert à
        tester l'adaptation du finetuning à un changement de masse pendant le vol.

        Si `self.freeze_inertia` : on ne change QUE la masse inertielle (via le solveur,
        set_links_inertial_mass), en LAISSANT l'inertie + l'invweight d'origine. La
        translation (vertical/horizontal) répond à la nouvelle masse, mais la boucle de
        rate Genesis (PID à gains fixes → couple/inertie) se comporte comme à la masse
        nominale. Diagnostic : le pretrain est cinématique en rotation (omega appliqué
        direct, sans inertie) ; figer l'inertie isole si la dérive horizontale d'une
        politique pré-entraînée à masse ≠ vient bien du dé-tuning du rate par l'inertie."""
        links = self.drone.links
        m0 = self._link_masses0
        # On ne rescale que les liens PORTEURS (corps, >0.1 g) et non fixes : les hélices
        # ~1e-7 kg sont négligeables et set_mass refuse une masse < EPS -> on les laisse.
        scalable = [(l, m) for l, m in zip(links, m0)
                    if m > 1e-4 and not bool(getattr(l, "is_fixed", False))]
        base_scalable = sum(m for _, m in scalable)
        base_other = float(sum(m0)) - base_scalable
        if base_scalable <= 0.0:
            return self.mass
        ratio = max(total_mass - base_other, 1e-4) / base_scalable
        for l, m in scalable:
            target = m * ratio
            if self.freeze_inertia:
                # masse seule : set_links_inertial_mass ne touche NI inertie NI invweight
                # (contrairement à set_mass). On met aussi à jour le cache python pour que
                # get_mass()/self.mass restent cohérents.
                l._solver.set_links_inertial_mass(float(target), [l.idx])
                l._inertial_mass = np.asarray(target, dtype=np.float32)
            else:
                l.set_mass(target)
        self.mass = float(sum(float(l.get_mass()) for l in links))
        return self.mass

    # ───────────────── Genesis → DroneState (code obs réel) ─────────────────
    def _refresh_state(self):
        """Recopie l'état Genesis dans `DroneState.latest`, comme le ferait
        `DroneState.update_from_log` à partir des logs cflib du vrai drone."""
        pos = self.drone.get_pos().detach().cpu().numpy().reshape(-1)[:3]
        quat = self.drone.get_quat().detach().cpu().numpy().reshape(-1)[:4]   # (w,x,y,z)
        vel = self.drone.get_vel().detach().cpu().numpy().reshape(-1)[:3]     # repère monde
        ang_world = self.drone.get_ang()                                      # rad/s, repère monde
        gyro_body = transform_by_quat(ang_world, inv_quat(self.drone.get_quat()))
        gyro_body = gyro_body.detach().cpu().numpy().reshape(-1)[:3]

        self.state.latest['pos'] = pos.astype(np.float64)
        self.state.latest['vel'] = vel.astype(np.float64)
        self.state.latest['quat_wxyz'] = quat.astype(np.float64)
        # gyro : deg/s repère corps (sert au log/résiduel, pas à l'obs) — comme cflib
        self.state.latest['gyro'] = gyro_body.astype(np.float64) * P.RAD_TO_DEG

    def get_obs(self) -> torch.Tensor:
        """Obs 27-dim construite par le VRAI `DroneState.get_obs()` (1-D, float32)."""
        return self.state.get_obs()

    # ───────────────────────────── step ─────────────────────────────
    def step(self, action):
        """action (4,) SI = [thrust_N, ωx, ωy, ωz]. -> (obs, info).

        Reproduit `drone_controller._rl_step` + `_send_action` :
          1. clip aux bornes physiques, push dans le FIFO (obs des pas suivants) ;
          2. action SI → consigne de rate (cf_params) → PID rate Genesis → RPM.
        """
        if torch.is_tensor(action):
            action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        # même clip que controller._rl_step (RLPolicy clippe déjà, ceinture+bretelles)
        action_np = np.clip(action_np, P.ACTION_LOW, P.ACTION_HIGH)

        # FIFO du vrai DroneState (l'action pleine autorité entre dans l'obs)
        self.state.push_action(action_np)

        # ── analogue firmware : action SI → consigne rate + %PWM (cf _send_action) ──
        T_N = float(action_np[0])
        omega_cmd = np.clip(action_np[1:], -P.SEND_OMEGA_MAX, P.SEND_OMEGA_MAX)  # rad/s corps
        # Genesis doit RÉALISER la force commandée -> inverse calibré (pas la map cflib).
        pwm_pct = genesis_force_to_pwm_pct(T_N)

        # PID cascade=1 : [p_sp, q_sp, r_sp, thrust_%PWM] (consignes en rad/s)
        pid_actions = torch.tensor([[omega_cmd[0], omega_cmd[1], omega_cmd[2], pwm_pct]],
                                   device=self.device, dtype=gs.tc_float)
        dummy_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        # boucle interne haute fréquence (consigne tenue constante)
        for _ in range(self.substeps):
            quat = self.drone.get_quat()
            gyro_body = transform_by_quat(self.drone.get_ang(), inv_quat(quat))  # rad/s corps
            rpm = self.pid.update(dummy_euler, gyro_body, pid_actions, self.physics_dt, cascade=1)
            self.drone.set_propellers_rpm(rpm)
            if self.target is not None:
                self.target.set_pos(self._goal_t, zero_velocity=True)
            self.scene.step()

        self._refresh_state()
        self.episode_length += 1

        pos = self.state.latest['pos']
        info = {
            "pos": pos,
            "pos_error": float(np.linalg.norm(pos - self.goal)),
            "thrust_N": T_N,
            "pwm_pct": pwm_pct,
        }
        return self.get_obs(), info

    # ───────────────────────────── reset ─────────────────────────────
    def set_goal(self, goal):
        """Change la cible en vol (comme `DroneState.set_goal` côté réel)."""
        self.goal = np.asarray(goal, dtype=np.float64)
        self._goal_t = torch.tensor(self.goal, device=self.device, dtype=gs.tc_float).unsqueeze(0)
        self.state.set_goal(self.goal)

    def reset(self):
        # pose initiale : autour du goal, à plat (comme lotf_sim.reset, serré)
        pos = self.goal + 0.2 * (np.random.rand(3) - 0.5)
        pos_t = torch.tensor(pos, device=self.device, dtype=gs.tc_float).unsqueeze(0)

        self.drone.set_pos(pos_t, zero_velocity=True)
        self.drone.set_quat(self.base_init_quat.unsqueeze(0), zero_velocity=True)
        self.drone.zero_all_dofs_velocity()
        self.pid.reset_idx(torch.arange(self.num_envs, device=self.device))

        # FIFO réinitialisé à l'action de hover (réinstancie le vrai DroneState)
        self.state = DroneState(goal=self.goal)
        self.episode_length = 0

        self._refresh_state()
        return self.get_obs(), None
