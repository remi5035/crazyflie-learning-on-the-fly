import csv
from functools import partial
from typing import Optional
import numpy as np

import chex
import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation
from flax.core import FrozenDict

from lotf.objects import Quadrotor, QuadrotorState, WorldBox
from lotf.utils import math as math_utils
from lotf.utils import spaces
from lotf.utils.pytrees import pytree_get_item, stack_pytrees
from lotf.utils.math import smooth_l1, rot_to_quat
from lotf.utils.random import random_rotation
import lotf.envs.env_base as env_base
from lotf.envs.env_base import EnvTransition


@jdc.pytree_dataclass
class EnvState(env_base.EnvState):
    """
    State of the vision-based pursuit environment.

    Attributes:
        time: elapsed simulation time.
        step_idx: current step count in the episode.
        quadrotor_state: physical state of the pursuer (attacker) drone.
        last_actions: history of pursuer actions (control latency).
        target_p: world position of the (static) chased target.
        last_seen_p: last world position at which the target was observed by
            the camera. Carried as the pursuer's only memory of the target.
        steps_since_seen: number of steps since the target was last in view.
        ever_seen: 1.0 once the target has been seen at least once, else 0.0.
    """
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    last_actions: jax.Array
    target_p: jax.Array
    last_seen_p: jax.Array
    steps_since_seen: jax.Array
    ever_seen: jax.Array


class VisionPursuitStateEnv(env_base.Env[EnvState]):
    """
    Pursuer (attacker) quadrotor that must catch a static target it can only
    perceive through an onboard camera.

    The policy never receives the absolute target position. Instead the camera
    is modeled as a simple pinhole FOV cone with a finite usable range: when the
    target falls inside the field of view and within `cam_max_range` metres, the
    observation contains the target's normalized image coordinates (its position
    in the camera frame) plus a range cue. Otherwise the target is "lost" and
    the policy must rely on memory features (last-seen bearing + time since seen)
    to reacquire it.

    Only the pursuer is controlled. The whole environment is differentiable, so
    backprop-through-time gradients flow through the interaction. The reward uses
    the privileged true target position (asymmetric training): the policy still
    only ever observes the camera, so it must learn vision-based pursuit.
    """

    def __init__(
        self,
        *,
        max_steps_in_episode=10000,
        dt=0.02,
        delay=0.02,
        yaw_scale=1.0,
        pitch_roll_scale=0.1,
        velocity_std=0.1,
        omega_std=0.1,
        quad_obj=None,
        reward_sharpness=1.0,
        action_penalty_weight=0.5,
        margin=0.5,
        side_length=10.0,
        # camera parameters
        cam_fov_h_deg=90.0,
        cam_fov_v_deg=70.0,
        cam_pitch_deg=15.0,
        cam_max_range=10.0,
        visible_sharpness=10.0,
        # target / interaction parameters
        catch_radius=0.4,
        catch_scale=1.5,
        catch_weight=4.0,
        center_weight=0.5,
        visible_weight=0.5,
        search_weight=1.0,
        success_weight=1.0,
        target_spawn_r_min=2.0,
        target_spawn_r_max=8.0,
        seen_time_tau=1.0,
        # spawn orientation
        init_yaw_toward_target=True,
        yaw_noise_scale=0.1,
    ):
        """Initializes environment, pursuer physics, camera model, and target."""

        # arena centred on the origin
        self.center = jnp.array([0.0, 0.0, 1.5])
        self.world_box = WorldBox(
            self.center - side_length / 2,
            self.center + side_length / 2,
        )

        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)

        # randomized initial-state distribution parameters
        self.yaw_scale = yaw_scale
        self.pitch_roll_scale = pitch_roll_scale
        self.velocity_std = velocity_std
        self.omega_std = omega_std

        # pursuer quadrotor model
        if quad_obj is not None:
            self.quadrotor = quad_obj
        else:
            self.quadrotor = Quadrotor.default_quadrotor()

        # physical constraints for normalization and clipping
        self.omega_min = self.quadrotor._omega_max * -1
        self.omega_max = self.quadrotor._omega_max
        self.thrust_min = self.quadrotor._thrust_min
        self.thrust_max = self.quadrotor._thrust_max
        self.v_min = jnp.array([-5.0, -5.0, -5.0])
        self.v_max = jnp.array([5.0, 5.0, 5.0])

        # action buffer size for delay simulation
        assert delay >= 0.0, "Delay must be non-negative"
        self.delay = np.array(delay)
        self.num_last_actions = int(np.ceil(delay / dt)) + 1

        self.reward_sharpness = reward_sharpness
        self.action_penalty_weight = action_penalty_weight
        self.margin = margin

        # reference action for steady hover
        thrust_hover = 9.81 * self.quadrotor._mass
        self.hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])

        # camera intrinsics / extrinsics (pinhole FOV cone)
        self.cam_max_range = cam_max_range
        self.visible_sharpness = visible_sharpness
        self.cam_half_tan_h = float(np.tan(np.radians(cam_fov_h_deg) / 2))
        self.cam_half_tan_v = float(np.tan(np.radians(cam_fov_v_deg) / 2))
        # camera optical axis (z_c) in body frame: forward (+x) tilted down by pitch
        pitch = np.radians(cam_pitch_deg)
        z_c = jnp.array([np.cos(pitch), 0.0, -np.sin(pitch)])
        x_c = jnp.array([0.0, 1.0, 0.0])             # image right = body +y
        y_c = jnp.cross(z_c, x_c)                     # image vertical axis
        # R_BC has the camera axes as columns (camera -> body); R_CB maps body -> camera
        self.R_BC = jnp.stack([x_c, y_c, z_c], axis=1)
        self.R_CB = self.R_BC.T
        self.cam_boresight_body = z_c

        # target / interaction parameters
        self.catch_radius = catch_radius
        self.catch_scale = catch_scale
        self.catch_weight = catch_weight
        self.center_weight = center_weight
        self.visible_weight = visible_weight
        self.search_weight = search_weight
        self.success_weight = success_weight
        self.target_spawn_r_min = target_spawn_r_min
        self.target_spawn_r_max = target_spawn_r_max
        self.seen_time_tau = seen_time_tau
        self.init_yaw_toward_target = init_yaw_toward_target
        self.yaw_noise_scale = yaw_noise_scale

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key, state: Optional[EnvState] = None
    ) -> tuple[EnvState, jax.Array]:
        """Spawns a static target and the pursuer on a shell around it."""

        key_tp, key_R, key_v, key_omega, key_dr, key_dir, key_r = jax.random.split(key, 7)

        # static target somewhere in the inner arena
        target_p = jax.random.uniform(
            key_tp,
            shape=(3,),
            minval=self.world_box.min + self.margin,
            maxval=self.world_box.max - self.margin,
        )

        # spawn the pursuer on a random shell around the target so the initial
        # distance is controlled (and often beyond camera range / out of FOV)
        direction = jax.random.normal(key_dir, shape=(3,))
        direction = direction / (jnp.linalg.norm(direction) + 1e-8)
        radius = jax.random.uniform(
            key_r, shape=(), minval=self.target_spawn_r_min, maxval=self.target_spawn_r_max
        )
        p = target_p + direction * radius
        p = jnp.clip(
            p, self.world_box.min + self.margin, self.world_box.max - self.margin
        )

        # initial heading: optionally bias yaw toward the target so the camera
        # sees it from the start (the policy can then learn to pursue before
        # learning to search).  A yaw_noise_scale of 0 = always facing target;
        # 1.0 = effectively random (falls back to the old behaviour).
        key_yaw_n, key_pitch_n, key_roll_n = jax.random.split(key_R, 3)
        if self.init_yaw_toward_target:
            to_target = target_p - p
            yaw_base = jnp.arctan2(to_target[1], to_target[0])
            yaw_noise = self.yaw_noise_scale * jax.random.uniform(
                key_yaw_n, minval=-jnp.pi, maxval=jnp.pi
            )
            yaw = yaw_base + yaw_noise
        else:
            yaw = self.yaw_scale * jax.random.uniform(
                key_yaw_n, minval=-jnp.pi, maxval=jnp.pi
            )
        pitch = self.pitch_roll_scale * jax.random.uniform(
            key_pitch_n, minval=-jnp.pi, maxval=jnp.pi
        )
        roll = self.pitch_roll_scale * jax.random.uniform(
            key_roll_n, minval=-jnp.pi, maxval=jnp.pi
        )
        rot = Rotation.from_euler("zyx", jnp.array([yaw, pitch, roll]))
        R = rot.as_matrix()

        v = self.velocity_std * jax.random.normal(key_v, shape=(3,))
        omega = self.omega_std * jax.random.normal(key_omega, shape=(3,))

        quadrotor_state = self.quadrotor.create_state(
            p=p, R=R, v=v, omega=omega, dr_key=key_dr
        )

        last_actions = jnp.tile(self.hovering_action, (self.num_last_actions, 1))

        state = EnvState(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            target_p=target_p,
            last_seen_p=p,                 # placeholder until first sighting
            steps_since_seen=jnp.array(0.0),
            ever_seen=jnp.array(0.0),
        )

        return state, self._get_obs(state)

    def _camera_view(self, state: EnvState):
        """Projects the target into the pinhole camera and gates by FOV + range.

        Returns (visible, img_x_n, img_y_n, norm_dist) where the image
        coordinates are normalized to [-1, 1] inside the field of view.
        """
        p = state.quadrotor_state.p
        R = state.quadrotor_state.R

        rel_world = state.target_p - p
        dist = jnp.linalg.norm(rel_world) + 1e-8

        # target direction expressed in the camera frame
        d_body = R.T @ rel_world
        d_cam = self.R_CB @ d_body

        # pinhole projection (guard the optical-axis denominator)
        in_front = d_cam[2] > 1e-3
        denom = jnp.where(in_front, d_cam[2], 1.0)
        img_x = d_cam[0] / denom
        img_y = d_cam[1] / denom
        img_x_n = img_x / self.cam_half_tan_h
        img_y_n = img_y / self.cam_half_tan_v

        # soft visibility: product of smooth gates on each FOV/range condition
        # so the gradient flows through `visible` (and thus p and R) during BPTT.
        # All arguments are in normalized (order-1) units so a single sharpness
        # gives a comparable transition band on each boundary.
        k = self.visible_sharpness
        gate_front = jax.nn.sigmoid(k * (d_cam[2] / dist))
        gate_x = jax.nn.sigmoid(k * (1.0 - jnp.abs(img_x_n)))
        gate_y = jax.nn.sigmoid(k * (1.0 - jnp.abs(img_y_n)))
        gate_dist = jax.nn.sigmoid(k * (1.0 - dist / self.cam_max_range))
        visible = gate_front * gate_x * gate_y * gate_dist

        # zero out the readout when the target is not visible
        img_x_n = visible * img_x_n
        img_y_n = visible * img_y_n
        norm_dist = visible * jnp.clip(dist / self.cam_max_range, 0.0, 1.0)

        return visible, img_x_n, img_y_n, norm_dist

    def _get_obs(self, state: EnvState) -> jax.Array:
        """Builds the camera-only observation plus memory features."""
        R = state.quadrotor_state.R
        v = state.quadrotor_state.v

        # own state the pursuer can estimate onboard (VIO/IMU): body velocity + attitude
        v_body = R.T @ v

        # camera readout of the target
        visible, img_x_n, img_y_n, norm_dist = self._camera_view(state)
        cam = jnp.array([visible, img_x_n, img_y_n, norm_dist])

        # memory: bearing to the last-seen position in the current body frame
        to_last_seen = state.last_seen_p - state.quadrotor_state.p
        bearing_body = R.T @ (to_last_seen / (jnp.linalg.norm(to_last_seen) + 1e-8))
        bearing_body = state.ever_seen * bearing_body
        norm_t_since = jnp.tanh(state.steps_since_seen * self.dt / self.seen_time_tau)
        memory = jnp.concatenate(
            [bearing_body, jnp.array([norm_t_since, state.ever_seen])]
        )

        return jnp.concatenate(
            [
                v_body,
                math_utils.vec(R),
                cam,
                memory,
                state.last_actions.flatten(),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: EnvState, action: jax.Array, res_model_params: FrozenDict, key: chex.PRNGKey
    ) -> EnvTransition:
        """Advances pursuer physics (with delay) and updates the camera memory."""

        action = jnp.clip(
            action, self.action_space.low, self.action_space.high
        )

        # update action buffer for delay simulation
        last_actions = jnp.roll(state.last_actions, shift=-1, axis=0)
        last_actions = last_actions.at[-1].set(action)

        # first (fractional) integration step caused by delay
        dt_1 = self.delay - (self.num_last_actions - 2) * self.dt
        action_1 = last_actions[0]
        f_1, omega_1 = action_1[0], action_1[1:]
        quadrotor_state = self.quadrotor.step(
            state.quadrotor_state, f_1, omega_1, res_model_params, dt_1
        )

        # complete the remaining time step with the subsequent action
        if dt_1 < self.dt:
            dt_2 = self.dt - dt_1
            action_2 = last_actions[1]
            f_2, omega_2 = action_2[0], action_2[1:]
            quadrotor_state = self.quadrotor.step(
                quadrotor_state, f_2, omega_2, res_model_params, dt_2
            )

        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
        )

        # update camera memory after the physics step (soft so gradients flow
        # through `visible` and therefore through p and R during BPTT)
        visible, _, _, _ = self._camera_view(next_state)
        last_seen_p = visible * next_state.target_p + (1.0 - visible) * state.last_seen_p
        steps_since_seen = (1.0 - visible) * (state.steps_since_seen + 1.0)
        ever_seen = jnp.maximum(state.ever_seen, visible)
        next_state = next_state.replace(
            last_seen_p=last_seen_p,
            steps_since_seen=steps_since_seen,
            ever_seen=ever_seen,
        )

        obs = self._get_obs(next_state)
        reward = self._get_reward(state, next_state)
        terminated = self._is_done(next_state)
        truncated = jnp.greater_equal(next_state.step_idx, self.max_steps_in_episode)

        return EnvTransition(next_state, obs, reward, terminated, truncated, dict())

    def _get_reward(
        self, last_state: EnvState, next_state: EnvState
    ) -> jax.Array:
        """Reward = rush at the target when seen, search for it when lost."""

        action = next_state.last_actions[-1]
        p = next_state.quadrotor_state.p
        R = next_state.quadrotor_state.R
        omega = next_state.quadrotor_state.omega
        target_p = next_state.target_p

        dist = jnp.linalg.norm(target_p - p)

        # dense pull toward the target (privileged shaping, smooth gradient)
        approach_cost = (
            smooth_l1(self.reward_sharpness * (p - target_p))
            / self.reward_sharpness
        )

        # strong attractor near the target for a clean "rush in" gradient
        proximity_reward = self.catch_weight * jnp.exp(-(dist / self.catch_scale) ** 2)

        visible, img_x_n, img_y_n, _ = self._camera_view(next_state)

        # when visible: keep the target centered in the frame and reward seeing it
        center_cost = visible * self.center_weight * (img_x_n ** 2 + img_y_n ** 2)
        visible_reward = self.visible_weight * visible

        # when lost: align the camera boresight with the last-seen bearing to reacquire
        boresight_world = R @ self.cam_boresight_body
        to_last_seen = next_state.last_seen_p - p
        dir_last_seen = to_last_seen / (jnp.linalg.norm(to_last_seen) + 1e-8)
        align = jnp.clip(jnp.dot(boresight_world, dir_last_seen), -1.0, 1.0)
        search_cost = (
            (1.0 - visible) * next_state.ever_seen
            * self.search_weight * (1.0 - align)
        )

        # control smoothness
        action_cost = self.action_penalty_weight * smooth_l1(action - self.hovering_action)
        omega_cost = 0.1 * smooth_l1(omega)

        cost = approach_cost + center_cost + search_cost + action_cost + omega_cost
        reward = proximity_reward + visible_reward - cost

        # terminal shaping (no gradient): reward fast capture, penalize leaving
        time_left = self.max_steps_in_episode - next_state.step_idx
        caught = self._is_caught(next_state)
        left = self._is_colliding(next_state)
        success_bonus = jax.lax.select(caught, time_left * self.success_weight, 0.0)
        leave_penalty = jax.lax.select(left, time_left * cost, 0.0)
        reward += jax.lax.stop_gradient(success_bonus - leave_penalty)

        return self.dt * reward

    def _is_caught(self, state: EnvState) -> jax.Array:
        """True if the pursuer is within the catch radius of the target."""
        dist = jnp.linalg.norm(state.target_p - state.quadrotor_state.p)
        return jnp.less(dist, self.catch_radius)

    def _is_colliding(self, state: EnvState) -> jax.Array:
        """True if the pursuer leaves the world box."""
        return jnp.logical_not(self.world_box.contains(state.quadrotor_state.p))

    def _is_done(self, state: EnvState) -> jax.Array:
        """Episode ends on capture or when the pursuer leaves the arena."""
        return jnp.logical_or(self._is_caught(state), self._is_colliding(state))

    @property
    def action_space(self) -> spaces.Box:
        """Returns the bounds for thrust and angular velocity."""
        low = jnp.concatenate([jnp.array([self.thrust_min * 4]), self.omega_min])
        high = jnp.concatenate([jnp.array([self.thrust_max * 4]), self.omega_max])
        return spaces.Box(low, high, shape=(4,))

    @property
    def observation_space(self) -> spaces.Box:
        """Returns bounds for body-frame state, camera readout, memory, and actions."""
        n = self.num_last_actions
        action_high_repeated = jnp.concatenate([self.action_space.high] * n)
        action_low_repeated = jnp.concatenate([self.action_space.low] * n)

        low = jnp.concatenate([
            self.v_min,                 # body-frame velocity
            -jnp.ones(9),               # R
            jnp.array([0.0, -1.0, -1.0, 0.0]),   # cam: visible, img_x_n, img_y_n, norm_dist
            -jnp.ones(3),               # last-seen bearing (body)
            jnp.array([0.0, 0.0]),      # norm time since seen, ever_seen
            action_low_repeated,
        ])
        high = jnp.concatenate([
            self.v_max,
            jnp.ones(9),
            jnp.array([1.0, 1.0, 1.0, 1.0]),
            jnp.ones(3),
            jnp.array([1.0, 1.0]),
            action_high_repeated,
        ])

        return spaces.Box(low=low, high=high, shape=(21 + n * 4,))

    @classmethod
    def generate_csv(cls, traj: EnvTransition, filename: str):
        """Exports trajectory data (pursuer + target) for playback/analysis."""
        from tqdm import tqdm
        num_trajectories = traj.reward.shape[0] if traj.reward.ndim > 1 else 1
        for i in tqdm(range(num_trajectories)):
            traj_i = pytree_get_item(traj, i) if num_trajectories > 1 else traj
            cls._generate_csv(traj_i, f"{filename}_{i}.csv")

    @staticmethod
    def _generate_csv(traj: EnvTransition, filename: str):
        """Writes a single trajectory to a csv file."""
        done = jnp.logical_or(traj.terminated, traj.truncated)
        traj_length = jnp.where(done)[0][0].item() + 1

        with open(filename, "w", newline="") as csvfile:
            fieldnames = [
                "index", "t", "px", "py", "pz", "qw", "qx", "qy", "qz",
                "vx", "vy", "vz", "tx", "ty", "tz",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(traj_length):
                transition = pytree_get_item(traj, i)
                t = transition.state.time
                p = transition.state.quadrotor_state.p
                R = transition.state.quadrotor_state.R
                quat = rot_to_quat(Rotation.from_matrix(R))
                tp = transition.state.target_p
                writer.writerow({
                    "index": i, "t": t, "px": p[0], "py": p[1], "pz": p[2],
                    "qw": quat[0], "qx": quat[1], "qy": quat[2], "qz": quat[3],
                    "vx": transition.state.quadrotor_state.v[0],
                    "vy": transition.state.quadrotor_state.v[1],
                    "vz": transition.state.quadrotor_state.v[2],
                    "tx": tp[0], "ty": tp[1], "tz": tp[2],
                })

    def plot_trajectories(self, traj: EnvTransition):
        """Visualizes pursuer paths and target positions in the XY plane."""
        from matplotlib import pyplot as plt
        import seaborn as sns

        num_trajs = traj.reward.shape[0]
        state: EnvState = traj.state
        done = np.logical_or(traj.terminated, traj.truncated)

        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        pur_color = '#2c3e50'
        tgt_color = '#e74c3c'

        for i in range(num_trajs):
            has_done = np.any(done[i])
            idx = np.where(done[i])[0][0].item() + 1 if has_done else done.shape[1]
            px, py = state.quadrotor_state.p[i, :idx, 0], state.quadrotor_state.p[i, :idx, 1]
            tx, ty = state.target_p[i, :idx, 0], state.target_p[i, :idx, 1]
            t = state.time[i, :idx]
            dist = np.linalg.norm(state.target_p[i, :idx] - state.quadrotor_state.p[i, :idx], axis=-1)

            ax1.plot(px, py, color=pur_color, alpha=0.6, linewidth=1.5, label="Pursuer" if i == 0 else "")
            ax1.scatter(tx[0], ty[0], color=tgt_color, s=60, marker='*',
                        edgecolors='white', zorder=5, label="Target" if i == 0 else "")
            ax2.plot(t, dist, color=pur_color, linewidth=1.2, alpha=0.6)

        ax2.axhline(self.catch_radius, color=tgt_color, linestyle='--', label="catch radius")
        ax2.axhline(self.cam_max_range, color='#8e44ad', linestyle=':', label="camera range")

        ax1.set_title("XY Paths", fontweight='bold')
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
        ax1.set_aspect("equal", adjustable="box")
        ax1.legend(loc='best', fontsize='small')

        ax2.set_title("Pursuer-Target Distance", fontweight='bold')
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Distance (m)")
        ax2.legend(loc='best', fontsize='small')

        sns.despine()
        plt.tight_layout()
        plt.show()
