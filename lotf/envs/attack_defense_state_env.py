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
    State of the attack-defense environment.

    Attributes:
        time: elapsed simulation time.
        step_idx: current step count in the episode.
        quadrotor_state: physical state of the defender drone.
        last_actions: history of defender actions (control latency).
        last_quadrotor_states: history of previous defender states.
        attacker_p: attacker position in world frame.
        attacker_v: attacker velocity in world frame.
        attacker_active: 1.0 if the attacker pursues this episode, else 0.0
            (an inactive attacker stays static = no threat).
    """
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    last_actions: jax.Array
    last_quadrotor_states: QuadrotorState
    attacker_p: jax.Array
    attacker_v: jax.Array
    attacker_active: jax.Array


class AttackDefenseStateEnv(env_base.Env[EnvState]):
    """
    Defender quadrotor that must hold a defended position while evading a
    point-mass attacker that is attracted to (homes in on) the defender.

    Only the defender is controlled by the trained policy. The attacker is a
    fully differentiable point-mass pursuer, so backprop-through-time gradients
    flow through the whole interaction.
    """

    def __init__(
        self,
        *,
        max_steps_in_episode=10000,
        dt=0.02,
        delay=0.02,
        yaw_scale=0.1,
        pitch_roll_scale=0.1,
        velocity_std=0.1,
        omega_std=0.1,
        quad_obj=None,
        reward_sharpness=1.0,
        action_penalty_weight=1.0,
        num_last_quad_states=10,
        margin=0.0,
        hover_height=1.0,
        hover_target=None,
        # attacker / interaction parameters
        catch_radius=0.3,
        evasion_weight=2.0,
        evasion_scale=0.6,
        attacker_v_max=4.0,
        attacker_a_max=8.0,
        attacker_kp=6.0,
        attacker_kd=3.0,
        attacker_spawn_r_min=1.0,
        attacker_spawn_r_max=1.5,
        attacker_active_prob=0.5,
    ):
        """Initializes environment, defender physics, defended target, and attacker."""

        # defended position (origin of the target frame)
        self.goal: jnp.ndarray = jnp.array([0.0, 0.0, hover_height])
        if hover_target is not None:
            self.goal = jnp.array(hover_target, dtype=jnp.float32)

        # world boundaries for collision and normalization
        side_length = 3.0
        self.world_box = WorldBox(
            self.goal - side_length / 2,
            self.goal + side_length / 2
        )

        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)

        # randomized initial-state distribution parameters
        self.yaw_scale = yaw_scale
        self.pitch_roll_scale = pitch_roll_scale
        self.velocity_std = velocity_std
        self.omega_std = omega_std

        # defender quadrotor model
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

        # reference action for steady hover
        thrust_hover = 9.81 * self.quadrotor._mass
        self.hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])

        self.num_last_quad_states = num_last_quad_states
        self.margin = margin

        # attacker / interaction parameters
        self.catch_radius = catch_radius
        self.evasion_weight = evasion_weight
        self.evasion_scale = evasion_scale
        self.attacker_v_max = attacker_v_max
        self.attacker_a_max = attacker_a_max
        self.attacker_kp = attacker_kp
        self.attacker_kd = attacker_kd
        self.attacker_spawn_r_min = attacker_spawn_r_min
        self.attacker_spawn_r_max = attacker_spawn_r_max
        self.attacker_active_prob = attacker_active_prob

        # bound on relative-vector observation components (finite for MinMax wrapper)
        self._rel_pos_max = float(jnp.linalg.norm(self.world_box.max - self.world_box.min))
        self._rel_vel_max = float(jnp.max(self.v_max)) + attacker_v_max

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key, state: Optional[EnvState] = None
    ) -> tuple[EnvState, jax.Array]:
        """Resets defender pose/velocity and spawns the attacker on a random shell."""

        key_p, key_R, key_v, key_omega, key_dr, key_adir, key_ar, key_act = jax.random.split(key, 8)

        # randomize defender starting position within the world box
        p = jax.random.uniform(
            key_p,
            shape=(3,),
            minval=self.world_box.min + self.margin,
            maxval=self.world_box.max - self.margin,
        )

        # randomize defender orientation
        rot = random_rotation(
            key_R, self.yaw_scale, self.pitch_roll_scale, self.pitch_roll_scale
        )
        R = rot.as_matrix()

        # randomize defender linear and angular velocities
        v = self.velocity_std * jax.random.normal(key_v, shape=(3,))
        omega = self.omega_std * jax.random.normal(key_omega, shape=(3,))

        quadrotor_state = self.quadrotor.create_state(
            p=p, R=R, v=v, omega=omega, dr_key=key_dr
        )

        # spawn attacker on a random shell around the defended position so it
        # never starts already touching the defender
        direction = jax.random.normal(key_adir, shape=(3,))
        direction = direction / (jnp.linalg.norm(direction) + 1e-8)
        radius = jax.random.uniform(
            key_ar, shape=(), minval=self.attacker_spawn_r_min, maxval=self.attacker_spawn_r_max
        )
        attacker_p = self.goal + direction * radius
        attacker_v = jnp.zeros(3)

        # threat is intermittent: some episodes have an inactive (static) attacker
        # so the policy learns to simply hold the defended position when safe.
        attacker_active = (
            jax.random.uniform(key_act, shape=()) < self.attacker_active_prob
        ).astype(jnp.float32)

        # history buffers
        last_actions = jnp.tile(self.hovering_action, (self.num_last_actions, 1))
        last_quadrotor_states = stack_pytrees([quadrotor_state] * self.num_last_quad_states)

        state = EnvState(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quadrotor_states,
            attacker_p=attacker_p,
            attacker_v=attacker_v,
            attacker_active=attacker_active,
        )

        return state, self._get_obs(state)

    def _get_obs(self, state: EnvState) -> jax.Array:
        """Builds the defender observation in the target / body frames."""
        p = state.quadrotor_state.p
        R = state.quadrotor_state.R
        v = state.quadrotor_state.v

        # defender position expressed in the (translation-only) target frame
        p_in_target = p - self.goal

        # attacker position and velocity in the defender body frame
        attacker_rel_pos_body = R.T @ (state.attacker_p - p)
        attacker_rel_vel_body = R.T @ (state.attacker_v - v)

        return jnp.concatenate(
            [
                p_in_target,
                math_utils.vec(R),
                v,
                attacker_rel_pos_body,
                attacker_rel_vel_body,
                state.last_actions.flatten(),
            ]
        )

    def _attacker_step(self, attacker_p, attacker_v, defender_p, active):
        """Differentiable point-mass pursuer attracted to the defender (Euler).

        When `active` is 0.0 the attacker stays static (no threat).
        """
        # damped spring attraction toward the defender
        a_cmd = self.attacker_kp * (defender_p - attacker_p) - self.attacker_kd * attacker_v
        a_cmd = _clip_norm(a_cmd, self.attacker_a_max) * active

        v_new = _clip_norm(attacker_v + a_cmd * self.dt, self.attacker_v_max) * active
        p_new = attacker_p + v_new * self.dt
        return p_new, v_new

    @partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: EnvState, action: jax.Array, res_model_params: FrozenDict, key: chex.PRNGKey
    ) -> EnvTransition:
        """Advances defender physics (with delay) and the attacker pursuit."""

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

        # advance the attacker toward the (updated) defender position
        attacker_p, attacker_v = self._attacker_step(
            state.attacker_p, state.attacker_v, quadrotor_state.p, state.attacker_active
        )

        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            attacker_p=attacker_p,
            attacker_v=attacker_v,
        )

        obs = self._get_obs(next_state)
        reward = self._get_reward(state, next_state)
        terminated = self._is_done(next_state)
        truncated = jnp.greater_equal(next_state.step_idx, self.max_steps_in_episode)

        return EnvTransition(next_state, obs, reward, terminated, truncated, dict())

    def _get_reward(
        self, last_state: EnvState, next_state: EnvState
    ) -> jax.Array:
        """Reward = hold the defended position, stay smooth, and evade the attacker."""

        action = next_state.last_actions[-1]
        p = next_state.quadrotor_state.p
        acc = next_state.quadrotor_state.acc

        # hold the defended position (target-frame origin)
        pos_cost = (
            smooth_l1(self.reward_sharpness * (p - self.goal))
            / self.reward_sharpness
        )
        vel_cost = 0.1 * smooth_l1(next_state.quadrotor_state.v)
        omega_cost = 0.1 * smooth_l1(next_state.quadrotor_state.omega)
        acc_cost = 0.1 * smooth_l1(acc)
        goal_cost = pos_cost + vel_cost + omega_cost + acc_cost

        # penalize excessive control effort
        action_cost = self.action_penalty_weight * smooth_l1(action - self.hovering_action)

        # smooth evasion shaping: high cost when the attacker is close, decaying
        # with distance. Provides a usable gradient before an actual touch.
        dist = jnp.linalg.norm(next_state.attacker_p - p)
        evasion_cost = self.evasion_weight * jnp.exp(-(dist / self.evasion_scale) ** 2)

        cost = goal_cost + action_cost + evasion_cost

        # massive terminal penalty on touch or leaving the arena
        time_left = self.max_steps_in_episode - next_state.step_idx
        done_cost = jax.lax.select(
            self._is_done(next_state), time_left * cost, 0.0
        )
        cost += jax.lax.stop_gradient(done_cost)

        return -self.dt * cost

    def _is_touched(self, state: EnvState) -> jax.Array:
        """True if the attacker is within the catch radius of the defender."""
        dist = jnp.linalg.norm(state.attacker_p - state.quadrotor_state.p)
        return jnp.less(dist, self.catch_radius)

    def _is_colliding(self, state: EnvState) -> jax.Array:
        """True if the defender leaves the world box."""
        return jnp.logical_not(self.world_box.contains(state.quadrotor_state.p))

    def _is_done(self, state: EnvState) -> jax.Array:
        """Episode ends on touch or when the defender leaves the arena."""
        return jnp.logical_or(self._is_touched(state), self._is_colliding(state))

    @property
    def action_space(self) -> spaces.Box:
        """Returns the bounds for thrust and angular velocity."""
        low = jnp.concatenate([jnp.array([self.thrust_min * 4]), self.omega_min])
        high = jnp.concatenate([jnp.array([self.thrust_max * 4]), self.omega_max])
        return spaces.Box(low, high, shape=(4,))

    @property
    def observation_space(self) -> spaces.Box:
        """Returns bounds for target-frame state, attacker rel pose, and buffered actions."""
        n = self.num_last_actions
        action_high_repeated = jnp.concatenate([self.action_space.high] * n)
        action_low_repeated = jnp.concatenate([self.action_space.low] * n)

        rel_pos_high = jnp.full(3, self._rel_pos_max)
        rel_vel_high = jnp.full(3, self._rel_vel_max)

        low = jnp.concatenate([
            self.world_box.min - self.goal,    # p in target frame
            -jnp.ones(9),                      # R
            self.v_min,                        # v
            -rel_pos_high,                     # attacker pos (body frame)
            -rel_vel_high,                     # attacker vel (body frame)
            action_low_repeated,
        ])
        high = jnp.concatenate([
            self.world_box.max - self.goal,
            jnp.ones(9),
            self.v_max,
            rel_pos_high,
            rel_vel_high,
            action_high_repeated,
        ])

        return spaces.Box(low=low, high=high, shape=(21 + n * 4,))

    @classmethod
    def generate_csv(cls, traj: EnvTransition, filename: str):
        """Exports trajectory data (defender + attacker) for playback/analysis."""
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
                "vx", "vy", "vz", "ax", "ay", "az",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(traj_length):
                transition = pytree_get_item(traj, i)
                t = transition.state.time
                p = transition.state.quadrotor_state.p
                R = transition.state.quadrotor_state.R
                quat = rot_to_quat(Rotation.from_matrix(R))
                ap = transition.state.attacker_p
                writer.writerow({
                    "index": i, "t": t, "px": p[0], "py": p[1], "pz": p[2],
                    "qw": quat[0], "qx": quat[1], "qy": quat[2], "qz": quat[3],
                    "vx": transition.state.quadrotor_state.v[0],
                    "vy": transition.state.quadrotor_state.v[1],
                    "vz": transition.state.quadrotor_state.v[2],
                    "ax": ap[0], "ay": ap[1], "az": ap[2],
                })

    def plot_trajectories(self, traj: EnvTransition):
        """Visualizes defender vs attacker paths in the XY plane."""
        from matplotlib import pyplot as plt
        import seaborn as sns

        num_trajs = traj.reward.shape[0]
        state: EnvState = traj.state
        done = np.logical_or(traj.terminated, traj.truncated)

        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        def_color = '#2c3e50'
        att_color = '#e74c3c'
        goal_color = '#27ae60'

        for i in range(num_trajs):
            idx = np.where(done[i])[0][0].item() + 1
            dx, dy = state.quadrotor_state.p[i, :idx, 0], state.quadrotor_state.p[i, :idx, 1]
            ax, ay = state.attacker_p[i, :idx, 0], state.attacker_p[i, :idx, 1]
            t = state.time[i, :idx]
            dist = np.linalg.norm(state.attacker_p[i, :idx] - state.quadrotor_state.p[i, :idx], axis=-1)

            ax1.plot(dx, dy, color=def_color, alpha=0.6, linewidth=1.5, label="Defender" if i == 0 else "")
            ax1.plot(ax, ay, color=att_color, alpha=0.6, linewidth=1.5, label="Attacker" if i == 0 else "")
            ax2.plot(t, dist, color=def_color, linewidth=1.2, alpha=0.6)

        ax1.scatter(self.goal[0], self.goal[1], color=goal_color, s=80, marker='*',
                    edgecolors='white', zorder=5, label="Defended pos")
        ax2.axhline(self.catch_radius, color=att_color, linestyle='--', label="catch radius")

        ax1.set_title("XY Paths", fontweight='bold')
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
        ax1.set_aspect("equal", adjustable="box")
        ax1.legend(loc='best', fontsize='small')

        ax2.set_title("Attacker-Defender Distance", fontweight='bold')
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Distance (m)")
        ax2.legend(loc='best', fontsize='small')

        sns.despine()
        plt.tight_layout()
        plt.show()


def _clip_norm(vec: jax.Array, max_norm: float) -> jax.Array:
    """Scales a vector down so its norm does not exceed max_norm (differentiable).

    Uses a soft norm (sqrt of sum-of-squares + eps) so the gradient stays finite
    at vec=0 (a plain jnp.linalg.norm has a NaN gradient at the origin).
    """
    norm = jnp.sqrt(jnp.sum(vec * vec) + 1e-12)
    scale = jnp.minimum(1.0, max_norm / norm)
    return vec * scale
