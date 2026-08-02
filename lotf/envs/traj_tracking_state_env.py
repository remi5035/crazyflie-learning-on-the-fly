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

from lotf.objects import Quadrotor, QuadrotorState, WorldBox, ReferenceTraj, RefTrajNames, TrajColumns
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
    State representation for the trajectory tracking environment.
    
    Attributes:
        time: elapsed simulation time.
        step_idx: current step count in the episode.
        quadrotor_state: physical state of the drone.
        last_actions: history of actions used to simulate control latency.
        last_quadrotor_states: history of previous physical states.
        init_ref_traj_idx: starting index on the reference trajectory.
    """
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    last_actions: jax.Array
    last_quadrotor_states: QuadrotorState
    init_ref_traj_idx: int = 0


class TrajTrackingStateEnv(env_base.Env[EnvState]):
    """State-based environment for tracking arbitrary, given trajectories."""

    def __init__(
        self,
        *,
        max_steps_in_episode=10000,
        dt=0.02,
        delay=0.02,
        yaw_scale=0.1,
        pitch_roll_scale=0.1,
        position_std=0.1,
        velocity_std=0.1,
        omega_std=0.1,
        quad_obj=None,
        num_last_quad_states=10,
        ref_traj_name: str = RefTrajNames.CIRCLE.value,
        from_start=False,
        skip_start=False,
        train_from_start=False,
    ):
        """Initialize the trajectory tracking environment."""
        self.world_box = WorldBox(
            jnp.array([-5.0, -5.0, 0.0]), jnp.array([5.0, 5.0, 3.0])
        )
        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)
        
        # noise parameters for initialization
        self.yaw_scale = yaw_scale
        self.pitch_roll_scale = pitch_roll_scale
        self.position_std = position_std
        self.velocity_std = velocity_std
        self.omega_std = omega_std

        # quadrotor physics setup
        if quad_obj is not None:
            self.quadrotor = quad_obj
        else:
            self.quadrotor = Quadrotor.default_quadrotor()

        self.omega_min = self.quadrotor._omega_max * -1
        self.omega_max = self.quadrotor._omega_max
        self.thrust_min = self.quadrotor._thrust_min
        self.thrust_max = self.quadrotor._thrust_max

        # control delay handling
        assert delay >= 0.0, "delay must be non-negative"
        self.delay = np.array(delay)
        self.num_last_actions = int(np.ceil(delay / dt)) + 1

        thrust_hover = 9.81 * self.quadrotor._mass
        self.hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])

        self.num_last_quad_states = num_last_quad_states

        # load reference trajectory data
        ref_traj_obj = ReferenceTraj.from_name(ref_traj_name)
        self.ref_traj = ref_traj_obj.ref_traj
        self.num_ref_traj_points = ref_traj_obj.num_waypoints
        self.min_init_ref_traj_idx = 0
        self.max_init_ref_traj_idx = self.num_ref_traj_points - self.max_steps_in_episode

        # calculate safety boundaries from reference trajectory
        pos_margin = jnp.array([0.5, 0.5, 0.5])
        vel_margin = jnp.array([0.5, 0.5, 0.5])
        self.world_box = WorldBox(
            ref_traj_obj.pos_bounds[0] - pos_margin, ref_traj_obj.pos_bounds[1] + pos_margin
        )
        self.v_min = ref_traj_obj.vel_bounds[0] - vel_margin
        self.v_max = ref_traj_obj.vel_bounds[1] + vel_margin

        # configure start behavior based on training or evaluation flags
        if from_start:
            self.max_init_ref_traj_idx = 1
            self.yaw_scale = 0.01
            self.pitch_roll_scale = 0.01
            self.position_std = 0.01
            self.velocity_std = 0.01
            self.omega_std = 0.01
        
        if skip_start:
            self.min_init_ref_traj_idx = 50 * 3     # skip first 3 seconds of speedup

        if train_from_start:
            self.max_init_ref_traj_idx = 1
            self.yaw_scale = 0.1
            self.pitch_roll_scale = 0.1
            self.position_std = 0.15
            self.velocity_std = 0.05
            self.omega_std = 0.05

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key, state: Optional[EnvState] = None
    ) -> tuple[EnvState, jax.Array]:
        """Resets the environment to a sampled state on the reference trajectory."""
        key_p, key_R, key_v, key_omega, key_dr = jax.random.split(key, 5)

        # sample start index
        start_traj_idx = jax.random.randint(
            key_p, shape=(), minval=self.min_init_ref_traj_idx, maxval=self.max_init_ref_traj_idx
        )

        # 1. position sampling
        p_target = jnp.array(self.ref_traj[start_traj_idx, TrajColumns.POS.slice])
        pos_noise = jax.random.normal(key_p, shape=(3,)) * self.position_std
        p = p_target + pos_noise
        p = jnp.clip(p, self.world_box.min, self.world_box.max)

        # 2. orientation sampling
        rot_target = math_utils.rot_from_quat(
            jnp.array(self.ref_traj[start_traj_idx, TrajColumns.QUAT.slice])
        )
        rot_noise = random_rotation(
            key_R, self.yaw_scale, self.pitch_roll_scale, self.pitch_roll_scale
        )
        R_target = rot_target.as_matrix()
        R_noise = rot_noise.as_matrix()
        R = R_target @ R_noise

        # 3. velocity sampling
        v_target = jnp.array(self.ref_traj[start_traj_idx, TrajColumns.VEL.slice])
        v_noise = jax.random.normal(key_v, shape=(3,)) * self.velocity_std
        v = v_target + v_noise
        v = jnp.clip(v, self.v_min, self.v_max)

        # 4. angular velocity sampling
        omega_target = jnp.array(self.ref_traj[start_traj_idx, TrajColumns.OMEGA.slice])
        omega_noise = jax.random.normal(key_omega, shape=(3,)) * self.omega_std
        omega = omega_target + omega_noise
        omega = jnp.clip(omega, self.omega_min, self.omega_max)

        # build initial states
        quadrotor_state = self.quadrotor.create_state(
            p=p, R=R, v=v, omega=omega, dr_key=key_dr
        )
        last_actions = jnp.tile(self.hovering_action, (self.num_last_actions, 1))
        last_quadrotor_states = stack_pytrees([quadrotor_state] * self.num_last_quad_states)

        state = EnvState(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quadrotor_states,
            init_ref_traj_idx=start_traj_idx,
        )

        obs = self._get_obs(state)
        return state, obs

    def _get_obs(self, state: EnvState) -> jax.Array:
        """Extracts observation vector from state."""
        return jnp.concatenate(
            [
                state.quadrotor_state.p,
                math_utils.vec(state.quadrotor_state.R),
                state.quadrotor_state.v,
                state.last_actions.flatten(),
            ]
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: EnvState, action: jax.Array, res_model_params: FrozenDict, key: chex.PRNGKey
    ) -> EnvTransition:
        """Advances environment by one time step with action delay."""
        action = jnp.clip(action, self.action_space.low, self.action_space.high)

        # update action history for delay
        last_actions = jnp.roll(state.last_actions, shift=-1, axis=0)
        last_actions = last_actions.at[-1].set(action)

        # first integration step for delay
        dt_1 = self.delay - (self.num_last_actions - 2) * self.dt
        action_1 = last_actions[0]
        f_1, omega_1 = action_1[0], action_1[1:]
        quadrotor_state = self.quadrotor.step(
            state.quadrotor_state, f_1, omega_1, res_model_params, dt_1
        )

        # second integration step if needed to complete dt
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

        obs = self._get_obs(next_state)
        reward = self._get_reward(state, next_state)
        terminated = self._is_colliding(next_state)
        truncated = jnp.greater_equal(next_state.step_idx, self.max_steps_in_episode)

        return EnvTransition(next_state, obs, reward, terminated, truncated, dict())

    def _get_reward(
        self, last_state: EnvState, next_state: EnvState
    ) -> jax.Array:
        """Calculates tracking reward based on proximity to reference."""
        pos = next_state.quadrotor_state.p
        vel = next_state.quadrotor_state.v
        action = next_state.last_actions[-1]

        # determine target point index
        target_idx = next_state.init_ref_traj_idx + next_state.step_idx
        target_idx = jax.lax.select(
            pred = target_idx >= self.num_ref_traj_points,
            on_true = self.num_ref_traj_points - 1,     # stay at end of trajectory
            on_false = target_idx
        )

        # get target values
        pos_target = jnp.array(self.ref_traj[target_idx, TrajColumns.POS.slice])
        vel_target = jnp.array(self.ref_traj[target_idx, TrajColumns.VEL.slice])

        # smoothness l1 costs
        pos_cost = 1.0 * smooth_l1((pos - pos_target))
        vel_cost = 1.0 * smooth_l1((vel - vel_target))
        tracking_cost = pos_cost + vel_cost

        action_cost = 0.1 * smooth_l1(action - self.hovering_action)
        cost = tracking_cost + action_cost

        # handle collision penalties
        time_left = self.max_steps_in_episode - next_state.step_idx
        collision_cost = jax.lax.select(
            self._is_colliding(next_state), time_left * cost, 0.0
        )
        cost += jax.lax.stop_gradient(collision_cost)

        return -self.dt * cost

    def _is_colliding(self, state: EnvState) -> jax.Array:
        """Checks if the drone is outside world boundaries."""
        return jnp.logical_not(self.world_box.contains(state.quadrotor_state.p))

    @property
    def action_space(self) -> spaces.Box:
        """Defines the action bounds for thrust and rates."""
        low = jnp.concatenate([jnp.array([self.thrust_min * 4]), self.omega_min])
        high = jnp.concatenate([jnp.array([self.thrust_max * 4]), self.omega_max])
        return spaces.Box(low, high, shape=(4,))

    @property
    def observation_space(self) -> spaces.Box:
        """Defines the observation vector bounds."""
        n = self.num_last_actions
        action_high_repeated = jnp.concatenate([self.action_space.high] * n)
        action_low_repeated = jnp.concatenate([self.action_space.low] * n)

        return spaces.Box(
            low=jnp.concatenate([self.world_box.min, -jnp.ones(9), self.v_min, action_low_repeated]),
            high=jnp.concatenate([self.world_box.max, jnp.ones(9), self.v_max, action_high_repeated]),
            shape=(15 + n * 4,),
        )

    @classmethod
    def generate_csv(cls, traj: EnvTransition, filename: str):
        """Exports a batch of trajectories to csv files."""
        from tqdm import tqdm
        num_dim = traj.reward.ndim
        if num_dim == 1:
            pytree_get_item(traj, None)
        num_trajectories = traj.reward.shape[0]
        for i in tqdm(range(num_trajectories)):
            traj_i = pytree_get_item(traj, i)
            cls._generate_csv(traj_i, f"{filename}_{i}.csv")

    @staticmethod
    def _generate_csv(traj: EnvTransition, filename: str):
        """Writes single trajectory data to a csv."""
        done = jnp.logical_or(traj.terminated, traj.truncated)
        traj_length = jnp.where(done)[0][0].item() + 1

        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["index", "t", "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            rows = []
            for i in range(traj_length):
                transition = pytree_get_item(traj, i)
                quat = rot_to_quat(Rotation.from_matrix(transition.state.quadrotor_state.R))
                rows.append({
                    "index": i,
                    "t": transition.state.time,
                    "px": transition.state.quadrotor_state.p[0],
                    "py": transition.state.quadrotor_state.p[1],
                    "pz": transition.state.quadrotor_state.p[2],
                    "qw": quat[0], "qx": quat[1], "qy": quat[2], "qz": quat[3],
                    "vx": transition.state.quadrotor_state.v[0],
                    "vy": transition.state.quadrotor_state.v[1],
                    "vz": transition.state.quadrotor_state.v[2],
                })
            writer.writerows(rows)

    def plot_trajectories(self, traj: EnvTransition, vertical_plane: bool = False, save_path=None):
        """Visualizes flight paths against the reference trajectory."""
        from matplotlib import pyplot as plt
        import seaborn as sns

        assert traj.reward.ndim == 2
        num_trajs = traj.reward.shape[0]
        state: EnvState = traj.state
        done = np.logical_or(traj.terminated, traj.truncated)

        sns.set_theme(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Palette matching previous plots
        ref_color = '#e67e22'   # Soft orange for reference
        traj_color = '#2c3e50'  # Dark slate for actual path
        arrow_color = '#3498db' # Light blue for orientation
        start_color = '#27ae60' # Emerald green
        end_color = '#e74c3c'   # Alizarin red

        # 1. Plot reference path with dashed style for distinction
        ref_traj_pos = self.ref_traj[:, TrajColumns.POS.slice]
        idx_v = 2 if vertical_plane else 1
        plane_label = "XZ" if vertical_plane else "XY"

        ax1.plot(ref_traj_pos[:, 0], ref_traj_pos[:, idx_v], 
                color=ref_color, linestyle='--', linewidth=2, alpha=0.8, 
                label=f"Reference Trajectory ({plane_label})")

        # 2. Set axes and equal aspect ratio
        bounds_margin = 0.5
        min_bounds = np.min(ref_traj_pos, axis=0)
        max_bounds = np.max(ref_traj_pos, axis=0)
        ax1.set_xlim(min_bounds[0] - bounds_margin, max_bounds[0] + bounds_margin)
        ax1.set_ylim(min_bounds[idx_v] - bounds_margin, max_bounds[idx_v] + bounds_margin)
        ax1.set_aspect("equal", adjustable="box")

        # 3. Plot drone trajectories
        for i in range(num_trajs):
            idx = np.where(done[i])[0][0].item() + 1
            x = state.quadrotor_state.p[i, :idx, 0]
            v_coord = state.quadrotor_state.p[i, :idx, idx_v]
            R = state.quadrotor_state.R[i, :idx]
            
            # Actual flight path
            ax1.plot(x, v_coord, color=traj_color, linewidth=1.5, alpha=0.7, 
                    label="Rollout Trajectory" if i == 0 else None)
            
            # Start/End points with white edges for visibility
            ax1.scatter(x[0], v_coord[0], color=start_color, s=50, edgecolors='white', zorder=5, 
                        label="Start" if i == 0 else None)
            ax1.scatter(x[-1], v_coord[-1], color=end_color, s=50, edgecolors='white', zorder=5, 
                        label="End" if i == 0 else None)

            # Orientation vectors (quivers)
            n_orient = 10
            v_dir = R[::n_orient, idx_v, 0]
            ax1.quiver(x[::n_orient], v_coord[::n_orient], R[::n_orient, 0, 0], v_dir, 
                    color=arrow_color, scale=12.0, width=0.005, alpha=0.8,
                    label="Heading" if i == 0 else None)

        # 4. Final Polish
        ax1.set_title(f"Quadrotor Tracking Performance ({plane_label} Plane)", fontweight='bold', pad=15, fontsize=14)
        ax1.set_xlabel(f"{plane_label[0]} Position (m)")
        ax1.set_ylabel(f"{plane_label[1]} Position (m)")

        ax1.legend(loc='best', fontsize='small', frameon=True)
        sns.despine(left=True, bottom=True) # Cleaner look for equal-aspect plots
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
