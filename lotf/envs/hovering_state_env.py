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
    State representation for the hovering environment.
    
    Attributes:
        time: elapsed simulation time.
        step_idx: current step count in the episode.
        quadrotor_state: physical state of the drone.
        last_actions: history of actions used to simulate control latency.
        last_quadrotor_states: history of previous physical states.
    """
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    last_actions: jax.Array
    last_quadrotor_states: QuadrotorState
    act_thrust: jax.Array   # poussée RÉELLEMENT appliquée (état du lag actionneur 1er ordre)


class HoveringStateEnv(env_base.Env[EnvState]):
    """
    Quadrotor hovering environment with control delay and state-based observations.
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
        omega_cmd_penalty_weight=0.0,
        action_rate_penalty_weight=0.0,
        actuator_tau=0.0,
        num_last_quad_states=10,
        margin=0.0,
        hover_height=1.0,
        hover_target=None,
    ):
        """Initializes environment parameters, quadrotor physics, and goal targets."""
        
        # define the target hovering position
        self.goal: jnp.ndarray = jnp.array([0.0, 0.0, hover_height])
        if hover_target is not None:
            self.goal = jnp.array(hover_target, dtype=jnp.float32)

        # setup world boundaries for collision and normalization
        side_length = 3.0
        self.world_box = WorldBox(
            self.goal - side_length / 2,
            self.goal + side_length / 2
        )

        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)
        
        # distribution parameters for randomized initial states
        self.yaw_scale = yaw_scale
        self.pitch_roll_scale = pitch_roll_scale
        self.velocity_std = velocity_std
        self.omega_std = omega_std

        # initialize the quadrotor model
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

        # calculate the size of the action buffer needed for delay simulation
        assert delay >= 0.0, "Delay must be non-negative"
        self.delay = np.array(delay)
        self.num_last_actions = int(np.ceil(delay / dt)) + 1

        self.reward_sharpness = reward_sharpness
        self.action_penalty_weight = action_penalty_weight
        self.omega_cmd_penalty_weight = omega_cmd_penalty_weight
        self.action_rate_penalty_weight = action_rate_penalty_weight
        # Lag d'actionneur 1er ordre sur la poussée (s) : T_appliqué suit T_commandé avec
        # cette cste de temps (= spin-up moteur). 0 = poussée instantanée (low-fi historique).
        # Distinct du `delay` (latence pure radio/calcul). ~0.04 s rapproche la low-fi de
        # Genesis (lag moteur) -> la policy apprend à ne pas être agressive en z.
        self.actuator_tau = actuator_tau

        # reference action for steady hover
        thrust_hover = 9.81 * self.quadrotor._mass
        self.hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])

        self.num_last_quad_states = num_last_quad_states
        self.margin = margin

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key, state: Optional[EnvState] = None
    ) -> tuple[EnvState, jax.Array]:
        """Resets environment state with randomized quadrotor pose and velocity."""

        key_p, key_R, key_v, key_omega, key_dr = jax.random.split(key, 5)
        
        # randomize starting position within the world box
        p = jax.random.uniform(
            key_p,
            shape=(3,),
            minval=self.world_box.min + self.margin,
            maxval=self.world_box.max - self.margin,
        )

        # randomize initial orientation
        rot = random_rotation(
            key_R, self.yaw_scale, self.pitch_roll_scale, self.pitch_roll_scale
        )
        R = rot.as_matrix()
        
        # randomize linear and angular velocities
        v = self.velocity_std * jax.random.normal(key_v, shape=(3,))
        omega = self.omega_std * jax.random.normal(key_omega, shape=(3,))

        quadrotor_state = self.quadrotor.create_state(
            p=p, R=R, v=v, omega=omega, dr_key=key_dr
        )

        # initialize history buffers with default values
        last_actions = jnp.tile(
            self.hovering_action, (self.num_last_actions, 1)
        )
        last_quadrotor_states = stack_pytrees([quadrotor_state] * self.num_last_quad_states)

        state = EnvState(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quadrotor_states,
            act_thrust=self.hovering_action[0],   # démarre au hover (lag actionneur)
        )

        return state, self._get_obs(state)

    def _lag_thrust(self, t_prev: jax.Array, t_cmd: jax.Array, dt) -> jax.Array:
        """Filtre 1er ordre du lag actionneur : ramène la poussée appliquée vers la commande.
        actuator_tau<=0 -> pas de lag (retourne la commande). dt/alpha sont statiques."""
        if self.actuator_tau <= 0.0:
            return t_cmd
        alpha = 1.0 - float(np.exp(-float(dt) / self.actuator_tau))
        return t_prev + alpha * (t_cmd - t_prev)

    def _get_obs(self, state: EnvState) -> jax.Array:
        """Flattens state components into a single observation vector."""
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
        """Updates environment physics, handles delay, and computes rewards."""
        
        # ensure actions stay within physical limits
        action = jnp.clip(
            action, self.action_space.low, self.action_space.high
        )

        # update action buffer for delay simulation
        last_actions = jnp.roll(state.last_actions, shift=-1, axis=0)
        last_actions = last_actions.at[-1].set(action)

        # handle the fractional time step caused by delay
        dt_1 = self.delay - (self.num_last_actions - 2) * self.dt
        action_1 = last_actions[0]
        f_1, omega_1 = action_1[0], action_1[1:]
        # actuator lag : la poussée appliquée suit la commande avec une cste de temps
        act_thrust = self._lag_thrust(state.act_thrust, f_1, dt_1)
        quadrotor_state = self.quadrotor.step(
            state.quadrotor_state, act_thrust, omega_1, res_model_params, dt_1
        )

        # complete the remaining time step with the subsequent action
        if dt_1 < self.dt:
            dt_2 = self.dt - dt_1
            action_2 = last_actions[1]
            f_2, omega_2 = action_2[0], action_2[1:]
            act_thrust = self._lag_thrust(act_thrust, f_2, dt_2)
            quadrotor_state = self.quadrotor.step(
                quadrotor_state, act_thrust, omega_2, res_model_params, dt_2
            )

        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            act_thrust=act_thrust,
        )

        obs = self._get_obs(next_state)
        reward = self._get_reward(state, next_state)
        terminated = self._is_colliding(next_state)
        truncated = jnp.greater_equal(next_state.step_idx, self.max_steps_in_episode)

        return EnvTransition(next_state, obs, reward, terminated, truncated, dict())

    def _get_reward(
        self, last_state: EnvState, next_state: EnvState
    ) -> jax.Array:
        """Computes a weighted reward based on pose accuracy and effort."""

        action = next_state.last_actions[-1]
        p = next_state.quadrotor_state.p
        acc = next_state.quadrotor_state.acc

        # penalize deviation from goal using smooth l1 for better gradient behavior
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

        # dedicated L1 penalty on the COMMANDED body rate (action[1:]). The smooth_l1
        # term above is quadratic near 0, so its gradient vanishes there: a small steady
        # omega bias (~0.01 rad/s) is essentially free and gets baked into the policy at
        # pretrain (cause of the 33g horizontal drift). |omega_cmd| has a CONSTANT gradient
        # at 0 -> actively pins the hover body-rate command to zero. Weight 0 = no-op.
        omega_cmd_cost = self.omega_cmd_penalty_weight * jnp.sum(jnp.abs(action[1:]))

        # penalize the action RATE (change between consecutive commands). action_penalty above
        # penalizes amplitude, not variation: an aggressive (long-horizon) policy can still
        # command sharp thrust changes that excite actuation lag (motor lag / PWM) on Genesis
        # and the real drone -> vertical oscillation at the start of a move. smooth_l1 of the
        # command delta damps this without a deadzone (quadratic near 0). Weight 0 = no-op.
        prev_action = next_state.last_actions[-2]
        action_rate_cost = self.action_rate_penalty_weight * smooth_l1(action - prev_action)
        cost = goal_cost + action_cost + omega_cmd_cost + action_rate_cost

        # apply massive penalty if collision occurs
        time_left = self.max_steps_in_episode - next_state.step_idx
        collision_cost = jax.lax.select(
            self._is_colliding(next_state), time_left * cost, 0.0
        )
        cost += jax.lax.stop_gradient(collision_cost)

        return -self.dt * cost

    def _is_colliding(self, state: EnvState) -> jax.Array:
        """Checks if the drone is outside the defined world box."""
        return jnp.logical_not(self.world_box.contains(state.quadrotor_state.p))

    @property
    def action_space(self) -> spaces.Box:
        """Returns the bounds for thrust and angular velocity."""
        low = jnp.concatenate([jnp.array([self.thrust_min * 4]), self.omega_min])
        high = jnp.concatenate([jnp.array([self.thrust_max * 4]), self.omega_max])
        return spaces.Box(low, high, shape=(4,))

    @property
    def observation_space(self) -> spaces.Box:
        """Returns the bounds for positions, rotations, and buffered actions."""
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
        """Exports trajectory data for external analysis or playback."""
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
            fieldnames = ["index", "t", "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(traj_length):
                transition = pytree_get_item(traj, i)
                t = transition.state.time
                p = transition.state.quadrotor_state.p
                R = transition.state.quadrotor_state.R
                quat = rot_to_quat(Rotation.from_matrix(R))
                writer.writerow({
                    "index": i, "t": t, "px": p[0], "py": p[1], "pz": p[2],
                    "qw": quat[0], "qx": quat[1], "qy": quat[2], "qz": quat[3],
                    "vx": transition.state.quadrotor_state.v[0],
                    "vy": transition.state.quadrotor_state.v[1],
                    "vz": transition.state.quadrotor_state.v[2],
                })

    def plot_trajectories(self, traj: EnvTransition):
        """Visualizes flight paths in 2d (x-y) and altitude (z) over time."""
        from matplotlib import pyplot as plt
        import seaborn as sns

        num_trajs = traj.reward.shape[0]
        state: EnvState = traj.state
        done = np.logical_or(traj.terminated, traj.truncated)

        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1]})

        # Colors for specific elements
        traj_color = '#2c3e50'  # Dark slate
        arrow_color = '#3498db' # Light blue
        start_color = '#27ae60' # Emerald green
        end_color = '#e74c3c'   # Alizarin red

        for i in range(num_trajs):
            idx = np.where(done[i])[0][0].item() + 1
            x, y, z = [state.quadrotor_state.p[i, :idx, j] for j in range(3)]
            R = state.quadrotor_state.R[i, :idx]
            t = state.time[i, :idx]
            
            # Left Plot: XY Trajectory
            ax1.plot(x, y, color=traj_color, alpha=0.6, linewidth=1.5, label="Trajectory" if i==0 else "")
            
            # Draw orientation arrows (Heading/X-axis of body frame)
            n_orient = 10
            ax1.quiver(x[::n_orient], y[::n_orient], R[::n_orient, 0, 0], R[::n_orient, 1, 0], 
                    color=arrow_color, scale=12.0, width=0.005, alpha=0.8, label="Heading" if i==0 else "")
            
            # Start/End points
            ax1.scatter(x[0], y[0], color=start_color, s=40, edgecolors='white', zorder=5, label="Start" if i==0 else "")
            ax1.scatter(x[-1], y[-1], color=end_color, s=40, edgecolors='white', zorder=5, label="End" if i==0 else "")
            
            # Right Plot: Z over Time
            ax2.plot(t, z, color=traj_color, linewidth=1.5, alpha=0.7)

        # Labels and Titling
        ax1.set_title("XY Flight Path", fontweight='bold', fontsize=14)
        ax1.set_xlabel("X Position (m)")
        ax1.set_ylabel("Y Position (m)")

        ax2.set_title("Altitude Profile", fontweight='bold', fontsize=14)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Z Position (m)")

        ax1.legend(loc='upper right', fontsize='small', frameon=True)
        sns.despine()
        plt.tight_layout()
        plt.show()
