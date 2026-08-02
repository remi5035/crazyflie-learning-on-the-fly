import numpy as np
from functools import partial

import chex
import jax
from jax import numpy as jnp
from flax.core import FrozenDict

from lotf.envs import HoveringStateEnv
from lotf.envs.env_base import EnvTransition
from lotf.envs.hovering_state_env import EnvState
from lotf.sensors import DoubleSphereCamera, CameraNames
from lotf.utils import spaces
from lotf.utils.pytrees import pytree_roll, pytree_at_set
from lotf.utils.math import normalize


class HoveringFeaturesEnv(HoveringStateEnv):
    """Hovering environment using projected visual features as observations."""

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
        skip_frames=1,
        margin=0.0,
        hover_height=1.0,
        hover_target=None,
    ):
        """Initializes the feature-based environment with a downward-looking camera."""
        super().__init__(
            max_steps_in_episode=max_steps_in_episode,
            dt=dt,
            delay=delay,
            yaw_scale=yaw_scale,
            pitch_roll_scale=pitch_roll_scale,
            velocity_std=velocity_std,
            omega_std=omega_std,
            quad_obj=quad_obj,
            reward_sharpness=reward_sharpness,
            action_penalty_weight=action_penalty_weight,
            num_last_quad_states=num_last_quad_states,
            margin=margin,
            hover_height=hover_height,
            hover_target=hover_target,
        )
        # setup camera model
        self.cam = DoubleSphereCamera.from_name(CameraNames.EXAMPLE_CAM)
        self.cam.pitch = -90.0  # orient camera to look straight down
        self.skip_frames = skip_frames

    @property
    def observation_space(self) -> spaces.Box:
        """Defines space for concatenated visual points and normalized action history."""
        num_frames = int(np.ceil(self.num_last_quad_states / self.skip_frames))
        # 7 features per frame, 2 coordinates each, plus action buffer
        return spaces.Box(
            low=1,
            high=-1,
            shape=(2 * 7 * num_frames + 4 * self.num_last_actions,),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: EnvState, action: jax.Array, res_model_params: FrozenDict, key: chex.PRNGKey
    ) -> EnvTransition:
        """Executes physics step and updates the history of quadrotor states."""

        action = jnp.clip(
            action, self.action_space.low, self.action_space.high
        )

        # update action buffer for control delay
        last_actions = jnp.roll(state.last_actions, shift=-1, axis=0)
        last_actions = last_actions.at[-1].set(action)

        # apply physics integration (handling fractional steps for delay)
        dt_1 = self.delay - (self.num_last_actions - 2) * self.dt
        action_1 = last_actions[0]
        f_1, omega_1 = action_1[0], action_1[1:]
        quadrotor_state = self.quadrotor.step(
            state.quadrotor_state, f_1, omega_1, res_model_params, dt_1
        )

        if dt_1 < self.dt:
            dt_2 = self.dt - dt_1
            action_2 = last_actions[1]
            f_2, omega_2 = action_2[0], action_2[1:]
            quadrotor_state = self.quadrotor.step(
                quadrotor_state, f_2, omega_2, res_model_params, dt_2
            )

        # update rolling buffer of physical states for feature projection history
        last_quad_states = state.last_quadrotor_states
        last_quad_states = pytree_roll(last_quad_states, shift=-1, axis=0)
        last_quad_states = pytree_at_set(last_quad_states, -1, quadrotor_state)

        next_state = state.replace(
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            last_quadrotor_states=last_quad_states,
        )

        obs = self._get_obs(next_state)
        reward = self._get_reward(state, next_state)
        terminated = self._is_colliding(next_state)
        truncated = jnp.greater_equal(next_state.step_idx, self.max_steps_in_episode)

        return EnvTransition(next_state, obs, reward, terminated, truncated, dict())

    def _get_obs(self, state: EnvState, asymmetric=False) -> jax.Array:
        """Projects 3d world features into 2d normalized camera coordinates."""

        def project_fn(p, R):
            # define 7 reference points on a plane below the target
            feature_pos = jnp.array(
                [
                    [0.5, 0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [-0.5, -0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                    [-0.5, 0, 0.0],
                    [0, 0.5, 0.0],
                    [0.5, 0, 0.0],
                ]
            )
            # offset features to be 1m below the hovering target
            planar_offset = jnp.array([self.goal[0], self.goal[1], self.goal[2] - 1.0])
            feature_pos = feature_pos + planar_offset

            # project world points into camera pixel coordinates
            points_C = self.cam.project_points_with_pose(feature_pos, p, R)
            points_C = points_C[:, :2]

            # normalize pixels to range [-1, 1]
            points_normalized = points_C / jnp.array(
                [self.cam.width, self.cam.height], dtype=float
            ) * 2.0 - 1.0
            return points_normalized

        # vmap projection across the history of states
        project_vmap = jax.vmap(project_fn, in_axes=(0, 0))
        points_normalized = project_vmap(
            state.last_quadrotor_states.p[::-self.skip_frames],
            state.last_quadrotor_states.R[::-self.skip_frames],
        )

        # prepare normalized action history for observation
        last_actions = state.last_actions.flatten()
        actions_low = jnp.concatenate([self.action_space.low] * self.num_last_actions)
        actions_high = jnp.concatenate([self.action_space.high] * self.num_last_actions)
        last_action_normalized = normalize(last_actions, actions_low, actions_high)

        return jnp.concatenate([points_normalized.flatten(), last_action_normalized])
