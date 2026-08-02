from functools import partial

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from flax.core import FrozenDict

from lotf.envs.env_base import EnvTransition, Env, EnvState
from lotf.utils import spaces
from lotf.utils.math import normalize
from lotf.utils.spaces import Space


class EnvWrapper(Env):
    """Base class for environment wrappers to modify behavior or observation/action spaces."""

    def __init__(self, env: Env):
        """Initialize the wrapper with an environment instance."""
        self._env = env

    def __getattr__(self, item):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self._env, item)

    def reset(self, key, state=None):
        """Resets the wrapped environment."""
        return self._env.reset(key, state)

    def _step(self, state, action, res_model_params, key) -> EnvTransition:
        """Internal step logic of the wrapped environment."""
        return self._env._step(state, action, res_model_params, key)

    @property
    def action_space(self) -> Space:
        """Returns the action space of the wrapped environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> Space:
        """Returns the observation space of the wrapped environment."""
        return self._env.observation_space

    def step(self, state, action, res_model_params, key) -> EnvTransition:
        """Executes a step in the wrapped environment."""
        return self._env.step(state, action, res_model_params, key)

    @property
    def unwrapped(self):
        """Returns the base environment without any wrappers."""
        return self._env.unwrapped


@jdc.pytree_dataclass
class LogEnvState(EnvState):
    """State container for the logging wrapper, including episode statistics."""
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(EnvWrapper):
    """Tracks and logs episode rewards and lengths."""

    def __init__(self, env: Env):
        """Initialize the logging wrapper."""
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, state=None):
        """Resets the env and initializes logging counters."""
        if state is not None:
            env_state = state.env_state
        else:
            env_state = None
            
        state, obs = self._env.reset(key, env_state)
        
        log_state = LogEnvState(
            env_state=state,
            episode_returns=0.0,
            episode_lengths=0,
            returned_episode_returns=0.0,
            returned_episode_lengths=0,
            timestep=0,
        )
        return log_state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: LogEnvState, action, res_model_param: FrozenDict, key) -> EnvTransition:
        """Updates logging metrics on each step."""
        transition = self._env.step(state.env_state, action, res_model_param, key)
        env_state, obs, reward, terminated, truncated, info = transition
        
        done = jnp.logical_or(terminated, truncated)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        
        # update state and reset episode counters if done
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        
        # populate info dict with log data
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        
        return transition._replace(state=state, info=info)

    def _get_obs(self, state, asymmetric=False):
        """Retrieves observation from the underlying state."""
        return self._env._get_obs(state.env_state, asymmetric)


class VecEnv(EnvWrapper):
    """Vectorizes environment methods for parallel execution across a batch."""

    def __init__(self, env):
        """Initialize vectorized versions of reset, step, and obs retrieval."""
        super().__init__(env)
        # vmap over the batch dimension
        self.reset = jax.vmap(self._env.reset, in_axes=(0, 0))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, None, 0))
        self._get_obs = jax.vmap(self._env._get_obs, in_axes=(0, None))


class MinMaxObservationWrapper(EnvWrapper):
    """Normalizes observations to a range of [-1, 1] based on space bounds."""

    def __init__(self, env):
        """Caches min/max bounds and validates for infinite values."""
        super().__init__(env)
        self._obs_min = jnp.array(self._env.observation_space.low)
        self._obs_max = jnp.array(self._env.observation_space.high)
        
        # safety checks for normalization
        assert jnp.isinf(self._obs_max).sum() == 0, "obs space has infinities"
        assert jnp.isinf(self._obs_min).sum() == 0, "obs space has infinities"

    @property
    def observation_space(self) -> spaces.Box:
        """Returns the new normalized observation space bounds."""
        return spaces.Box(low=-1.0, high=1.0, shape=self._obs_min.shape)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, state=None):
        """Resets the environment and returns a normalized observation."""
        state, obs = self._env.reset(key, state)
        obs = normalize(obs, self._obs_min, self._obs_max)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action, res_model_params, key) -> EnvTransition:
        """Steps the environment and normalizes the resulting observation."""
        transition = self._env.step(state, action, res_model_params, key)
        obs = normalize(transition.obs, self._obs_min, self._obs_max)
        return transition._replace(obs=obs)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action, res_model_params, key) -> EnvTransition:
        """Internal step with normalized observation output."""
        transition = self._env._step(state, action, res_model_params, key)
        obs = normalize(transition.obs, self._obs_min, self._obs_max)
        return transition._replace(obs=obs)
