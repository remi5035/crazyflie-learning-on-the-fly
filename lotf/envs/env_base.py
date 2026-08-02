from functools import partial
from typing import Any, Dict, Generic, Optional, TypeVar
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from flax.core import FrozenDict

from lotf.utils.pytrees import CustomPyTree, tree_select
from lotf.utils.spaces import Space

TEnvState = TypeVar("TEnvState", bound="EnvState")


class EnvTransition(NamedTuple):
    """Container for a single environment transition."""
    state: TEnvState
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    info: Dict[str, Any]


@jdc.pytree_dataclass
class EnvState(CustomPyTree):
    """Base class for environment states using pytree registration."""
    pass


class Env(Generic[TEnvState]):
    """Abstract base class for jax-compatible environments."""

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: TEnvState, action: jax.Array, res_model_params: FrozenDict, key: chex.PRNGKey
    ) -> EnvTransition:
        """
        Executes an environment step and handles automatic resets.

        Args:
            state: current environment state.
            action: action to apply.
            res_model_params: parameters for the residual dynamics model.
            key: rng key for stochasticity.

        Returns:
            an EnvTransition containing the next state or reset state.
        """
        key_step, key_reset = jax.random.split(key)
        
        # perform the underlying environment physics/logic step
        step_state, step_obs, reward, terminated, truncated, info = self._step(
            state, action, res_model_params, key_step
        )
        
        # prepare potential reset state
        reset_state, reset_obs = self.reset(key_reset, state)
        
        # auto-reset state if termination or truncation occurs
        done = jnp.logical_or(terminated, truncated)
        state = tree_select(done, reset_state, step_state)
        obs = tree_select(done, reset_obs, step_obs)

        return EnvTransition(state, obs, reward, terminated, truncated, info)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, state: Optional[TEnvState] = None
    ) -> tuple[TEnvState, jax.Array]:
        """Resets the environment to an initial state."""
        raise NotImplementedError

    def _step(
        self, state: TEnvState, action: jax.Array, res_model_params: FrozenDict, key: chex.PRNGKey
    ) -> EnvTransition:
        """Internal environment-specific step logic."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Returns the name of the environment class."""
        return type(self).__name__

    @property
    def action_space(self) -> Space:
        """Returns the action space definition."""
        raise NotImplementedError

    @property
    def observation_space(self) -> Space:
        """Returns the observation space definition."""
        raise NotImplementedError

    @property
    def unwrapped(self):
        """Returns the base environment instance."""
        return self


def rollout(
    env,
    key,
    policy,
    res_model_params: FrozenDict,
    state: Optional[EnvState] = None,
    *,
    real_step: bool = False,
    num_steps=None,
) -> EnvTransition:
    """
    Executes a fixed-length rollout of a given policy.

    Args:
        env: environment instance.
        key: rng key for the rollout.
        policy: callable policy mapping (obs, key) -> action.
        res_model_params: parameters for the residual dynamics model.
        state: optional initial state.
        real_step: if true, uses auto-resetting step(); otherwise uses raw _step().
        num_steps: number of steps to simulate.

    Returns:
        concatenated transitions for the entire trajectory.
    """
    if num_steps is None:
        num_steps = env.max_steps_in_episode
        
    state, obs = env.reset(key, state)
    
    # initialize transition container for concatenation
    trans_init = EnvTransition(
        state, obs, jnp.array(0), jnp.array(0), jnp.array(0), dict()
    )

    def step_fn(step_state, key_step):
        env_state, obs = step_state
        key_policy, key_step = jax.random.split(key_step)
        
        # sample action from policy
        action = policy(obs, key_policy)
        
        # select between auto-resetting step or raw transition
        if real_step:
            trans = env.step(env_state, action, res_model_params, key_step)
        else:
            trans = env._step(env_state, action, res_model_params, key_step)
            
        return (trans.state, trans.obs), trans

    # scan over time steps
    keys_steps = jax.random.split(key, num_steps)
    _, transitions = jax.lax.scan(step_fn, (state, obs), keys_steps)
    
    # prepend the initial state to the trajectory
    transitions = jax.tree.map(
        lambda l0, l1: jnp.concatenate([l0[None], l1]), trans_init, transitions
    )

    return transitions
