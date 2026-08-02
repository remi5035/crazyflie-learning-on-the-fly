"""BPTT training loop for recurrent (GRU/LSTM) policies.

Identical in structure to bptt.py but the RunnerState carries a hidden_state
array that is threaded through every step and reset to zeros at episode
termination boundaries.

The policy must have the signature:
    action, new_hidden = train_state.apply_fn(params, obs, hidden)

where hidden has shape (num_envs, hidden_size).
"""

from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState
from flax.core import FrozenDict

from lotf.envs.env_base import Env, EnvState


class TrajectoryState(PyTreeNode):
    """Holds the transition data collected during a rollout."""
    reward: jnp.array


def progress_callback_host(episode_loss):
    episode, loss = episode_loss
    print(f"Episode: {episode}, Loss: {loss:.2f}")


NUM_EPOCHS_PER_CALLBACK = 10


def progress_callback(episode, loss):
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(progress_callback_host, eps_lss),
        false_fun=lambda eps_lss: None,
        operand=(episode, loss),
    )


def grad_callback_host(episode_grad):
    episode, grad = episode_grad
    print(f"Episode: {episode}, Grad max: {grad:.4f}")


def grad_callback(episode, grad_norm):
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(grad_callback_host, eps_lss),
        false_fun=lambda eps_lss: None,
        operand=(episode, grad_norm),
    )


class RunnerState(NamedTuple):
    """Training-loop state, extended with GRU hidden state."""
    train_state: TrainState
    env_state: EnvState
    last_obs: jax.Array
    hidden_state: jax.Array   # (num_envs, hidden_size)
    key: chex.PRNGKey
    epoch_idx: int


def train(
    env: Env,
    env_state: EnvState,
    obs: jax.Array,
    train_state: TrainState,
    initial_hidden: jax.Array,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    res_model_params: FrozenDict,
    key: chex.PRNGKey,
):
    """BPTT training loop for recurrent policies.

    Args:
        env:                  environment instance (VecEnv-wrapped).
        env_state:            initial env state after reset.
        obs:                  initial observation, shape (num_envs, obs_dim).
        train_state:          Flax TrainState whose apply_fn is the GRU policy.
        initial_hidden:       zero carry, shape (num_envs, hidden_size).
        num_epochs:           number of gradient updates.
        num_steps_per_epoch:  rollout length (= episode length for BPTT).
        num_envs:             number of parallel environments.
        res_model_params:     fixed residual-dynamics parameters.
        key:                  PRNG key.

    Returns:
        dict with keys "runner_state" and "metrics" (per-epoch losses).
    """

    runner_state = RunnerState(
        train_state, env_state, obs, initial_hidden, key, epoch_idx=0
    )

    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def _train(
        env,
        num_epochs,
        num_steps_per_epoch,
        num_envs,
        res_model_params: FrozenDict,
        runner_state: RunnerState,
    ):
        def epoch_fn(epoch_state: RunnerState, _unused):

            @partial(jax.value_and_grad, has_aux=True)
            def loss_fn(params, runner_state: RunnerState):

                def rollout(runner_state: RunnerState):
                    def step_fn(old_runner_state: RunnerState, _unused):
                        train_state, env_state, last_obs, hidden, key, epoch_idx = (
                            old_runner_state
                        )

                        # recurrent forward pass: policy takes (obs, hidden)
                        action, new_hidden = train_state.apply_fn(
                            params, last_obs, hidden
                        )

                        # step all envs
                        key, key_ = jax.random.split(key)
                        key_step = jax.random.split(key_, num_envs)
                        (
                            env_state,
                            obs,
                            reward,
                            terminated,
                            _truncated,
                            _info,
                        ) = env.step(env_state, action, res_model_params, key_step)

                        # reset hidden state for envs that just terminated
                        # terminated shape: (num_envs,)
                        reset_mask = terminated[:, None]           # (num_envs, 1)
                        new_hidden = jnp.where(
                            reset_mask,
                            jnp.zeros_like(new_hidden),
                            new_hidden,
                        )

                        runner_state = RunnerState(
                            train_state, env_state, obs, new_hidden, key, epoch_idx
                        )
                        return runner_state, TrajectoryState(reward=reward)

                    runner_state, trajectory = jax.lax.scan(
                        step_fn, runner_state, None, num_steps_per_epoch
                    )
                    return runner_state, trajectory

                runner_state, trajectory = rollout(runner_state)
                loss = -trajectory.reward.sum() / num_envs
                return loss, runner_state

            train_state = epoch_state.train_state
            (loss, epoch_state), grad = loss_fn(train_state.params, epoch_state)

            train_state = train_state.apply_gradients(grads=grad)

            leaves = jax.tree_util.tree_leaves(grad)
            grad_vec = jnp.concatenate([jnp.ravel(l) for l in leaves])
            grad_max = jnp.max(jnp.abs(grad_vec))

            progress_callback(epoch_state.epoch_idx, loss)
            grad_callback(epoch_state.epoch_idx, grad_max)

            epoch_state = epoch_state._replace(
                train_state=train_state,
                epoch_idx=epoch_state.epoch_idx + 1,
            )
            return epoch_state, loss

        runner_state_final, losses = jax.lax.scan(
            epoch_fn, runner_state, None, num_epochs
        )
        return {"runner_state": runner_state_final, "metrics": losses}

    return _train(
        env, num_epochs, num_steps_per_epoch, num_envs, res_model_params, runner_state
    )
