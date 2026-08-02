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
    """Prints training progress from the host process."""
    episode, loss = episode_loss
    print(f"Episode: {episode}, Loss: {loss:.2f}")


NUM_EPOCHS_PER_CALLBACK = 10


def progress_callback(episode, loss):
    """Triggers a host-side debug callback for loss logging."""
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(
            progress_callback_host, eps_lss
        ),
        false_fun=lambda eps_lss: None,
        operand=(episode, loss),
    )


def grad_callback_host(episode_grad):
    """Prints gradient statistics from the host process."""
    episode, grad = episode_grad
    print(f"Episode: {episode}, Grad max: {grad:.4f}")


def grad_callback(episode, grad_norm):
    """Triggers a host-side debug callback for gradient logging."""
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(
            grad_callback_host, eps_lss
        ),
        false_fun=lambda eps_lss: None,
        operand=(episode, grad_norm),
    )


class RunnerState(NamedTuple):
    """Represents the complete state of the training loop."""
    train_state: TrainState
    env_state: EnvState
    last_obs: jax.Array
    key: chex.PRNGKey
    epoch_idx: int


def train(
    env: Env,
    env_state: EnvState,
    obs: jax.Array,
    train_state: TrainState,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    res_model_params: FrozenDict,
    key: chex.PRNGKey,
):
    """
    Executes the training loop for a given environment using JAX transformations.

    Args:
        env: the environment instance.
        env_state: the initial state of the environment.
        obs: the initial observation.
        train_state: the flax train state containing params and optimizer.
        num_epochs: total number of training iterations.
        num_steps_per_epoch: rollout length per epoch.
        num_envs: number of parallel environments.
        res_model_params: fixed parameters for the residual dynamics model.
        key: rng key for stochastic operations.

    Returns:
        a dictionary containing the final runner state and training metrics.
    """

    # initialize the primary state container
    runner_state = RunnerState(train_state, env_state, obs, key, epoch_idx=0)

    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def _train(env, num_epochs, num_steps_per_epoch, num_envs, res_model_params: FrozenDict, runner_state: RunnerState):

        def epoch_fn(epoch_state: RunnerState, _unused):
            """Performs a single rollout, gradient update, and logging step."""

            @partial(jax.value_and_grad, has_aux=True)
            def loss_fn(params, runner_state: RunnerState):

                def rollout(runner_state: RunnerState):
                    """Simulates the environment for a fixed number of steps."""
                    def step_fn(old_runner_state: RunnerState, _unsused):

                        # extract current states and observations
                        train_state, env_state, last_obs, key, epoch_idx = (
                            old_runner_state
                        )

                        # generate action using the current policy params
                        action = train_state.apply_fn(params, last_obs)

                        # split keys for stochastic environment transitions
                        key, key_ = jax.random.split(key)
                        key_step = jax.random.split(key_, num_envs)
                        
                        # execute step in the environment
                        (
                            env_state,
                            obs,
                            reward,
                            _terminated,
                            _truncated,
                            info,
                        ) = env.step(env_state, action, res_model_params, key_step)
                        
                        runner_state = RunnerState(
                            train_state, env_state, obs, key, epoch_idx
                        )

                        return (
                            runner_state,
                            TrajectoryState(reward=reward),
                        )

                    # scan over steps to generate a trajectory
                    runner_state, trajectory = jax.lax.scan(
                        step_fn, runner_state, None, num_steps_per_epoch
                    )
                    return runner_state, trajectory

                # calculate loss based on accumulated rewards
                runner_state, trajectory = rollout(runner_state)
                loss = -trajectory.reward.sum() / num_envs
                return loss, runner_state

            # differentiate loss with respect to training parameters
            train_state = epoch_state.train_state
            (loss, epoch_state), grad = loss_fn(
                train_state.params, epoch_state
            )
            
            # apply gradients to update the model weights
            train_state = train_state.apply_gradients(grads=grad)

            # flatten gradients to calculate max absolute value for monitoring
            leaves = jax.tree_util.tree_leaves(grad)
            flattened_leaves = [jnp.ravel(leaf) for leaf in leaves]
            grad_vec = jnp.concatenate(flattened_leaves)
            grad_max = jnp.max(jnp.abs(grad_vec))

            # trigger callbacks for logging at specific intervals
            progress_callback(epoch_state.epoch_idx, loss)
            grad_callback(epoch_state.epoch_idx, grad_max)
            
            # update the state for the next epoch iteration
            epoch_state = epoch_state._replace(
                train_state=train_state, epoch_idx=epoch_state.epoch_idx + 1
            )

            return epoch_state, loss

        # scan over epochs to execute the full training loop
        runner_state_final, losses = jax.lax.scan(
            epoch_fn, runner_state, None, num_epochs
        )

        return {"runner_state": runner_state_final, "metrics": losses}
    
    return _train(env, num_epochs, num_steps_per_epoch, num_envs, res_model_params, runner_state)
