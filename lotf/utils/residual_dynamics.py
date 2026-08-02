from typing import Callable, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from lotf.modules import ResidualDynamicsMLP


def get_residual_dyn_model_apply_fn() -> Callable:
    """
    Returns a vectorized apply function for the residual dynamics mlp.
    """
    def apply_fn(params, x):
        # initialize model architecture
        model = ResidualDynamicsMLP([19, 128, 128, 3], initial_scale=1.0)
        return model.apply(params, x)

    # vectorize over the parameter axis (first axis)
    parallel_apply_fn = jax.vmap(apply_fn, in_axes=(0, None))
    return parallel_apply_fn


def exact_mass_residual(x: jax.Array, m_nominal: float, m_real: float) -> jax.Array:
    """
    Exact residual acceleration that corrects a nominal-mass dynamics model so
    that it reproduces a quadrotor of true mass ``m_real``.

    The forward simulator integrates ``dv/dt = g + R @ [0, 0, f_d / m_nominal] + a_res``.
    For this to match the true system ``dv/dt = g + R @ [0, 0, f_d / m_real]``, the
    residual must be the difference of the two thrust accelerations:

        a_res = R @ [0, 0, f_d * (1/m_real - 1/m_nominal)]

    With ``m_real > m_nominal`` the coefficient is negative, so the residual removes
    upward thrust acceleration (the source of the steady-state z offset when a policy
    tuned for the lighter model flies the heavier one).

    Args:
        x: 19-d feature vector built in ``Quadrotor.step`` (p, R flattened, v, f_d, omega_d).
        m_nominal: mass of the nominal model used in the forward simulation [kg].
        m_real: true mass to reproduce [kg].

    Returns:
        Residual acceleration in the world frame, shape (3,).
    """
    # third column of R (body z-axis expressed in world frame) and total thrust
    R_body_z = jnp.array([x[5], x[8], x[11]])
    f_d = x[15]
    coeff = 1.0 / m_real - 1.0 / m_nominal
    return coeff * f_d * R_body_z


def get_constant_residual_apply_fn(residual_vec) -> Callable:
    """
    Returns an apply function matching the residual MLP ensemble interface
    (``apply_fn(params, x) -> (num_models, 3)``) but returning a fixed residual
    acceleration regardless of the state. ``params`` is ignored.

    This is the state-invariant approximation of the exact mass residual: near
    hover (R ~ I, f_d ~ const) the exact residual collapses to a constant
    world-frame vector, e.g. ``[0, 0, -5.5]`` for the heavy example quad.

    Drop-in replacement for :func:`get_residual_dyn_model_apply_fn`.
    """
    residual_vec = jnp.asarray(residual_vec, dtype=jnp.float32)

    def apply_fn(params, x):
        # leading singleton axis so downstream jnp.mean(preds, axis=0) is a no-op
        return residual_vec[None, :]

    return apply_fn


def get_mass_residual_apply_fn() -> Callable:
    """
    Returns an apply function matching the residual MLP ensemble interface
    (``apply_fn(params, x) -> (1, 3)``) that computes a mass-mismatch residual
    proportional to thrust, ``a_res = coeff * f_d * R_z``, where the scalar
    ``coeff = 1/m_real - 1/m_nominal`` is carried by ``params`` (a *traced*
    array) instead of being baked in as a Python constant.

    Unlike :func:`get_exact_mass_residual_apply_fn` (which closes over static
    ``m_real`` and forces a re-compile whenever the estimated mass changes), the
    coefficient here is a runtime value, so the JIT trace stays fixed across
    successive finetunes — only the array value changes. This is what lets the
    persistent finetune worker estimate a fresh mass each `f` without recompiling.

    Drop-in replacement for :func:`get_residual_dyn_model_apply_fn`: ``params``
    is the scalar (or 0-d / 1-element array) coefficient.
    """
    def apply_fn(params, x):
        coeff = jnp.asarray(params).reshape(())           # scalar coefficient
        R_body_z = jnp.array([x[5], x[8], x[11]])         # body z-axis in world frame
        f_d = x[15]                                       # total thrust
        # leading singleton axis so the downstream jnp.mean(preds, axis=0) is a no-op
        return (coeff * f_d * R_body_z)[None, :]

    return apply_fn


def get_exact_mass_residual_apply_fn(m_nominal: float, m_real: float) -> Callable:
    """
    Returns an apply function matching the residual MLP ensemble interface
    (``apply_fn(params, x) -> (num_models, 3)``) but computing the exact
    mass-mismatch residual analytically. ``params`` is ignored.

    Drop-in replacement for :func:`get_residual_dyn_model_apply_fn` so it works
    unchanged in both evaluation rollouts and BPTT finetuning.
    """
    def apply_fn(params, x):
        # leading singleton axis so the downstream jnp.mean(preds, axis=0) is a no-op
        return exact_mass_residual(x, m_nominal, m_real)[None, :]

    return apply_fn


@jax.jit
def mse_loss(state: TrainState, x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes the mean squared error loss"""
    preds = state.apply_fn(state.params, x)
    return jnp.mean((preds - y) ** 2)


@jax.jit
def full_loss(state: TrainState, x: jax.Array, y: jax.Array, lambda_reg: float) -> jax.Array:
    """Computes total loss including spectral regularization"""
    preds = state.apply_fn(state.params, x)
    mse = jnp.mean((preds - y) ** 2)
    spec_norm = compute_spectral_norm(state.params)
    return mse + lambda_reg * spec_norm


@jax.jit
def train_step(state: TrainState, x: jax.Array, y: jax.Array, lambda_reg: float) -> TrainState:
    """Performs a single gradient descent update step"""
    def loss_fn(params):
        preds = state.apply_fn(params, x)
        mse = jnp.mean((preds - y) ** 2)
        spec_norm = compute_spectral_norm(params)
        return mse + lambda_reg * spec_norm
    
    # compute gradients with respect to params
    grads = jax.grad(loss_fn)(state.params)

    # update training state using optax optimizer
    return state.apply_gradients(grads=grads)


@jax.jit
def compute_spectral_norm(params: dict) -> jax.Array:
    """
    Approximates regularization by summing the l2-norm of kernel weights.
    """
    reg = 0.0
    for layer in params['params'].values():
        if 'kernel' in layer:
            W = layer['kernel']
            # compute spectral norm (largest singular value)
            reg += jnp.linalg.norm(W, ord=2)
    return reg


def predict_fn(params: dict, x: jax.Array) -> jax.Array:
    """Forward pass function for making predictions"""
    model = ResidualDynamicsMLP([19, 128, 128, 3], initial_scale=1.0)
    return model.apply(params, x)


def init_fn(learning_rate: float, seed: int) -> Tuple[dict, TrainState]:
    """Initializes model parameters and flax trainstate"""
    model = ResidualDynamicsMLP([19, 128, 128, 3], initial_scale=1.0)
    model_params = model.initialize(jax.random.PRNGKey(seed))
    
    # define optimizer
    tx = optax.adam(learning_rate)
    
    train_state = TrainState.create(
        apply_fn=model.apply, params=model_params, tx=tx
    )
    return model_params, train_state


@partial(jax.jit, static_argnames=('lambda_reg', 'num_epochs', 'eval_every'))
def train(
    train_state: TrainState, 
    X: jax.Array, 
    y: jax.Array, 
    lambda_reg: float, 
    num_epochs: int, 
    eval_every: int
) -> TrainState:
    """
    JIT-compiled training loop using jax.lax.scan for performance.
    """
    def scan_fn(carry, epoch):
        current_state = carry

        # execute optimization step
        new_state = train_step(current_state, X, y, lambda_reg)

        def do_log(state_to_log):
            """Helper for conditional side-effect logging"""
            train_mse_loss = mse_loss(state_to_log, X, y)
            train_total_loss = full_loss(state_to_log, X, y, lambda_reg)
            
            jax.debug.print(
                "Epoch {e}/{t} | Train MSE: {mse} | Total Loss: {loss}",
                e=epoch,
                t=num_epochs,
                mse=train_mse_loss,
                loss=train_total_loss,
            )
            return None
        
        # trigger logging only at specified intervals
        jax.lax.cond(epoch % eval_every == 0, do_log, lambda _: None, new_state)

        return new_state, None

    # create epoch range for scan
    epochs = jnp.arange(num_epochs + 1)
    final_state, _ = jax.lax.scan(scan_fn, train_state, xs=epochs, length=num_epochs + 1)

    return final_state


def create_vec_funcs():
    """
    Creates vectorized versions of init, train, and predict functions.
    useful for ensemble training or hyperparameter sweeps.
    """
    parallel_init_fn = jax.vmap(init_fn, in_axes=(None, 0))
    parallel_train_fn = jax.vmap(train, in_axes=(0, None, None, None, None, None))
    parallel_predict_fn = jax.vmap(predict_fn, in_axes=(0, None))

    return parallel_init_fn, parallel_train_fn, parallel_predict_fn
