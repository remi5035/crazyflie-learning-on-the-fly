"""GRU-based recurrent policy for partially observable environments.

The policy encodes the observation with a small MLP, feeds it into a GRU cell,
then decodes the hidden state into an action.  The caller is responsible for
carrying the hidden state across time steps (see lotf/algos/bptt_recurrent.py).
"""

from typing import Union

import jax
import jax.numpy as jnp
from flax import linen as nn


class GRUPolicy(nn.Module):
    """Recurrent policy: MLP encoder → GRU cell → MLP decoder.

    Call signature:
        action, new_hidden = model.apply(params, obs, hidden)

    Args:
        obs_dim:      dimension of the flattened observation vector.
        hidden_size:  number of GRU hidden units.
        action_dim:   dimension of the action vector.
        encoder_size: width of the single hidden encoder layer.
        decoder_size: width of the single hidden decoder layer (0 = linear).
        action_bias:  constant added to the raw output (e.g. hovering thrust).
        initial_scale: scale for output layer initialisation.
    """

    obs_dim: int
    hidden_size: int
    action_dim: int
    encoder_size: int = 256
    decoder_size: int = 256
    action_bias: Union[float, jnp.ndarray] = 0.0
    initial_scale: float = 0.01

    @nn.compact
    def __call__(
        self, obs: jax.Array, hidden: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            obs:    observation vector, shape (obs_dim,).
            hidden: GRU carry from previous step, shape (hidden_size,).

        Returns:
            action:     action vector, shape (action_dim,).
            new_hidden: updated GRU carry, shape (hidden_size,).
        """
        # --- encoder ---
        x = nn.Dense(
            self.encoder_size,
            kernel_init=nn.initializers.variance_scaling(
                1.0, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(obs)
        x = nn.relu(x)

        # --- GRU cell ---
        new_hidden, _ = nn.GRUCell(self.hidden_size)(hidden, x)

        # --- decoder ---
        h = new_hidden
        if self.decoder_size > 0:
            h = nn.Dense(
                self.decoder_size,
                kernel_init=nn.initializers.variance_scaling(
                    1.0, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(h)
            h = nn.relu(h)

        action = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(h)

        return action + self.action_bias, new_hidden

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def initialize(self, key: jax.Array) -> dict:
        """Initialise parameters with a dummy forward pass."""
        obs = jax.random.normal(key, (self.obs_dim,))
        hidden = jnp.zeros((self.hidden_size,))
        return self.init(key, obs, hidden)

    def initial_hidden(self, num_envs: int = 1) -> jax.Array:
        """Returns a zero-initialised carry for *num_envs* parallel envs."""
        if num_envs == 1:
            return jnp.zeros((self.hidden_size,))
        return jnp.zeros((num_envs, self.hidden_size))
