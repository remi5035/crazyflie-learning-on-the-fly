from typing import Union, List

import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """
    Standard multi-layer perceptron module.

    Example:
        >>> network = MLP([2, 3, 1])
        >>> key = jax.random.key(0)
        >>> params = network.initialize(key)
    """

    feature_list: list
    nonlinearity: callable = nn.relu
    initial_scale: float = 1.0
    action_bias: Union[float, jnp.ndarray] = 0.0

    @nn.compact
    def __call__(self, x):
        """Defines the forward pass for the mlp."""
        # iterate through hidden layers
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
            
        # apply final output layer
        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        return x + self.action_bias

    def initialize(self, key):
        """Initializes the model parameters using a dummy input."""
        x_rand = jax.random.normal(key, (self.feature_list[0],))
        return self.init(key, x_rand)
    

class LoRADense(nn.Module):
    """Low-rank adaptation (LoRA) layer for dense weights."""
    r: int
    features: int
    lora_alpha: float = 1.0

    @nn.compact
    def __call__(self, x):
        """Applies the lora bypass to the input."""
        in_features = x.shape[-1]
        scaling = self.lora_alpha / self.r

        # initialize low-rank matrices
        A_stddev = 1 / jnp.sqrt(self.r)
        lora_A = self.param("lora_A", nn.initializers.normal(stddev=A_stddev), (in_features, self.r))
        lora_B = self.param("lora_B", nn.initializers.zeros, (self.r, self.features))

        return x @ lora_A @ lora_B * scaling


class LoraMLP(nn.Module):
    """MLP wrapper that adds lora adapters to each layer of a base MLP."""
    
    base_mlp: MLP  # pre-initialized base mlp module
    lora_ranks: list[int]  # LoRA ranks for each layer
    lora_alpha: float = 1.0

    @nn.compact
    def __call__(self, x):
        """Defines the forward pass combining base weights and lora adapters."""
        layer_sizes = self.base_mlp.feature_list
        num_layers = len(layer_sizes) - 1
        assert len(self.lora_ranks) == num_layers, "LoRA ranks must match number of layers"
        act_fn = self.base_mlp.nonlinearity

        for i in range(num_layers - 1):
            # compute base path
            x_base = nn.Dense(
                features=layer_sizes[i + 1],
                name=f"base_dense_{i}",
                use_bias=True,
                kernel_init=nn.initializers.zeros,  # weights loaded later via initialize_with_base
                bias_init=nn.initializers.zeros,
            )(x)
            
            # compute lora path
            rank = self.lora_ranks[i]
            if rank == 0:
                x = x_base
            else:
                x_lora = LoRADense(
                    r=rank, 
                    features=layer_sizes[i + 1], 
                    lora_alpha=self.lora_alpha,
                    name=f"lora_dense_{i}"
                )(x)
                x = x_base + x_lora
            x = act_fn(x)

        # final output layer processing
        x_base = nn.Dense(
            features=layer_sizes[-1],
            name=f"base_dense_{num_layers - 1}",
            use_bias=True,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)
        
        rank = self.lora_ranks[-1]
        if rank == 0:
            x = x_base
        else:
            x_lora = LoRADense(
                r=rank, 
                features=layer_sizes[-1], 
                lora_alpha=self.lora_alpha,
                name=f"lora_dense_{num_layers - 1}"
            )(x)
            x = x_base + x_lora
        
        return x + self.base_mlp.action_bias
    

    def initialize_with_base(self, key, base_params):
        """Initializes LoRA parameters and injects frozen base weights."""
        x_rand = jax.random.normal(key, (self.base_mlp.feature_list[0],))
        lora_params = self.init(key, x_rand)

        # map base MLP weights into the corresponding lora base dense parameters
        for i in range(len(self.base_mlp.feature_list) - 1):
            base_layer = f'Dense_{i}'
            lora_base_layer = f'base_dense_{i}'
            lora_params['params'][lora_base_layer] = base_params['params'][base_layer]

        return lora_params
    


class ResidualDynamicsMLP(nn.Module):
    """MLP designed to model residual dynamics."""

    feature_list: List[int]
    nonlinearity: callable = nn.relu
    initial_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        """Computes the residual dynamics forward pass."""
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
            
        # final delta prediction layer
        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)
        return x

    def initialize(self, key):
        """Initializes the residual model parameters."""
        x_rand = jax.random.normal(key, (self.feature_list[0],))
        return self.init(key, x_rand)
