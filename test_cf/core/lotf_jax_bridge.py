"""Pont de poids entre l'Actor torch (test_cf/RL-real/RL_policy.py) et le MLP
flax de lotf (`lotf.modules.MLP`).

But : permettre à la pipeline Genesis (torch) de FAIRE SA FINETUNE avec lotf-JAX
(comme la pipeline Gazebo de scripts/), puis de revoler le résultat en torch.
Les DEUX réseaux ont la même architecture [27,512,512,4] + biais hover ; il faut
juste (a) transposer les poids (torch Linear = (out,in), flax Dense = (in,out))
et (b) utiliser la MÊME non-linéarité. L'Actor torch utilise **tanh** → on
instancie le MLP lotf avec `nonlinearity=jax.nn.tanh` (le défaut lotf est relu !).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze

from lotf.modules import MLP

NONLIN = jax.nn.tanh   # IMPORTANT : l'Actor torch est en Tanh, pas en ReLU (défaut lotf)


def make_lotf_mlp(obs_dim, action_dim, hover_action, initial_scale=0.01):
    return MLP([obs_dim, 512, 512, action_dim], nonlinearity=NONLIN,
               initial_scale=initial_scale, action_bias=jnp.asarray(hover_action))


def torch_sd_to_flax(sd) -> dict:
    """state_dict torch (actor.{0,2,4}.weight/bias + action_bias) -> params flax.

    flax Dense_i.kernel = weight.T (transpose), bias inchangé.
    """
    def w(k): return np.asarray(sd[k])
    params = {"params": {
        "Dense_0": {"kernel": jnp.asarray(w("actor.0.weight").T), "bias": jnp.asarray(w("actor.0.bias"))},
        "Dense_1": {"kernel": jnp.asarray(w("actor.2.weight").T), "bias": jnp.asarray(w("actor.2.bias"))},
        "Dense_2": {"kernel": jnp.asarray(w("actor.4.weight").T), "bias": jnp.asarray(w("actor.4.bias"))},
    }}
    return freeze(params)


def flax_to_torch_sd(params, action_bias) -> dict:
    """params flax -> state_dict torch chargeable par RL_policy.Actor."""
    import torch
    p = params["params"]
    def k(name): return torch.tensor(np.asarray(p[name]["kernel"]).T.copy())
    def b(name): return torch.tensor(np.asarray(p[name]["bias"]).copy())
    return {
        "action_bias": torch.tensor(np.asarray(action_bias).copy()),
        "actor.0.weight": k("Dense_0"), "actor.0.bias": b("Dense_0"),
        "actor.2.weight": k("Dense_1"), "actor.2.bias": b("Dense_1"),
        "actor.4.weight": k("Dense_2"), "actor.4.bias": b("Dense_2"),
    }
