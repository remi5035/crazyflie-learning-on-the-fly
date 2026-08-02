import copy

from jax.tree_util import tree_map_with_path
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import freeze, unfreeze


def lora_only_mask(params: dict) -> dict:
    """Returns a PyTree mask with True for LoRA params and False elsewhere."""
    
    def is_lora_param(path):
        return any("lora" in str(k).lower() for k in path)

    def mask_fn(tree):
        return tree_map_with_path(lambda path, _: is_lora_param(path), tree)

    return {"params": mask_fn(params["params"])}


def partition_params(params, mask):
    """Splits a param PyTree into (frozen, trainable) using a boolean mask PyTree."""
    flat_params = flatten_dict(unfreeze(params))
    flat_mask = flatten_dict(mask)

    trainable = {}
    frozen = {}

    for key, param in flat_params.items():
        if flat_mask.get(key, False):
            trainable[key] = param
        else:
            frozen[key] = param

    return freeze(unflatten_dict(frozen)), freeze(unflatten_dict(trainable))


def recursive_merge(d1, d2):
    """Recursively merge two nested dicts (PyTrees). d2 takes precedence."""
    merged = copy.deepcopy(d1)
    for k, v in d2.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = recursive_merge(merged[k], v)
        else:
            merged[k] = v
    return merged
