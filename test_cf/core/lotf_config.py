"""Chargeur de `configs/lotf_config.yaml` — source unique des hyperparamètres de la
pipeline LOTF/Genesis (test_cf/). Importé par finetune_lotf_jax.py,
finetune_lotf_worker.py et lotf_genesis_eval.py.

    from lotf_config import CFG
    CFG["bptt"]["lr"]          # accès dict
    CFG.bptt.lr                # accès attribut (équivalent)

Les constantes plateforme/env physiques restent dans cf_params.py (qui fait
foi) ; le bloc `platform_ref` du YAML n'est qu'une recopie de référence.
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "lotf_config.yaml"


class _Section(dict):
    """dict avec accès attribut (CFG.bptt.lr) en plus de l'accès clé."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _wrap(obj):
    if isinstance(obj, dict):
        return _Section({k: _wrap(v) for k, v in obj.items()})
    return obj


def load(path: Path | str = CONFIG_PATH) -> _Section:
    with open(path, "r") as f:
        return _wrap(yaml.safe_load(f))


CFG = load()
