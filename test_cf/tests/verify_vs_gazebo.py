"""Vérifie que l'étage RÉSIDU du port torch (test_cf/Genesis) reproduit l'étage
résidu lotf-JAX utilisé par la pipeline Gazebo (scripts/finetune_lotf.py).

On prend un VRAI rollout Genesis (rollout.npz) et on ajuste le même résidu de
dynamique avec les deux implémentations, puis on compare les prédictions :
  - torch : lotf_residual.fit_residual            (1 MLP [19,128,128,3])
  - JAX   : lotf.utils.residual_dynamics + create_vec_funcs  (ENSEMBLE de 3,
            exactement comme scripts/finetune_lotf.py / le papier Sec. III-B)

Exécutable dans .venv (jax+lotf+torch présents). NE touche pas à Genesis.
"""
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent                  # tests/
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())   # racine du dépôt
TEST_CF = REPO / "test_cf"
sys.path.insert(0, str(TEST_CF / "RL-real"))
sys.path.insert(0, str(REPO))

# --- dataset commun (construit par le port torch, formule identique aux 2 pipelines) ---
from lotf_residual import build_residual_dataset, fit_residual   # torch (test_cf)
import torch

# --- étage résidu lotf-JAX (référence Gazebo) ---
import jax, jax.numpy as jnp
from lotf.utils.residual_dynamics import create_vec_funcs

# hyperparams résidu = ceux de scripts/finetune_lotf.py
FT_NUM_MODELS, FT_RES_LR, FT_RES_LAMBDA, FT_RES_EPOCHS, FT_RES_EVAL = 3, 1e-2, 1e-3, 100, 10


def fit_jax_ensemble(X, y):
    init_fn, train_fn, predict_fn = create_vec_funcs()
    _, states = init_fn(FT_RES_LR, jnp.arange(FT_NUM_MODELS, dtype=jnp.int32))
    states = train_fn(states, jnp.asarray(X), jnp.asarray(y),
                      FT_RES_LAMBDA, FT_RES_EPOCHS, FT_RES_EVAL)
    preds = predict_fn(states.params, jnp.asarray(X))      # (num_models, N, 3)
    return np.asarray(jnp.mean(preds, axis=0))


def main(npz="rollout.npz", window_sec=None):
    npz_path = Path(npz) if Path(npz).is_absolute() else TEST_CF / "data" / npz
    log = {k: v for k, v in np.load(npz_path).items()}
    X, y = build_residual_dataset(log, window_sec=window_sec)
    print(f"[data] {npz}: X{X.shape}  |y|_median={np.median(np.linalg.norm(y,1)):.3f} m/s²")
    print(f"       y mean = {y.mean(0).round(3)}  (le terme z = écart Genesis cf2x ↔ modèle nominal)")

    # torch (test_cf)
    torch.manual_seed(0)
    m = fit_residual(X, y, epochs=200, device="cpu", verbose=False)
    with torch.no_grad():
        pred_torch = m(torch.from_numpy(X)).numpy()

    # lotf-JAX (Gazebo)
    pred_jax = fit_jax_ensemble(X, y)

    print("\n[résultats] résidu appris : torch (test_cf) vs lotf-JAX (Gazebo)")
    print(f"  moyenne torch    = {pred_torch.mean(0).round(3)}")
    print(f"  moyenne JAX      = {pred_jax.mean(0).round(3)}")
    print(f"  moyenne cible y  = {y.mean(0).round(3)}")
    for ax, nm in enumerate("xyz"):
        rmse_tj = np.sqrt(np.mean((pred_torch[:, ax] - pred_jax[:, ax]) ** 2))
        corr = np.corrcoef(pred_torch[:, ax], pred_jax[:, ax])[0, 1]
        rt = np.sqrt(np.mean((pred_torch[:, ax] - y[:, ax]) ** 2))
        rj = np.sqrt(np.mean((pred_jax[:, ax] - y[:, ax]) ** 2))
        print(f"  axe {nm}: RMSE(torch↔JAX)={rmse_tj:6.3f}  corr={corr:+.3f}  | "
              f"RMSE/y torch={rt:.3f} JAX={rj:.3f}")
    print(f"  RMSE global torch↔JAX = {np.sqrt(np.mean((pred_torch-pred_jax)**2)):.3f} m/s²")


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "rollout.npz"
    main(npz)
