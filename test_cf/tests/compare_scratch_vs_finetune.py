"""Diagnostic finetuning sur un VRAI log : qualité du résidu + base vs finetuné vs scratch.

Répond à deux questions sur un log de vol (réel ou Genesis) :

  A. Le résidu est-il FIABLE ? (pas juste "la moyenne colle")
     - split train/test : fit sur 70 %, MSE sur les 30 % held-out (généralisation) ;
     - plage de thrust des données : si T est quasi constant (hover statique), le
       gradient ∂a/∂T du résidu n'est PAS contraint -> le BPTT peut l'exploiter à tort.

  B. base vs FINETUNÉ (depuis la base) vs SCRATCH (réentraîné à 0), MÊME résidu, MÊMES
     epochs, évalués DANS l'env d'entraînement (HoveringStateEnv + résidu appris). C'est
     le "rejeu dans son propre monde" : si le finetuné hover bien ICI mais mal sur le vrai
     drone -> sim-to-real ; s'il hover mal ICI -> finetune/résidu. scratch≫finetune ->
     ta base est trop ancrée (sous-adaptation).

À lancer dans `.venv` (jax + lotf + torch) :
    cd test_cf
    ../.venv/bin/python tests/compare_scratch_vs_finetune.py --log measurements/real/real_run_lotf.npz
    ../.venv/bin/python tests/compare_scratch_vs_finetune.py --log measurements/online_ft/log.npz \
        --base models/model_pretrain.pt --epochs 100 --window-sec 2.0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import torch
import optax
from flax.training.train_state import TrainState

THIS = Path(__file__).resolve().parent
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())
TEST_CF = REPO / "test_cf"
for p in (TEST_CF / "core", TEST_CF / "RL-real", REPO):
    sys.path.insert(0, str(p))

import cf_params as P                                          # noqa: E402
from lotf.objects import Quadrotor                             # noqa: E402
from lotf.envs import HoveringStateEnv, rollout                # noqa: E402
from lotf.envs.wrappers import MinMaxObservationWrapper        # noqa: E402
from lotf.algos import bptt                                    # noqa: E402

import finetune_lotf_jax as F                                  # noqa: E402
from lotf_residual import build_residual_dataset              # noqa: E402
from lotf_jax_bridge import make_lotf_mlp, torch_sd_to_flax    # noqa: E402


def residual_quality(X, y, train_frac=0.7):
    """Fit sur train, MSE held-out + plage de thrust (le gradient ∂a/∂T en dépend)."""
    n = len(X); idx = np.random.default_rng(0).permutation(n)
    ntr = int(train_frac * n)
    tr, te = idx[:ntr], idx[ntr:]
    params = F.fit_residual_ensemble(X[tr], y[tr])
    _, _, predict_fn = F._vec_funcs()
    pred_te = np.asarray(predict_fn(params, jnp.asarray(X[te]))).mean(0)   # (n_te,3)
    mse_te = float(np.mean((pred_te - y[te]) ** 2))
    mse_naive = float(np.mean((y[te] - y[tr].mean(0)) ** 2))   # baseline : prédire la moyenne
    T = X[:, 15]
    print("\n── A. QUALITÉ DU RÉSIDU ──")
    print(f"  samples={n}  | y moyen={y.mean(0).round(3)}  "
          f"|y|_med={np.median(np.linalg.norm(y, axis=1)):.3f}")
    print(f"  thrust T : min={T.min():.3f} max={T.max():.3f} std={T.std():.4f} N "
          f"({'PLAGE OK' if T.std() > 0.01 else '⚠ QUASI-CONSTANT -> ∂a/∂T non contraint'})")
    print(f"  MSE held-out (30%) = {mse_te:.4f}  vs prédire-la-moyenne = {mse_naive:.4f}  "
          f"({'le MLP généralise' if mse_te < mse_naive else '⚠ pas mieux que la moyenne -> surapprend/bruit'})")


def make_eval_env(res_quad, target):
    """Env d'ENTRAÎNEMENT (crazyflie + résidu appris), non vectorisé, pour évaluer."""
    return MinMaxObservationWrapper(HoveringStateEnv(
        max_steps_in_episode=251, dt=P.DT, delay=P.DELAY, quad_obj=res_quad,
        margin=0.5, hover_target=list(target)))


def eval_in_train_env(params, net, res_quad, res_params, target, n=8):
    env = make_eval_env(res_quad, target)

    def policy(obs, key):
        return net.apply(params, obs)
    keys = jax.random.split(jax.random.key(0), n)
    tr = jax.vmap(rollout, in_axes=(None, 0, None, None))(env, keys, policy, res_params)
    p = np.asarray(tr.state.quadrotor_state.p)              # (n,T,3)
    term = np.asarray(tr.terminated)                        # sortie de boîte = VRAI crash
    done = term | np.asarray(tr.truncated)                  # +fin d'épisode (pour le masque)
    fd = np.where(done.any(1), done.argmax(1), p.shape[1] - 1)
    z_off = np.mean([p[i, max(0, k-50):k+1, 2].mean() for i, k in enumerate(fd)]) - target[2]
    err = np.mean([np.linalg.norm(p[i, max(0, k-50):k+1] - np.asarray(target), axis=1).mean()
                   for i, k in enumerate(fd)])
    n_crash = int(term.any(1).sum())                        # crashs réels (hors fin d'épisode)
    return float(z_off), float(err), n_crash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="measurements/real/real_run_lotf.npz")
    ap.add_argument("--base", default="models/model_pretrain.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--target", type=float, nargs=3, default=list(P.HOVER_GOAL))
    args = ap.parse_args()
    target = args.target

    log_path = (TEST_CF / args.log) if not Path(args.log).is_absolute() else Path(args.log)
    log = {k: v for k, v in np.load(log_path).items()}
    win = None if args.window_sec <= 0 else args.window_sec
    X, y = build_residual_dataset(log, window_sec=win)
    print(f"[diag] log={log_path.name}  window={win}s")

    # A. qualité du résidu
    residual_quality(X, y)

    # résidu "pipeline" (fit sur tout, comme en vol) + quad augmenté pour l'éval
    res_params = F.fit_residual_ensemble(X, y)
    res_quad = Quadrotor.from_name("crazyflie_quad",
                                   {"use_high_fidelity": False, "use_forward_residual": True})

    # contexte BPTT réel (mêmes fonctions que la pipeline)
    ctx = F.build_bptt_context(target, args.epochs)
    net = ctx["policy_net"]

    base_path = (TEST_CF / args.base) if not Path(args.base).is_absolute() else Path(args.base)
    ckpt = torch.load(base_path, map_location="cpu")
    base = torch_sd_to_flax(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
    scratch_init = net.initialize(jax.random.key(123))     # politique RÉINITIALISÉE à 0

    print(f"\n── B. base vs finetuné vs scratch ({args.epochs} epochs, éval dans l'env d'entraînement) ──")
    ft = F.run_bptt(ctx, base, res_params)
    sc = F.run_bptt(ctx, scratch_init, res_params)

    rows = [("base (non adaptée)", base), ("FINETUNÉE (depuis base)", ft),
            ("SCRATCH (réentraînée à 0)", sc)]
    print(f"  {'politique':28s} {'offset z':>9s} {'err pos':>9s} {'crashs/8':>9s}")
    res = {}
    for name, params in rows:
        zo, er, nc = eval_in_train_env(params, net, res_quad, res_params, target)
        res[name] = (zo, er, nc)
        print(f"  {name:28s} {zo:+9.3f} {er:9.3f} {nc:9d}")

    # interprétation
    print("\n── LECTURE ──")
    b, ftv, scv = res["base (non adaptée)"][1], res["FINETUNÉE (depuis base)"][1], res["SCRATCH (réentraînée à 0)"][1]
    if ftv > b - 0.01:
        print("  • Le finetuné n'améliore PAS la base DANS SON PROPRE ENV -> le BPTT/résidu")
        print("    n'optimise pas comme voulu (revoir résidu/epochs/lr), AVANT de blâmer le drone.")
    else:
        print("  • Le finetuné améliore bien la base dans son env -> la brique finetune marche.")
        print("    Si c'est moins bon sur le VRAI drone => écart SIM-TO-REAL (delay/actionneur/")
        print("    bruit capteur/calibration thrust), pas le finetune.")
    if scv < ftv - 0.01:
        print("  • scratch MEILLEUR que finetuné -> base trop ancrée : ↑ epochs ou ↑ lr du finetune.")
    elif scv > ftv + 0.05:
        print("  • scratch PIRE/instable -> la base est essentielle (le from-scratch n'est pas un bon plan).")
    else:
        print("  • scratch ≈ finetuné -> la base n'est ni un frein ni un atout ici.")


if __name__ == "__main__":
    main()
