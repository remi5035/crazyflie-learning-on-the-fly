"""LOTF online finetune: fit residual → short BPTT against augmented sim.

Run after a flight:
    python finetune_lotf.py --base model_pretrain.pt --log run1_lotf.npz \
        --epochs 30 --out model_ft.pt
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from RL_policy import Actor
from lotf_sim import CrazyflieDiffSim
from lotf_residual import build_residual_dataset, fit_residual, load_log_npz


def bptt_finetune(policy, residual_model, *, epochs, num_envs, horizon, lr,
                  grad_clip, device):
    # Même précaution de stabilité que pretrain_lotf : le sim augmenté du résiduel
    # reste un quadrirotor instable → on garde un horizon court + grad-clip serré +
    # sauvegarde du meilleur modèle (cf. README, section stabilité BPTT).
    residual_model.eval()
    for p in residual_model.parameters():
        p.requires_grad_(False)

    sim = CrazyflieDiffSim(num_envs=num_envs, device=device, residual_model=residual_model)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_return = -float("inf")
    best_state = copy.deepcopy(policy.state_dict())

    for ep in range(epochs):
        obs = sim.reset()
        total = torch.zeros(num_envs, device=device)
        for _ in range(horizon):
            a = policy(obs)
            obs, r = sim.step(a)
            total = total + r
        loss = -total.mean()
        opt.zero_grad(); loss.backward()
        gnorm = nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        opt.step(); sched.step()

        ep_return = total.mean().item()
        if ep_return > best_return:
            best_return = ep_return
            best_state = copy.deepcopy(policy.state_dict())
        print(f"  [bptt] ep={ep:3d}  return={ep_return:9.2f}  "
              f"gnorm={float(gnorm):8.2f}  best={best_return:8.2f}")

    policy.load_state_dict(best_state)
    return policy


def finetune_from_log(base_state, log, *, window_sec=2.0, res_epochs=200,
                      bptt_epochs=30, num_envs=10, horizon=80, lr=1e-4,
                      grad_clip=0.3, device="cpu", min_samples=50, verbose=True):
    """Full LOTF update from a flight log → returns (new_policy_state, residual_model).

    Reusable by both the offline CLI (main) and the in-flight hot-swap loop
    (drone_controller). Raises ValueError if the window has too few clean samples.
    """
    X, y = build_residual_dataset(log, window_sec=window_sec)
    if verbose:
        print(f"  [ft] {X.shape[0]} samples,  |y|_median="
              f"{np.median(np.linalg.norm(y, axis=1)):.3f} m/s²")
    if X.shape[0] < min_samples:
        raise ValueError(f"not enough usable samples ({X.shape[0]} < {min_samples})")

    res_model = fit_residual(X, y, epochs=res_epochs, device=device, verbose=verbose)

    policy = Actor().to(device)
    policy.load_state_dict(base_state, strict=True)
    bptt_finetune(policy, res_model, epochs=bptt_epochs, num_envs=num_envs,
                  horizon=horizon, lr=lr, grad_clip=grad_clip, device=device)
    return policy.state_dict(), res_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("../models/model_ft_torch.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--res-epochs", type=int, default=200)
    ap.add_argument("--window-sec", type=float, default=2.0,
                    help="Keep only the last N seconds of the flight log for the "
                         "residual fit (LOTF paper uses ~2 s). 0 = whole flight.")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    window = None if args.window_sec <= 0 else args.window_sec
    print(f"[finetune] residual+BPTT from {args.log}"
          + (f" (last {window:.1f} s)" if window else " (whole flight)"))
    log = load_log_npz(args.log)

    ckpt = torch.load(args.base, map_location=args.device)
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    try:
        new_state, res_model = finetune_from_log(
            sd, log, window_sec=window, res_epochs=args.res_epochs,
            bptt_epochs=args.epochs, num_envs=args.num_envs, horizon=args.horizon,
            lr=args.lr, grad_clip=args.grad_clip, device=args.device)
    except ValueError as e:
        raise SystemExit(f"{e} — fly longer or increase --window-sec.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': new_state}, args.out)
    torch.save(res_model.state_dict(), args.out.with_suffix('.residual.pt'))
    print(f"[done] policy   → {args.out}")
    print(f"       residual → {args.out.with_suffix('.residual.pt')}")


if __name__ == "__main__":
    main()
