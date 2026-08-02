"""Simplified BPTT pretraining of the LOTF hover policy on the nominal sim.

Mirrors `examples/state_hovering/1_train_base_policy.ipynb`. Outputs a .pt
that loads directly via `RL_policy.RLPolicy`.

Run:
    python pretrain_lotf.py --epochs 300 --out model_pretrain.pt

Stabilité du BPTT
-----------------
On rétropropage à travers tout le rollout d'une dynamique de quadrirotor, qui
est un équilibre INSTABLE : la sensibilité ∂loss/∂poids (donc la norme de
gradient) croît ~exponentiellement avec `horizon`. Un pic de gradient, amplifié
par un grand `lr`, éjecte la politique du bon bassin → le retour oscille puis
explose. Réglages par défaut choisis pour rester stables :

    horizon = 80     (rollout plus court → produit de Jacobiennes plus court)
    lr      = 5e-4   (petit pas → robuste aux pics de gradient)
    grad-clip = 0.5  (borne directement l'exploding gradient — remède standard)
    num-envs = 128   (moins de variance sur le gradient)

La norme de gradient AVANT clipping est affichée (`gnorm`) : si elle s'envole,
réduis encore `horizon`/`lr`. On sauvegarde le MEILLEUR modèle (par retour) et
pas le dernier, car même stable le retour reste un peu bruité d'une époque à
l'autre.
"""
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn

from RL_policy import Actor
from lotf_sim import CrazyflieDiffSim


def train(epochs, num_envs, horizon, lr, grad_clip, out_path, device, seed):
    torch.manual_seed(seed)
    sim = CrazyflieDiffSim(num_envs=num_envs, device=device)
    policy = Actor().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

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
        opt.zero_grad()
        loss.backward()
        # clip_grad_norm_ renvoie la norme AVANT clipping → utile pour diagnostiquer
        gnorm = nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        opt.step()

        ep_return = total.mean().item()
        if ep_return > best_return:
            best_return = ep_return
            best_state = copy.deepcopy(policy.state_dict())
        if ep % 10 == 0:
            print(f"epoch {ep:4d}  return={ep_return:9.2f}  "
                  f"gnorm={float(gnorm):8.2f}  best={best_return:8.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': best_state}, out_path)
    print(f"[done] best return={best_return:.2f} → saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--num-envs", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=150)    # 80 × 0.02 = 1.6 s rollout
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("../models/model_pretrain_torch.pt"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.epochs, args.num_envs, args.horizon, args.lr, args.grad_clip,
          args.out, args.device, args.seed)
