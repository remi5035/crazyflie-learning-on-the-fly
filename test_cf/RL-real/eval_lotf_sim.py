"""Teste un checkpoint sur l'environnement d'entraînement d'origine (lotf_sim).

C'est le pendant de `pretrain_lotf.py` côté évaluation : on charge une politique
(`RL_policy.RLPolicy`) et on la déroule dans `CrazyflieDiffSim` — exactement le
sim sur lequel elle a été entraînée — pour vérifier qu'elle plane AVANT le pont
Genesis (`../lotf_genesis_eval.py`) ou le vrai drone (`test_main.py`).

Exemples :
    python eval_lotf_sim.py --ckpt model_pretrain.pt
    python eval_lotf_sim.py --ckpt model_pretrain.pt --num-envs 64 --steps 250
    python eval_lotf_sim.py --ckpt model_pretrain.pt --plot eval_lotf.png

Critère « ça plane » : erreur de position finale de l'ordre de quelques cm.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from RL_policy import RLPolicy
from lotf_sim import CrazyflieDiffSim
import cf_params as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="../models/model_pretrain.pt",
                    help="Checkpoint .pt (27->512->512->4).")
    ap.add_argument("--num-envs", type=int, default=64,
                    help="Rollouts en parallèle (stats moyennées dessus).")
    ap.add_argument("--steps", type=int, default=250,
                    help="Pas de sim (250 × 0.02 = 5 s).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", type=str, default=None,
                    help="Si fourni, sauvegarde un PNG (erreur pos + thrust moyen).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    agent = RLPolicy(args.ckpt)
    sim = CrazyflieDiffSim(num_envs=args.num_envs)

    obs = sim.reset()
    pos_err, thrust, omega_mag = [], [], []
    with torch.no_grad():
        for _ in range(args.steps):
            a = agent.get_action(obs)                       # (N,4) SI, déjà clippé
            obs, _ = sim.step(a)
            pos_err.append((sim.p_pos - sim.goal).norm(dim=-1).mean().item())
            thrust.append(a[:, 0].mean().item())
            omega_mag.append(a[:, 1:].norm(dim=-1).mean().item())

    pos_err = np.array(pos_err)
    tail = max(1, int(2.0 / P.DT))                          # 2 dernières secondes
    print(f"\n=== eval lotf_sim ({args.num_envs} envs, {args.steps} pas, "
          f"{args.steps * P.DT:.1f} s) ===")
    print(f"erreur position : moyenne={pos_err.mean():.3f} m | "
          f"finale={pos_err[-1]:.3f} m | max={pos_err.max():.3f} m")
    print(f"erreur moy. 2 dernières s : {pos_err[-tail:].mean():.3f} m")
    print(f"thrust moyen (régime établi) : {np.mean(thrust[-tail:]):.3f} N "
          f"(hover idéal = {P.T_HOVER:.3f} N)")
    verdict = "PLANE [OK]" if pos_err[-tail:].mean() < 0.1 else "NE PLANE PAS [X]"
    print(f"verdict : {verdict}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(args.steps) * P.DT
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(t, pos_err, label="erreur position (m)")
        ax[0].axhline(0.1, color="r", ls="--", alpha=0.5, label="seuil 0.1 m")
        ax[0].set_ylabel("‖p − goal‖ (m)"); ax[0].grid(True); ax[0].legend()
        ax[0].set_title(f"Eval lotf_sim — {Path(args.ckpt).name}")
        ax[1].plot(t, thrust, label="thrust moyen (N)")
        ax[1].axhline(P.T_HOVER, color="g", ls="--", alpha=0.5, label="T_hover")
        ax[1].set_ylabel("thrust (N)"); ax[1].set_xlabel("temps (s)")
        ax[1].grid(True); ax[1].legend()
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"[plot] -> {args.plot}")


if __name__ == "__main__":
    main()
