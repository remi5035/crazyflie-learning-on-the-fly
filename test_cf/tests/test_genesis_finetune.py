"""Test simple « learning-on-the-fly » DANS Genesis, en réutilisant lotf.

Boucle du papier (Sec. III-B) : voler -> log -> fit résidu + BPTT (lotf-JAX) ->
revoler, et VÉRIFIER que l'offset de hover diminue. La finetune tourne en
SOUS-PROCESSUS via `finetune_lotf_jax.py` dans un venv qui contient jax+lotf
(par défaut ../.venv), exactement comme la pipeline Gazebo lance un nœud JAX
séparé. Le venv Genesis (torch) n'a donc PAS besoin de JAX.

À lancer dans le venv Genesis (torch + genesis) :
    python test_genesis_finetune.py --ckpt models/model_pretrain.pt
    python test_genesis_finetune.py --ckpt models/model_pretrain.pt \
        --jax-python /chemin/vers/.venv/bin/python --steps 600

Vérifie :
  1. la politique de base vole et on logge un rollout Genesis (format _lotf.npz) ;
  2. finetune lotf-JAX (résidu ensemble + BPTT HoveringStateEnv(crazyflie_quad)) ;
  3. la politique finetunée vole dans Genesis et l'offset |z-cible| baisse.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve().parent                  # tests/
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())   # racine du dépôt
TEST_CF = REPO / "test_cf"
sys.path.insert(0, str(TEST_CF / "core"))
sys.path.insert(0, str(TEST_CF / "RL-real"))
import cf_params as P                              # noqa: E402
from drone_state import _quat_wxyz_to_R           # noqa: E402


def rollout(env, agent, steps, goal):
    """Vole `agent` dans `env` pendant `steps`, renvoie (log_dict, offset_z_final)."""
    obs, _ = env.reset()
    log = {'t': [], 'p': [], 'R': [], 'v': [], 'T_N': [], 'omega': []}
    zs = []
    with torch.no_grad():
        for i in range(steps):
            a = agent.get_action(obs)
            obs, info = env.step(a)
            an = a.cpu().numpy(); lat = env.state.latest
            log['t'].append(i * P.DT)
            log['p'].append(lat['pos'].copy())
            log['R'].append(_quat_wxyz_to_R(lat['quat_wxyz']).flatten())
            log['v'].append(lat['vel'].copy())
            log['T_N'].append(float(an[0]))
            log['omega'].append(an[1:].copy())
            zs.append(info['pos'][2])
    z_final = float(np.mean(zs[-int(2 / P.DT):]))   # 2 dernières s
    return log, abs(z_final - goal[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/model_pretrain.pt")
    ap.add_argument("--jax-python", default=str(REPO / ".venv" / "bin" / "python"),
                    help="python d'un venv avec jax+lotf+torch (lance finetune_lotf_jax.py)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--goal", type=float, nargs=3, default=list(P.HOVER_GOAL))
    ap.add_argument("--no-viewer", action="store_true", default=True)
    args = ap.parse_args()

    import genesis as gs
    gs.init(logging_level="warning")
    from lotf_genesis_env import LotfHoverEnv
    from RL_policy import RLPolicy

    env = LotfHoverEnv(goal=args.goal, show_viewer=not args.no_viewer, visualize_target=False)
    goal = np.asarray(args.goal)

    # Genesis force cuda comme device torch par défaut ; la pipeline real (obs
    # DroneState numpy) tourne sur CPU -> on crée les politiques sous contexte CPU.
    # 1) baseline
    with torch.device("cpu"):
        base = RLPolicy(str(TEST_CF / args.ckpt))
    log0, off0 = rollout(env, base, args.steps, goal)
    log_path = TEST_CF / "data" / "_genesis_ft_log.npz"
    np.savez_compressed(log_path, **{k: np.asarray(v) for k, v in log0.items()})
    print(f"\n[1] baseline : offset |z-cible| = {off0:.3f} m  (log -> {log_path.name})")

    # 2) finetune lotf-JAX en sous-processus
    out_ckpt = TEST_CF / "models" / "model_ft_genesis.pt"
    cmd = [args.jax_python, str(TEST_CF / "core" / "finetune_lotf_jax.py"),
           "--base", str(TEST_CF / args.ckpt), "--log", str(log_path),
           "--out", str(out_ckpt), "--target", *map(str, args.goal),
           "--epochs", str(args.epochs)]
    print(f"[2] finetune lotf-JAX : {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-800:])
    if r.returncode != 0:
        print("STDERR:\n", r.stderr[-1500:]); sys.exit("finetune échouée")

    # 3) revol de la politique finetunée
    with torch.device("cpu"):
        ft = RLPolicy(str(out_ckpt))
    _, off1 = rollout(env, ft, args.steps, goal)
    print(f"\n[3] finetunée : offset |z-cible| = {off1:.3f} m")
    print(f"\n=== VERDICT : offset {off0:.3f} -> {off1:.3f} m  "
          f"({'AMÉLIORÉ' if off1 < off0 else 'PAS AMÉLIORÉ'}) ===")


if __name__ == "__main__":
    main()
