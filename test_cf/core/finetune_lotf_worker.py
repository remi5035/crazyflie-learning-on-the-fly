"""Worker JAX persistant pour le finetune LOTF — supprime le cold-start.

Le problème : lancer `finetune_lotf_jax.py` en sous-processus à chaque finetune
repaie ~25 s d'`import jax`/`lotf` + JIT à chaque fois. Ce worker garde un
processus CHAUD : il importe et compile UNE fois (warmup), puis sert chaque
finetune en quelques secondes.

Architecture inchangée vs Gazebo : la finetune reste un PROCESSUS séparé (venv
jax+lotf), mais long-vécu. `lotf_genesis_eval.py --online-finetune` le lance une
fois au démarrage et lui envoie des requêtes pendant le vol.

Protocole (texte, 1 ligne par message ; stderr fusionné dans stdout) :
    worker -> @READY                         (warmup terminé, prêt)
    parent -> {"base": "...", "log": "...", "out": "..."}\\n   (sur stdin)
    worker -> ...lignes de progression...
    worker -> @RESULT {"ok": true, "out": "...", "sec": 1.2, "n": 96, "ymed": 8.8}

Config FIXE en argv (--target/--epochs/--window-sec/--num-samples) : tout ce dont
dépend la compilation JIT doit rester constant pour garder les caches chauds —
l'env du BPTT (epochs, num_envs) et le nombre d'échantillons du fit résiduel.
On ré-échantillonne donc chaque rollout à `--num-samples` lignes exactement.

Usage :
    python finetune_lotf_worker.py --target 0 0 0.5 --epochs 15 --window-sec 2.0
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve().parent                  # core/
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())   # racine du dépôt
sys.path.insert(0, str(REPO / "test_cf" / "RL-real"))
sys.path.insert(0, str(REPO))

import jax                                              # noqa: E402
import jax.numpy as jnp                                 # noqa: E402
import cf_params as P                                   # noqa: E402
from lotf_config import CFG                              # noqa: E402  (source unique des hyperparams)
from lotf_residual import build_residual_dataset        # noqa: E402
from lotf_jax_bridge import torch_sd_to_flax, flax_to_torch_sd  # noqa: E402
import finetune_lotf_jax as F                           # noqa: E402  (contexte + fit + bptt)


def emit(line: str = "") -> None:
    """Écrit une ligne sur stdout et flush (le parent lit ligne par ligne)."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def resample(X, y, n, rng=None):
    """Ré-échantillonne (avec remise) X,y à EXACTEMENT n lignes -> shape JIT fixe.

    Le fit résiduel minimise une MSE moyenne sur l'échantillon ; ré-échantillonner
    à n lignes préserve (en espérance) la distribution empirique du rollout tout
    en gelant la shape compilée, donc en gardant le cache JIT chaud d'un appel à
    l'autre (le nb de lignes utiles varie sinon avec la durée/le nettoyage).

    `rng` : np.random.Generator seedé -> finetune REPRODUCTIBLE (sinon les indices
    tirés varient d'un appel à l'autre, ce qui rendait le résultat non déterministe
    à données égales). Défaut np.random (compat appelants externes seedant le global)."""
    in_dim = X.shape[1] if X.ndim == 2 and X.shape[1] else 19
    m = X.shape[0]
    if m == 0:
        return np.zeros((n, in_dim), np.float32), np.zeros((n, 3), np.float32)
    if m >= n:
        # assez de données -> SOUS-échantillonnage SANS remise (sous-ensemble propre).
        idx = np.arange(n) if m == n else (
            rng.permutation(m)[:n] if rng is not None else np.random.permutation(m)[:n])
    else:
        # fenêtre plus courte que n (ex. 2 s -> 99 samples vs num_samples=100) : NE PAS
        # bootstrap-upsampler avec remise. Sur un hover quasi-statique (samples quasi
        # identiques), un bootstrap aléatoire repondère ~1/3 des points et rend le résidu
        # MLP mal conditionné -> le BPTT 'exploite' ses gradients parasites et DIVERGE
        # (offset 0.13 m -> >1 m, validé). On répète les m samples de façon DÉTERMINISTE
        # et uniforme jusqu'à n (shape JIT fixe préservée, pas de repondération aléatoire).
        idx = np.arange(n) % m
    return X[idx], y[idx]


def load_base_flax(base_path):
    ckpt = torch.load(base_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    return torch_sd_to_flax(sd)


def do_finetune(ctx, req, win, n_samples, res_mode):
    """Une finetune (caches chauds) : résidu -> BPTT -> .pt torch. Renvoie un dict."""
    t0 = time.time()
    base_flax = load_base_flax(req["base"])

    log_path = Path(req["log"])
    log = {k: v for k, v in np.load(log_path).items()}
    X, y = build_residual_dataset(log, window_sec=win)
    ymed = float(np.median(np.linalg.norm(y, axis=1))) if len(y) else 0.0
    Xr, yr = resample(X, y, n_samples, rng=np.random.default_rng(0))   # reproductible

    res_params = F.fit_residual_ensemble(Xr, yr, mode=res_mode)   # cache chaud (N fixe)
    new_flax = F.run_bptt(ctx, base_flax, res_params)    # cache chaud (env/epochs fixes)

    new_sd = flax_to_torch_sd(new_flax, P.HOVERING_ACTION)
    out = Path(req["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": new_sd}, out)
    return {"ok": True, "out": str(out), "sec": round(time.time() - t0, 2),
            "n": int(len(X)), "ymed": round(ymed, 3)}


def main():
    ap = argparse.ArgumentParser()
    O = CFG["online"]
    ap.add_argument("--target", type=float, nargs=3, default=list(P.HOVER_GOAL))
    ap.add_argument("--epochs", type=int, default=O["bptt_epochs"], help="epochs BPTT (FIXE)")
    ap.add_argument("--window-sec", type=float, default=O["window_sec"])
    ap.add_argument("--num-samples", type=int, default=O["num_samples"],
                    help="taille FIXE du dataset résiduel (ré-échantillonné) pour garder le JIT chaud")
    args = ap.parse_args()
    win = None if args.window_sec <= 0 else args.window_sec
    n_samples = args.num_samples
    res_mode = O.get("residual_mode", "mlp")             # 'mlp' | 'constant' (cf. lotf_config)

    # ── warmup : build env + JIT-compile fit résiduel ET bptt, une seule fois ──
    t0 = time.time()
    ctx = F.build_bptt_context(args.target, args.epochs, res_mode=res_mode)
    dummy_X = np.zeros((n_samples, 19), np.float32)
    dummy_y = np.zeros((n_samples, 3), np.float32)
    res_d = F.fit_residual_ensemble(dummy_X, dummy_y, mode=res_mode)
    base_d = ctx["policy_net"].init(jax.random.key(0), jnp.zeros((ctx["obs_dim"],)))
    new_d = F.run_bptt(ctx, base_d, res_d)
    jax.block_until_ready(jax.tree_util.tree_leaves(new_d))
    emit(f"[worker] warmup terminé en {time.time() - t0:.1f}s "
         f"(target={[float(x) for x in args.target]}, epochs={args.epochs}, "
         f"N={n_samples}, résidu={res_mode})")
    emit("@READY")

    # ── boucle de service : 1 requête JSON par ligne ──
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            emit(f"@RESULT {json.dumps({'ok': False, 'error': f'JSON invalide: {e}'})}")
            continue
        if req.get("cmd") == "quit":
            break
        try:
            result = do_finetune(ctx, req, win, n_samples, res_mode)
        except Exception as e:                           # noqa: BLE001
            traceback.print_exc(file=sys.stdout)
            result = {"ok": False, "error": str(e)}
        emit(f"@RESULT {json.dumps(result)}")


if __name__ == "__main__":
    main()
