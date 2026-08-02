"""Vérifie que le MLP de résiduel apprend bien l'effet du changement de masse,
dans Genesis (équivalent de `examples/residual_dynamics/learned_residual_offset_demo.ipynb`,
partie « 3a : résidu APPRIS vs résidu EXACT », mais sur CE simulateur).

But (1re étape, MLP seul) : voler le drone Genesis sous la politique nominale,
acquérir un log de vol, ajuster le MLP de résiduel sur `y = a_mesurée - a_nominale`
(différence finie de la vitesse, EXACTEMENT le pipeline réel `build_residual_dataset`),
puis comparer la prédiction du MLP au résiduel ANALYTIQUE exact du décalage de masse :

    a_res^exact = R @ [0, 0, T_N * (1/m_real - 1/m_nom)]

avec :
  * m_nom  = cf_params.MASS (= masse que `a_nom` suppose, 0.027 kg)
  * m_real = env.mass        (= masse RÉELLEMENT simulée par Genesis)

Comme l'actionneur Genesis est « honnête » (il délivre vraiment T_N Newtons, cf
`lotf_genesis_env.genesis_force_to_pwm_pct`), tout l'écart `a_mes - a_nom` provient
du décalage de masse : le MLP devrait donc reproduire la formule exacte ci-dessus.

Aucun JAX : acquisition Genesis + entraînement MLP (torch) + comparaison (numpy)
tournent tous dans le venv Genesis.

Exemple :
    source ../../genesis_venv/bin/activate         # depuis test_cf/core
    python core/lotf_residual_check.py --ckpt models/model_pretrain.pt --no-viewer \
        --steps 1500 --mass 0.042
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import genesis as gs

THIS_DIR = Path(__file__).resolve().parent              # core/
REPO = next(p for p in THIS_DIR.parents if (p / "lotf").is_dir())   # racine du dépôt
TEST_CF = REPO / "test_cf"
REAL_DIR = TEST_CF / "RL-real"
if str(REAL_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DIR))

from lotf_genesis_env import LotfHoverEnv          # noqa: E402
import cf_params as P                              # noqa: E402
from RL_policy import RLPolicy                     # noqa: E402  (Actor LOTF + biais hover)
from drone_state import _quat_wxyz_to_R           # noqa: E402  (même conversion que le vrai log)
from lotf_residual import build_residual_dataset, fit_residual   # noqa: E402  (pipeline réelle)


def exact_mass_residual(X: np.ndarray, m_nom: float, m_real: float) -> np.ndarray:
    """Résiduel analytique exact du décalage de masse, sur les features 19-d.

    Features : [p(3) | R aplatie ligne-major(9) | v(3) | T_N(1) | omega(3)].
    R aplatie ligne-major -> 3e colonne (axe z corps en monde) = indices 5, 8, 11.
    a_res = R @ [0,0, T_N*(1/m_real - 1/m_nom)] = coeff * T_N * R[:, :, 2].
    """
    R_body_z = X[:, [5, 8, 11]]                 # (N, 3) axe z du corps en repère monde
    T_N = X[:, 15:16]                           # (N, 1)
    coeff = 1.0 / m_real - 1.0 / m_nom
    return coeff * T_N * R_body_z


def acquire_log(env, agent, steps, waypoints=None):
    """Vole `steps` pas sous `agent` et enregistre le log SI (format finetune_lotf).

    Si `waypoints` est fourni, on cycle la cible toutes les ~steps/len pas pour
    exciter l'inclinaison (R) et la poussée (T_N) -> données plus variées que le
    pur hover (sinon le résidu x/y est ~constant nul et la corrélation est mal
    définie). Le résiduel reste dominé par l'axe z (perte de poussée)."""
    log = {'t': [], 'p': [], 'R': [], 'v': [], 'T_N': [], 'omega': []}
    obs, _ = env.reset()
    seg = max(1, steps // len(waypoints)) if waypoints else steps
    with torch.no_grad():
        for i in range(steps):
            if waypoints and i % seg == 0:
                env.set_goal(waypoints[(i // seg) % len(waypoints)])
            action = agent.get_action(obs)
            obs, info = env.step(action)
            a = action.cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
            lat = env.state.latest
            log['t'].append(i * P.DT)
            log['p'].append(lat['pos'].copy())
            log['R'].append(_quat_wxyz_to_R(lat['quat_wxyz']).flatten())
            log['v'].append(lat['vel'].copy())
            log['T_N'].append(float(a[0]))
            log['omega'].append(a[1:].copy())
            if i % 100 == 0:
                p = info["pos"]
                print(f"  step {i:4d} | pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"| err={info['pos_error']:.3f} m | T={info['thrust_N']:.3f} N")
    for k in log:
        log[k] = np.asarray(log[k])
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/model_pretrain.pt",
                    help="Checkpoint .pt de la politique nominale (27->512->512->4).")
    ap.add_argument("--steps", type=int, default=1500, help="Pas de politique (50 Hz). 1500 = 30 s.")
    ap.add_argument("--mass", type=float, default=None,
                    help="Masse TOTALE Genesis [kg] (défaut = masse URDF cf2x). "
                         "m_real du résiduel exact. m_nom = cf_params.MASS.")
    ap.add_argument("--res-epochs", type=int, default=400, help="Époques d'entraînement du MLP résiduel.")
    ap.add_argument("--no-viewer", action="store_true", help="Désactive la fenêtre 3D.")
    ap.add_argument("--plot", type=str, default="assets/figures/residual_check.png",
                    help="Fichier PNG du nuage MLP vs exact (relatif à test_cf/).")
    ap.add_argument("--save-dataset", type=str, default=None,
                    help="Si fourni, sauvegarde X,y du dataset acquis (.npz).")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    gs.init(logging_level=None if args.verbose else "warning")

    ckpt_path = (TEST_CF / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    with torch.device("cpu"):                       # obs DroneState (numpy) + réseau sur CPU
        agent = RLPolicy(str(ckpt_path))

    env = LotfHoverEnv(show_viewer=not args.no_viewer, visualize_target=True)

    m_nom = float(P.MASS)                            # masse supposée par a_nom (cf_params)
    if args.mass is not None:
        env.set_total_mass(args.mass)
    m_real = float(env.mass)                         # masse réellement simulée par Genesis

    print("\n=== 1. Acquisition du dataset (vol sous politique nominale) ===")
    print(f"  m_nom (a_nom, cf_params.MASS) = {m_nom*1000:.1f} g")
    print(f"  m_real (Genesis simulé)       = {m_real*1000:.1f} g")
    print(f"  hover thrust nominal = {P.G*m_nom:.3f} N  |  hover thrust réel = {P.G*m_real:.3f} N")

    # quelques waypoints autour du goal -> excite R et T_N (sinon résidu x/y constant)
    g = np.asarray(P.HOVER_GOAL, dtype=np.float64)
    waypoints = [g, g + [0.4, 0.0, 0.0], g + [0.0, 0.4, 0.0],
                 g + [-0.4, 0.0, 0.2], g + [0.0, -0.4, -0.1]]
    log = acquire_log(env, agent, args.steps, waypoints=waypoints)

    X, y = build_residual_dataset(log, window_sec=None)   # MÊME builder que le pipeline réel
    print(f"\n  Dataset acquis : X{X.shape}  y{y.shape}")
    print(f"  biais moyen y      = {y.mean(0).round(3)} m/s²  (le gros terme z = perte de poussée)")
    print(f"  |y| médian         = {np.median(np.linalg.norm(y, axis=1)):.3f} m/s²")
    exact_mean = exact_mass_residual(X, m_nom, m_real).mean(0)
    print(f"  résidu exact moyen = {exact_mean.round(3)} m/s²  (ordre de grandeur attendu)")

    if args.save_dataset:
        out = (TEST_CF / args.save_dataset) if not Path(args.save_dataset).is_absolute() else Path(args.save_dataset)
        np.savez_compressed(out, X=X, y=y, m_nom=m_nom, m_real=m_real)
        print(f"  dataset -> {out}")

    print(f"\n=== 2. Apprentissage du MLP de résiduel ({args.res_epochs} époques) ===")
    res_model = fit_residual(X, y, epochs=args.res_epochs, device="cpu", verbose=True)

    print("\n=== 3. Comparaison : résidu APPRIS (MLP) vs résidu EXACT (formule masse) ===")
    res_model.eval()
    with torch.no_grad():
        model_res = res_model(torch.from_numpy(X)).cpu().numpy()    # (N, 3)
    exact_res = exact_mass_residual(X, m_nom, m_real)               # (N, 3)

    print(f"  moyenne  MLP     = {model_res.mean(0).round(3)}")
    print(f"  moyenne  exact   = {exact_res.mean(0).round(3)}")
    print(f"  moyenne  données = {y.mean(0).round(3)}")
    print()
    for ax, name in enumerate(["x", "y", "z"]):
        rmse = np.sqrt(np.mean((model_res[:, ax] - exact_res[:, ax]) ** 2))
        std = exact_res[:, ax].std()
        corr = np.corrcoef(model_res[:, ax], exact_res[:, ax])[0, 1] if std > 1e-9 else float("nan")
        print(f"  axe {name}: RMSE(MLP vs exact)={rmse:7.4f}  corr={corr:+.3f}  "
              f"(σ_exact={std:.3f})")
    rmse_exact = np.sqrt(np.mean((model_res - exact_res) ** 2))
    rmse_data = np.sqrt(np.mean((model_res - y) ** 2))
    print(f"\n  RMSE global MLP vs exact   = {rmse_exact:.4f} m/s²")
    print(f"  RMSE global MLP vs données = {rmse_data:.4f} m/s²")
    print("  (MLP vs exact petit ET MLP vs données ~ bruit FD  =>  le MLP apprend bien l'effet de masse)")

    # nuage de points MLP vs exact par axe
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, name in enumerate(["x", "y", "z"]):
            a = axes[ax]
            a.scatter(exact_res[:, ax], model_res[:, ax], s=3, alpha=0.15)
            lim = [min(exact_res[:, ax].min(), model_res[:, ax].min()),
                   max(exact_res[:, ax].max(), model_res[:, ax].max())]
            a.plot(lim, lim, "r--", lw=1)
            a.set_xlabel(f"exact {name} (m/s²)"); a.set_ylabel(f"MLP {name} (m/s²)")
            a.set_title(f"axe {name}")
        fig.suptitle(f"Résidu appris (MLP) vs exact — Genesis "
                     f"(m_nom={m_nom*1000:.0f} g, m_real={m_real*1000:.0f} g)")
        plt.tight_layout()
        out = (TEST_CF / args.plot) if not Path(args.plot).is_absolute() else Path(args.plot)
        fig.savefig(out, dpi=120)
        print(f"\n  nuage MLP vs exact -> {out}")
    except Exception as e:
        print(f"  [plot] ignoré ({e})")


if __name__ == "__main__":
    main()
