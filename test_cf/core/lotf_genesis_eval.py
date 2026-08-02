"""Évalue une politique LOTF (27→512→512→4, action SI) dans Genesis.

Test en simulation Genesis AVANT le vrai Crazyflie, avec la politique inchangée
(même réseau, même obs 27-dim, même action SI). On réutilise le checkpoint
produit par `RL-real/pretrain_lotf.py` (ou `finetune_lotf.py`).

Exemples :
    source ../../genesis_venv/bin/activate
    python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt
    python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt --record
    python core/lotf_genesis_eval.py --ckpt models/model_pretrain.pt --log-lotf measurements/genesis/rollout.npz

`--log-lotf` écrit un .npz au MÊME format que `RL-real/drone_controller.save_lotf_log`
(clés t, p, R, v, T_N, omega), donc consommable tel quel par `RL-real/finetune_lotf.py`
pour ajuster le résiduel sur des rollouts Genesis.

`--online-finetune` reproduit le « learning-on-the-fly » du vrai contrôleur
(`drone_controller`) : pendant le vol, tape `f`+Entrée pour déclencher, sur les
dernières secondes, un fit résiduel + BPTT court (en fond) puis un hot-swap de la
politique — sans interrompre le vol. Tape `q`+Entrée pour quitter. Le calcul tourne
dans un WORKER JAX PERSISTANT (`finetune_lotf_worker.py`, venv jax+lotf), démarré
au lancement et gardé chaud : le warmup (~25 s d'import + JIT) se fait UNE fois en
fond pendant le vol, donc chaque finetune ne coûte plus que quelques secondes —
même architecture « processus séparé » que le nœud JAX de Gazebo, sans le
cold-start. Le temps loggé est le temps SIM (dt=0.02 exact) pour un résiduel propre.
"""
import argparse
import sys
import threading
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
from lotf_config import CFG                         # noqa: E402  (source unique des hyperparams)
from RL_policy import RLPolicy                      # noqa: E402  (Actor LOTF + biais hover)
from drone_state import _quat_wxyz_to_R            # noqa: E402  (même conversion que le vrai log)
# Cerveau « learning-on-the-fly » agnostique du backend (partagé avec lotf_real_eval).
from lotf_online import (JaxFinetuneWorker, _start_finetune, _stdin_listener,   # noqa: E402
                         ONLINE_FT_WINDOW_SEC, ONLINE_FT_BPTT_EPOCHS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/model_pretrain.pt",
                        help="Checkpoint .pt de la politique LOTF (27->512->512->4).")
    parser.add_argument("--steps", type=int, default=750,
                        help="Nombre de pas de politique (50 Hz). 750 = 15 s.")
    parser.add_argument("--record", action="store_true",
                        help="Enregistre une vidéo dans assets/videos/.")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Désactive la fenêtre 3D (utile sans display).")
    parser.add_argument("--log-lotf", type=str, default=None,
                        help="Si fourni, sauvegarde le rollout SI au format _lotf.npz.")
    parser.add_argument("--online-finetune", action="store_true",
                        help="Learning-on-the-fly : tape 'f'+Entrée en vol pour fit "
                             "résiduel + BPTT + hot-swap (comme drone_controller).")
    parser.add_argument("--goal", type=float, nargs=3, default=None,
                        help="Goal [x y z] (défaut = cf_params.HOVER_GOAL).")
    parser.add_argument("--mass", type=float, default=None,
                        help="Masse TOTALE Genesis [kg] posée au démarrage "
                             "(défaut = masse URDF cf2x ~42 g). Ex: --mass 0.027.")
    parser.add_argument("--freeze-inertia", action="store_true",
                        help="Diagnostic : --mass (et bouton 'm') ne changent QUE la masse, "
                             "l'inertie reste celle d'origine. Isole l'effet de l'inertie sur "
                             "la boucle de rate (dérive horizontale d'une politique masse≠).")
    parser.add_argument("--diag-csv", type=str, default=None,
                        help="Diagnostic dérive : dump un CSV par pas (ω COMMANDÉ par la "
                             "policy vs gyro RÉEL, roll/pitch/tilt, xy). Compare un vol 27g et "
                             "33g : ω cmd persistant -> politique ; ω cmd ~0 mais attitude qui "
                             "ne suit pas / sur-oscille -> actionnement (mixer/PID de rate).")
    parser.add_argument("--jax-python", type=str,
                        default=str(REPO / ".venv" / "bin" / "python"),
                        help="Python d'un venv jax+lotf+torch pour le finetune en ligne "
                             "(sous-processus finetune_lotf_jax.py, comme le nœud JAX Gazebo).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Réaffiche les logs INFO de Genesis (FPS, etc.).")
    args = parser.parse_args()

    # Par défaut on coupe les INFO Genesis (la ligne 'Running at X FPS' noie tes prints).
    gs.init(logging_level=None if args.verbose else "warning")

    ckpt_path = (TEST_CF / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    # Genesis force cuda comme device torch par défaut ; la pipeline real tourne sur
    # CPU (obs construite par DroneState en numpy). On crée donc la politique sous un
    # contexte CPU explicite pour garder obs et réseau sur le même device.
    with torch.device("cpu"):
        agent = RLPolicy(str(ckpt_path))
    agent_ref = [agent]   # conteneur mutable → permet le hot-swap depuis le worker

    env = LotfHoverEnv(
        goal=args.goal,
        show_viewer=not args.no_viewer,
        visualize_target=True,
        visualize_camera=args.record,
        freeze_inertia=args.freeze_inertia,
    )

    # buffer SI (format finetune_lotf), nécessaire au --log-lotf ET à l'online-ft
    keep_log = bool(args.log_lotf) or args.online_finetune
    log = {'t': [], 'p': [], 'R': [], 'v': [], 'T_N': [], 'omega': []} if keep_log else None

    if args.mass is not None:
        env.set_total_mass(args.mass)   # masse cible posée avant le 1er print/reset

    flags = {"request": False, "finetuning": False, "quit": False, "mass_toggle": False}
    mass_cycle = list(CFG["genesis_bridge"].get("mass_cycle_kg", [0.020, 0.040]))
    mass_idx = -1                       # premier appui sur 'm' -> mass_cycle[0]
    ft_worker = None
    print(f"[genesis] masse simulée au démarrage : {env.mass * 1000:.0f} g")
    if args.online_finetune:
        # On démarre le worker JAX persistant TOUT DE SUITE : son warmup (~25 s,
        # import + JIT) tourne en fond pendant le vol, donc le 1er 'f' est déjà
        # rapide. Les finetunes suivantes touchent les caches chauds.
        ft_worker = JaxFinetuneWorker(args.jax_python, env.goal,
                                      ONLINE_FT_BPTT_EPOCHS, ONLINE_FT_WINDOW_SEC)
        threading.Thread(target=_stdin_listener, args=(flags,), daemon=True).start()
        print("\n[online-ft] ACTIF (worker JAX en warmup). En vol : 'f'+Entrée = "
              "finetune+hot-swap, 'm'+Entrée = changer la masse "
              f"({'/'.join(f'{m*1000:.0f}g' for m in mass_cycle)}), 'q'+Entrée = quitter.\n")

    obs, _ = env.reset()
    if args.record and env.cam is not None:
        env.cam.start_recording()

    pos_errs = []
    # Diagnostic dérive (--diag-csv) : à chaque pas, ω COMMANDÉ par la policy (action[1:])
    # vs gyro RÉEL Genesis (state.latest['gyro'], deg/s corps) + attitude (roll/pitch/tilt).
    diag = [] if args.diag_csv else None
    # Excitation de poussée avant le fit (cf lotf_config.online.excite_*) : un hover
    # statique a T ~constant -> ∂a/∂T du résidu non contraint. On dither la poussée
    # pendant excite_sec avant chaque 'f' pour étaler T (cf tests/test_residual_real.py).
    _o = CFG["online"]
    excite_steps = int(_o.get("excite_sec", 0.0) / P.DT)
    excite_amp = float(_o.get("excite_amp_N", 0.0))
    excite_freq = float(_o.get("excite_freq_hz", 1.5))
    excite_end = -1                                   # >=0 : excitation en cours jusqu'à ce step
    with torch.no_grad():
        for i in range(args.steps):
            if flags["quit"]:
                break

            action = agent_ref[0].get_action(obs)           # (4,) SI, comme le vrai contrôleur
            if excite_end >= 0 and i < excite_end:           # dither la poussée pendant l'excitation
                action = action.clone()
                action[0] = action[0] + excite_amp * float(np.sin(2 * np.pi * excite_freq * i * P.DT))
            obs, info = env.step(action)

            pos_errs.append(info["pos_error"])
            if log is not None:
                # même contenu que drone_controller._log_lotf_row (depuis state.latest),
                # mais t = temps SIM (dt=0.02 exact) → dérivée d'accélération propre.
                a = action.cpu().numpy()
                lat = env.state.latest
                log['t'].append(i * P.DT)
                log['p'].append(lat['pos'].copy())
                log['R'].append(_quat_wxyz_to_R(lat['quat_wxyz']).flatten())
                log['v'].append(lat['vel'].copy())
                log['T_N'].append(float(a[0]))
                log['omega'].append(a[1:].copy())

            if diag is not None:
                a = action.cpu().numpy()
                lat = env.state.latest
                R = _quat_wxyz_to_R(lat['quat_wxyz'])
                # angles ZYX (rad) depuis R (corps->monde) ; tilt = angle de l'axe z corps / vertical
                roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                pitch = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
                yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                tilt  = np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
                p = lat['pos']; g = env.goal
                diag.append((
                    i * P.DT, float(p[0]), float(p[1]), float(p[2]),
                    float(np.hypot(p[0] - g[0], p[1] - g[1])),               # xy_err (m)
                    roll, pitch, yaw, tilt,                                   # attitude (deg)
                    float(a[1]), float(a[2]), float(a[3]),                    # ω COMMANDÉ (rad/s, corps)
                    float(lat['gyro'][0]), float(lat['gyro'][1]), float(lat['gyro'][2]),  # gyro RÉEL (deg/s)
                    float(a[0]), float(info['pwm_pct']),                      # T_N, PWM
                ))

            # bouton 'm' : change la masse Genesis en vol (20 g -> 40 g -> 20 g -> ...)
            if flags["mass_toggle"]:
                flags["mass_toggle"] = False
                # On AVANCE jusqu'à une masse RÉELLEMENT différente de l'actuelle : sinon, si
                # la masse de départ == mass_cycle[0] (cas --mass 0.027), le 1er appui tombait
                # sur la même valeur -> no-op silencieux (le drone ne dérivait pas, et le
                # finetune apprenait un résidu nul). On saute donc toute entrée ≈ masse courante.
                for _ in range(len(mass_cycle)):
                    mass_idx = (mass_idx + 1) % len(mass_cycle)
                    if abs(mass_cycle[mass_idx] - env.mass) > 1e-4:
                        break
                new_m = env.set_total_mass(mass_cycle[mass_idx])
                print(f"\n[mass] masse Genesis -> {new_m * 1000:.0f} g "
                      f"(la policy va dériver ; tape 'f' pour ré-adapter)\n")

            # déclenchement du finetune : on EXCITE d'abord (dither poussée) pour varier T,
            # puis on lance le fit sur la fenêtre fraîchement excitée (un seul à la fois).
            if args.online_finetune and flags["request"] and not flags["finetuning"]:
                if excite_steps > 0 and excite_end < 0:
                    excite_end = i + excite_steps          # démarre l'excitation, fit différé
                    print(f"\n[excite] dither poussée {_o['excite_sec']:.0f}s "
                          f"(±{excite_amp:.3f} N @ {excite_freq:.1f} Hz) pour varier T avant le fit...\n")
                elif excite_end < 0 or i >= excite_end:
                    excite_end = -1                        # excitation finie -> fit
                    _start_finetune(agent_ref, log, flags,
                                    {"worker": ft_worker, "goal": env.goal,
                                     "adapt_steps": CFG["online"]["adapt_steps"],
                                     "refill_sec": CFG["online"]["refill_sec"],
                                     "diverge_err_m": CFG["online"]["diverge_err_m"],
                                     "diverge_factor": CFG["online"]["diverge_factor"]})

            if i % 25 == 0:
                p = info["pos"]
                print(f"step {i:4d} | pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"| err={info['pos_error']:.3f} m "
                      f"| T={info['thrust_N']:.3f} N "
                      f"| PWM={info['pwm_pct']:.0f}%")

            if args.record and env.cam is not None:
                env.cam.render()

    if args.record and env.cam is not None:
        record_out = TEST_CF / "assets" / "videos" / "lotf_genesis.mp4"
        record_out.parent.mkdir(parents=True, exist_ok=True)
        env.cam.stop_recording(save_to_filename=str(record_out), fps=50)
        print(f"[record] -> {record_out}")

    pos_errs = np.array(pos_errs)
    print(f"\n=== résumé ({len(pos_errs)} pas, {len(pos_errs) * P.DT:.1f} s) ===")
    print(f"erreur position : moyenne={pos_errs.mean():.3f} m | "
          f"finale={pos_errs[-1]:.3f} m | max={pos_errs.max():.3f} m")
    print(f"erreur moy. sur les 2 dernières s : "
          f"{pos_errs[-int(2 / P.DT):].mean():.3f} m")

    if diag is not None and diag:
        import csv
        out = (TEST_CF / args.diag_csv) if not Path(args.diag_csv).is_absolute() else Path(args.diag_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["t", "x", "y", "z", "xy_err",
                "roll_deg", "pitch_deg", "yaw_deg", "tilt_deg",
                "wx_cmd", "wy_cmd", "wz_cmd",            # ω commandé par la policy (rad/s, corps)
                "gx_real", "gy_real", "gz_real",         # gyro réel Genesis (deg/s, corps)
                "T_N", "pwm_pct"]
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols); w.writerows(diag)
        d = np.array(diag, dtype=float)
        tail = d[-int(2 / P.DT):]    # 2 dernières s = régime établi
        # ω commandé en roll/pitch (rad/s) = ce que la policy DEMANDE comme inclinaison ;
        # tilt (deg) = inclinaison réelle. Les deux moyennés en fin de vol caractérisent la dérive.
        wcmd_rp = np.abs(tail[:, 9:11]).mean()
        print(f"\n[diag] -> {out}  ({len(diag)} pas)")
        print(f"[diag] 2 dernières s : tilt={tail[:, 8].mean():.2f}° | "
              f"|ω_cmd roll/pitch|={wcmd_rp:.4f} rad/s | xy_err={tail[:, 4].mean()*100:.1f} cm")
        print("[diag] lecture : |ω_cmd| élevé + tilt suivi -> POLITIQUE (a) ; "
              "|ω_cmd|~0 mais tilt/oscillation -> ACTIONNEMENT mixer/PID (b).")

    if args.log_lotf:
        out = (TEST_CF / args.log_lotf) if not Path(args.log_lotf).is_absolute() else Path(args.log_lotf)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            t=np.asarray(log['t']), p=np.asarray(log['p']), R=np.asarray(log['R']),
            v=np.asarray(log['v']), T_N=np.asarray(log['T_N']), omega=np.asarray(log['omega']),
        )
        print(f"[log] rollout SI -> {out}  ({len(log['t'])} lignes, format finetune_lotf)")

    if ft_worker is not None:
        ft_worker.close()


if __name__ == "__main__":
    main()
