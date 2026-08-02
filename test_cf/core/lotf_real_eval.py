"""Évalue/adapte une politique LOTF sur le VRAI Crazyflie (learning-on-the-fly).

Point d'entrée du drone réel : il branche le MÊME cœur que la démo Genesis
(`lotf_online` : worker JAX persistant + adaptation incrémentale + garde-fou
anti-divergence) sur la frontière radio cflib (`lotf_real_env.LotfRealEnv`) au lieu
de Genesis. La politique, l'obs 27-dim et l'action SI sont inchangées.

Deux venvs (comme la démo Genesis, mais SANS genesis) :
    * `RL-real/.venv`  (cflib + torch) → lance CE script (contrôle 50 Hz + radio).
    * `../.venv`       (jax + lotf)    → worker de finetune, lancé en SOUS-PROCESSUS.

Exemple :
    cd test_cf
    RL-real/.venv/bin/python core/lotf_real_eval.py \
        --model models/model_pretrain.pt --online-finetune \
        --jax-python ../.venv/bin/python --session real_run

Commandes clavier (pynput) — kill switch REPRIS À L'IDENTIQUE de l'ancienne
pipeline éprouvée (`RL-real/test_main.py` + `drone_controller.run`) :
    ENTRÉE      → APPUYER = décolle/vole ; RELÂCHER = KILL SWITCH (sort de la boucle
                  → `send_stop_setpoint()` final coupe les moteurs, le drone chute)
    ESPACE      → APPUYER = fail-safe (hover bas stabilisé à 0.3 m) ;
                  RELÂCHER = atterrissage contrôlé puis fin
    f           → déclenche l'adaptation en ligne (résidu + BPTT + hot-swap)

Le kill pose le drapeau `quit` PUIS appelle `env.emergency_stop()` depuis le callback
clavier (latence ~ms). CRUCIAL : `emergency_stop` coupe les moteurs ET arrête le thread
de setpoint du MotionCommander — sans ça, ce thread cflib renvoie un hover toutes les
~200 ms et RE-STABILISE le vol après le lâché (bug observé : moteurs rallumés quelques
secondes). PAS de `disarm`/`xset`/dead-man backup. Le `finally` ré-appelle stop en filet.

Enveloppe de vol reprise de `RL-real/drone_controller.run` (l'interface qui
marchait) : décollage → warmup hover (t<5 s) → fenêtre RL t∈[5,35] s → auto-land.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from threading import Event

import numpy as np
import torch

from pynput import keyboard
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

THIS_DIR = Path(__file__).resolve().parent              # core/
REPO = next(p for p in THIS_DIR.parents if (p / "lotf").is_dir())   # racine du dépôt
TEST_CF = REPO / "test_cf"
REAL_DIR = TEST_CF / "RL-real"
if str(REAL_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DIR))

import cf_params as P                              # noqa: E402
from lotf_config import CFG                         # noqa: E402  (source unique des hyperparams)
from RL_policy import RLPolicy                      # noqa: E402  (Actor LOTF + biais hover)
from drone_state import _quat_wxyz_to_R            # noqa: E402  (même conversion que le vrai log)
from utils import plot_history, save_history_to_csv, plot_thrust   # noqa: E402
from lotf_real_env import LotfRealEnv               # noqa: E402  (frontière radio cflib)
# Cerveau « learning-on-the-fly » agnostique du backend (partagé avec lotf_genesis_eval).
from lotf_online import (JaxFinetuneWorker, _start_finetune,        # noqa: E402
                         ONLINE_FT_WINDOW_SEC, ONLINE_FT_BPTT_EPOCHS)

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

# Enveloppe de vol (s, depuis le décollage) — cf drone_controller.run.
WARMUP_END_S = 5.0          # hover firmware avant de rendre la main à la policy
RL_END_S = 40.0             # auto-land : fin de la fenêtre RL
# Déclenchement AUTOMATIQUE du finetune à un instant FIXE de vol RL (reproductibilité
# entre runs : pour tracer une courbe moyennée sur 5 essais, tous déclenchent au même t).
AUTO_FT_RL_SEC = 15.0       # s après le début de la fenêtre RL (= t - WARMUP_END_S)

def _save_log_lotf(log, out):
    if not log['t']:
        print('[lotf] log vide, rien à sauver.'); return
    np.savez_compressed(
        out,
        t=np.asarray(log['t']), p=np.asarray(log['p']), R=np.asarray(log['R']),
        v=np.asarray(log['v']), T_N=np.asarray(log['T_N']), omega=np.asarray(log['omega']))
    print(f"[log] rollout SI -> {out}  ({len(log['t'])} lignes, format finetune_lotf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/model_pretrain.pt",
                    help="Checkpoint .pt de la politique LOTF (27->512->512->4).")
    ap.add_argument("--session", type=str, default="measurements/real/real_run",
                    help="Préfixe des sorties (logs/plots/.npz).")
    ap.add_argument("--online-finetune", action="store_true",
                    help="Learning-on-the-fly : 'f' en vol = fit résiduel + BPTT + "
                         "hot-swap (worker JAX persistant, comme la démo Genesis).")
    ap.add_argument("--goal", type=float, nargs=3, default=None,
                    help="Goal [x y z] (défaut = cf_params.HOVER_GOAL).")
    ap.add_argument("--jax-python", type=str,
                    default=str(REPO / ".venv" / "bin" / "python"),
                    help="Python d'un venv jax+lotf+torch pour le finetune en ligne "
                         "(sous-processus, comme le nœud JAX Gazebo).")
    args = ap.parse_args()

    cflib.crtp.init_drivers()
    logging.basicConfig(level=logging.ERROR)

    goal = np.asarray(args.goal, dtype=np.float64) if args.goal is not None \
        else np.asarray(P.HOVER_GOAL, dtype=np.float64)
    model_path = (TEST_CF / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    with torch.device("cpu"):
        agent = RLPolicy(str(model_path))
    agent_ref = [agent]                          # conteneur mutable → hot-swap

    # buffer SI (format finetune_lotf) : alimente --online-finetune ET la sauvegarde.
    log = {'t': [], 'p': [], 'R': [], 'v': [], 'T_N': [], 'omega': []}
    history_cmd = {'time': [], 'x_pos': [], 'y_pos': [], 'z_pos': []}
    # Marqueurs (wall-clock) pour retracer les courbes : déclenchement du finetune,
    # hot-swap réussi, et temps de calcul (hot-swap − déclenchement) de chaque step.
    finetune_trigger_times = []
    hotswap_times = []
    compute_times = []

    def _on_hotswap(t_hs):
        """Appelé à chaque hot-swap réussi : enregistre l'instant + le temps de calcul
        (depuis le dernier déclenchement de finetune)."""
        hotswap_times.append(t_hs)
        compute_times.append(
            t_hs - finetune_trigger_times[-1] if finetune_trigger_times else float('nan'))

    # Drapeaux partagés avec le listener clavier ET le cerveau online (lotf_online,
    # qui lit/écrit `quit`, `request`, `finetuning`). Kill switch IDENTIQUE à l'ancienne
    # pipeline éprouvée : `is_flying` (ENTRÉE pressée), `quit` (ENTRÉE relâchée = kill),
    # `fail_safe`/`land_fail_safe` (ESPACE pressé/relâché).
    flags = {"is_flying": False, "fail_safe": False, "land_fail_safe": False,
             "quit": False, "request": False, "finetuning": False}
    env_holder = [None]   # rempli avec l'env -> coupe d'urgence instantanée au relâché

    deck_attached = Event()

    def _on_deck(_, value_str):
        if int(value_str):
            deck_attached.set()

    # ── Listener clavier (kill éprouvé de RL-real/test_main.py + coupe instantanée) ──
    # ENTRÉE pressée = vole ; ESPACE pressé = fail-safe. Le relâché pose un drapeau et
    # ARRÊTE le listener (return False).
    def on_press(key):
        if key == keyboard.Key.enter:
            flags["is_flying"] = True
        if key == keyboard.Key.space:
            flags["fail_safe"] = True
        if getattr(key, "char", None) == "f":
            if not args.online_finetune:
                print("[online-ft] désactivé — relancer avec --online-finetune")
            elif flags["finetuning"]:
                print("[online-ft] déjà en cours, ignoré")
            else:
                flags["request"] = True

    def on_release(key):
        # ENTRÉE relâchée = KILL INSTANTANÉ : on pose `quit` (la boucle ne renverra plus
        # aucun setpoint -> pas de course) PUIS on coupe DIRECTEMENT ici, dans le thread
        # clavier (latence ~ms). `emergency_stop` coupe ET arrête le thread de setpoint du
        # MotionCommander — sinon ce thread re-stabilise le vol 200 ms plus tard (bug
        # observé : moteurs rallumés après le lâché). Le `finally` ré-appelle stop en filet.
        if key == keyboard.Key.enter:
            flags["quit"] = True
            env = env_holder[0]
            if env is not None:
                env.emergency_stop()
            print("\n[KILL] ENTRÉE relâchée — moteurs coupés (instantané, sans rallumage).")
            return False
        # ESPACE relâché = atterrissage contrôlé.
        if key == keyboard.Key.space:
            flags["land_fail_safe"] = True
            return False

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache=str(REAL_DIR / 'cache'))) as scf:
        env = LotfRealEnv(scf, goal=goal)
        env_holder[0] = env                       # active la coupe d'urgence du clavier

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        scf.cf.param.add_update_callback(group='deck', name='bcFlow2', cb=_on_deck)
        if not deck_attached.wait(timeout=5):
            print('Aucun flow deck détecté !'); listener.stop(); sys.exit(1)

        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)
        env.setup_logs()

        ft_worker = None
        if args.online_finetune:
            # Worker JAX persistant lancé TOUT DE SUITE : son warmup (~25 s) tourne en
            # fond pendant le décollage/warmup, donc le 1er 'f' est déjà rapide.
            ft_worker = JaxFinetuneWorker(args.jax_python, goal,
                                          ONLINE_FT_BPTT_EPOCHS, ONLINE_FT_WINDOW_SEC)
            print("\n[online-ft] ACTIF (worker JAX en warmup). En vol : 'f' = "
                  "finetune+hot-swap.\n")

        print("\nPRÊT. Maintenez ENTRÉE pour voler (relâcher = KILL), ESPACE = fail-safe.")
        take_off = False
        start_time = 0.0
        pos_errs = []
        # Excitation de poussée avant le fit (cf lotf_config.online.excite_*) : étale T
        # pour contraindre ∂a/∂T du résidu (un hover statique le laisse libre).
        _o = CFG["online"]
        excite_steps = int(_o.get("excite_sec", 0.0) / P.DT)
        excite_amp = float(_o.get("excite_amp_N", 0.0))
        excite_freq = float(_o.get("excite_freq_hz", 1.5))
        excite_end = -1                                  # >=0 : excitation en cours jusqu'à ce step
        auto_ft_triggered = False                         # finetune auto déclenché une seule fois

        try:
            while not flags["quit"]:
                # ENTRÉE relâchée → flags["quit"] : la condition du while sort, le
                # `send_stop_setpoint()` du finally coupe les moteurs (kill éprouvé).
                if not flags["is_flying"]:
                    time.sleep(env.dt); continue

                if not take_off:
                    env.takeoff()
                    start_time = time.time()
                    take_off = True

                t = time.time() - start_time

                # ESPACE : pressé = fail-safe (hover bas 0.3 m), relâché = atterrissage.
                # On teste `land_fail_safe` AVANT `fail_safe` (corrige le bug de
                # `drone_controller.run` où `if fail_safe` masquait le land, ESPACE
                # restant pressé+relâché) : l'atterrissage l'emporte donc bien au relâché.
                if flags["land_fail_safe"]:
                    env.state.RL = False
                    env.land(); break

                elif flags["fail_safe"]:
                    env.hover_idle(0.3)

                elif t >= RL_END_S:
                    env.state.RL = False
                    env.land(); break

                elif t >= WARMUP_END_S:
                    # ───── fenêtre RL active ─────
                    env.state.RL = True
                    obs = env.get_obs()
                    action = agent_ref[0].get_action(obs)        # (4,) SI
                    if excite_end >= 0 and env.episode_length < excite_end:   # dither poussée
                        action = action.clone()
                        action[0] = action[0] + excite_amp * float(
                            np.sin(2 * np.pi * excite_freq * env.episode_length * P.DT))
                    obs, info = env.step(action)

                    pos_errs.append(info["pos_error"])
                    # ligne de log SI (même contenu que drone_controller._log_lotf_row)
                    a = action.detach().cpu().numpy()
                    lat = env.state.latest
                    log['t'].append(time.time())
                    log['p'].append(lat['pos'].copy())
                    log['R'].append(_quat_wxyz_to_R(lat['quat_wxyz']).flatten())
                    log['v'].append(lat['vel'].copy())
                    log['T_N'].append(float(a[0]))
                    log['omega'].append(a[1:].copy())

                    history_cmd['time'].append(t - WARMUP_END_S)
                    history_cmd['x_pos'].append(float(env.goal[0]))
                    history_cmd['y_pos'].append(float(env.goal[1]))
                    history_cmd['z_pos'].append(float(env.goal[2]))
                    env.thrust_history['time'].append(t)

                    # déclenchement AUTOMATIQUE à un instant FIXE de vol RL (toujours le
                    # même entre runs → courbes superposables sur 5 essais).
                    if (args.online_finetune and not auto_ft_triggered
                            and not flags["finetuning"]
                            and (t - WARMUP_END_S) >= AUTO_FT_RL_SEC):
                        flags["request"] = True
                        auto_ft_triggered = True
                        print(f"\n[online-ft] déclenchement AUTOMATIQUE à "
                              f"t_RL={t - WARMUP_END_S:.1f}s\n")

                    # déclenchement du finetune : on EXCITE d'abord (dither poussée) pour
                    # varier T, puis on fit sur la fenêtre excitée (un seul à la fois).
                    if args.online_finetune and flags["request"] and not flags["finetuning"]:
                        if excite_steps > 0 and excite_end < 0:
                            excite_end = env.episode_length + excite_steps
                            print(f"\n[excite] dither poussée {_o['excite_sec']:.0f}s "
                                  f"(±{excite_amp:.3f} N @ {excite_freq:.1f} Hz) avant le fit...\n")
                        elif excite_end < 0 or env.episode_length >= excite_end:
                            excite_end = -1
                            finetune_trigger_times.append(time.time())   # instant de déclenchement
                            _start_finetune(agent_ref, log, flags,
                                            {"worker": ft_worker, "goal": env.goal,
                                             "adapt_steps": CFG["online"]["adapt_steps"],
                                             "refill_sec": CFG["online"]["refill_sec"],
                                             "diverge_err_m": CFG["online"]["diverge_err_m"],
                                             "diverge_factor": CFG["online"]["diverge_factor"],
                                             "on_hotswap": _on_hotswap})

                    if env.episode_length % 25 == 0:
                        p = info["pos"]
                        print(f"t={t-WARMUP_END_S:5.1f}s | pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                              f"| err={info['pos_error']:.3f} m | T={info['thrust_N']:.3f} N "
                              f"| PWM={info['pwm_pct']:.0f}%")

                elif t >= 1.0:
                    env.hover_idle(float(env.goal[2]))
                else:
                    env.hover_idle(float(env.goal[2]) + 0.1)
        finally:
            try:
                scf.cf.commander.send_stop_setpoint()
            except Exception:
                pass
            flags["quit"] = True
            if ft_worker is not None:
                ft_worker.close()
            listener.stop()

    # ── résumé + sorties ──
    if pos_errs:
        pe = np.asarray(pos_errs)
        print(f"\n=== résumé ({len(pe)} pas RL, {len(pe) * P.DT:.1f} s) ===")
        print(f"erreur position : moyenne={pe.mean():.3f} m | finale={pe[-1]:.3f} m | "
              f"max={pe.max():.3f} m")

    # Instants de hot-swap convertis dans le référentiel propre à chaque courbe :
    #   plot_history → relatif au début RL (state.start_flight_time)
    #   plot_thrust  → relatif au décollage (start_time)
    hs_hist = [t - env.state.start_flight_time for t in hotswap_times
               if env.state.start_flight_time]
    hs_thrust = [t - start_time for t in hotswap_times if start_time]

    # Évènements pour le CSV, dans le repère de env.state.history (relatif au début RL).
    sft = env.state.start_flight_time or 0.0
    events = [{"type": "finetune_trigger", "time": tt - sft}
              for tt in finetune_trigger_times]
    events += [{"type": "hotswap", "time": tt - sft, "compute_time_s": ct}
               for tt, ct in zip(hotswap_times, compute_times)]
    if compute_times:
        print("[online-ft] temps de calcul (hot-swap − déclenchement) : "
              + ", ".join(f"{c:.2f}s" for c in compute_times))

    session_path = TEST_CF / args.session if not Path(args.session).is_absolute() else Path(args.session)
    session_path.parent.mkdir(parents=True, exist_ok=True)
    _save_log_lotf(log, f"{session_path}_lotf.npz")
    try:
        save_history_to_csv(env.state.history, history_cmd, f"{session_path}.csv",
                            events=events)
        plot_history(env.state.history, history_cmd, f"{session_path}_traj.png",
                     hotswaps=hs_hist)
        plot_thrust(env.thrust_history, f"{session_path}_thrust.png",
                    hotswaps=hs_thrust)
    except Exception as e:
        print(f"[plots] non générés : {e}")


if __name__ == '__main__':
    main()
