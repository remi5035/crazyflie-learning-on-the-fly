"""Cœur « learning-on-the-fly » — agnostique du backend (Genesis OU vrai drone).

Ce module isole le CERVEAU de l'adaptation en ligne, sans aucune dépendance à
Genesis ni au Crazyflie : il opère sur un simple buffer de rollout
`{t, p, R, v, T_N, omega}` (rempli par la boucle de vol, quel que soit le backend)
et un `agent_ref` mutable (`[RLPolicy]`) pour le hot-swap.

Briques fournies :
    * `JaxFinetuneWorker` — worker JAX persistant (`finetune_lotf_worker.py`), gardé
      chaud entre deux finetunes (supprime le cold-start ~25 s d'import + JIT).
    * `_start_finetune` — adaptation INCRÉMENTALE en vol (résidu → BPTT court →
      hot-swap, répété sur fenêtres rafraîchies) + GARDE-FOU anti-divergence.
    * `_wait_sim_seconds`, `_recent_hover_err`, `_stdin_listener` — utilitaires.

`lotf_genesis_eval.py` (démo Genesis) et `lotf_real_eval.py` (vrai Crazyflie)
importent tous deux ce module : un seul cerveau, deux frontières interchangeables.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent              # core/
REPO = next(p for p in THIS_DIR.parents if (p / "lotf").is_dir())   # racine du dépôt
TEST_CF = REPO / "test_cf"
REAL_DIR = TEST_CF / "RL-real"
if str(REAL_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DIR))

import cf_params as P                              # noqa: E402  (single source of truth)
from lotf_config import CFG                         # noqa: E402  (source unique des hyperparams)
from RL_policy import RLPolicy                      # noqa: E402  (Actor LOTF + biais hover)

# Réglages de l'online-finetune (lotf_config.yaml, section `online`). Le worker JAX
# reçoit ces valeurs ; voir la note ⚠ du YAML sur l'écart vs article/Gazebo (30/step
# incrémental). Le fit tourne en fond pendant que le vol continue avec la policy actuelle.
ONLINE_FT_WINDOW_SEC = CFG["online"]["window_sec"]
ONLINE_FT_BPTT_EPOCHS = CFG["online"]["bptt_epochs"]


class JaxFinetuneWorker:
    """Worker JAX persistant (`finetune_lotf_worker.py`) gardé chaud entre deux
    finetunes — supprime le cold-start (~25 s d'import JAX + JIT par appel).

    On lance le processus UNE fois ; le warmup (build env + compile) se fait en
    fond pendant le vol. Chaque finetune n'est plus qu'un échange stdin/stdout :
    quelques secondes au lieu de ~25 s. Toujours un processus séparé (venv
    jax+lotf), comme le nœud JAX de la pipeline Gazebo."""

    def __init__(self, jax_python, target, epochs, window_sec):
        cmd = [jax_python, str(THIS_DIR / "finetune_lotf_worker.py"),
               "--target", *map(str, target),
               "--epochs", str(epochs), "--window-sec", str(window_sec)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._lock = threading.Lock()          # une finetune à la fois
        self.ready = False
        threading.Thread(target=self._await_ready, daemon=True).start()

    def _readline(self):
        line = self.proc.stdout.readline()
        return line.rstrip("\n") if line else None

    def _await_ready(self):
        """Consomme la sortie du warmup jusqu'à @READY (en fond)."""
        while True:
            line = self._readline()
            if line is None:
                print("[online-ft] worker JAX terminé avant @READY (warmup échoué ?)")
                return
            if line == "@READY":
                self.ready = True
                print("[online-ft] worker JAX chaud — finetunes désormais rapides")
                return
            print(f"[worker] {line}")

    def finetune(self, base, log, out):
        """Envoie une requête et bloque jusqu'au @RESULT. Renvoie le dict résultat
        (ou {'ok': False, ...}). Sérialisé : un seul finetune à la fois."""
        with self._lock:
            # Attendre la fin du warmup AVANT de lire stdout : sinon ce reader et
            # le thread `_await_ready` liraient le pipe en même temps et se voleraient
            # des lignes (corruption du protocole). Une fois `ready` vu, _await_ready
            # a rendu la main -> un seul lecteur.
            while not self.ready:
                if self.proc.poll() is not None:
                    return {"ok": False, "error": "worker JAX arrêté avant @READY"}
                time.sleep(0.05)
            if self.proc.poll() is not None:
                return {"ok": False, "error": "worker JAX arrêté"}
            req = json.dumps({"base": str(base), "log": str(log), "out": str(out)})
            self.proc.stdin.write(req + "\n")
            self.proc.stdin.flush()
            while True:
                line = self._readline()
                if line is None:
                    return {"ok": False, "error": "worker JAX fermé pendant la finetune"}
                if line.startswith("@RESULT "):
                    return json.loads(line[len("@RESULT "):])
                print(f"[worker] {line}")

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _stdin_listener(flags):
    """Thread de fond : 'f'+Entrée → finetune, 'q'+Entrée → quitter.

    `readline()` (et non `for line in sys.stdin`) pour éviter le read-ahead qui
    retarde la livraison ligne-à-ligne quand stdin est un pipe (test scripté)."""
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        c = line.strip().lower()
        if c == "f":
            flags["request"] = True
        elif c == "m":
            flags["mass_toggle"] = True
        elif c == "q":
            flags["quit"] = True
            break


def _wait_sim_seconds(buffer, flags, sim_sec, wall_timeout):
    """Bloque (dans le thread de finetune) jusqu'à ce que le vol ait produit
    `sim_sec` de données FRAÎCHES dans `buffer`, ou jusqu'au timeout mur.

    On attend du temps mesuré par `buffer['t'][-1]` (temps SIM côté Genesis,
    temps mur côté vrai drone), pas du temps mur direct : la fenêtre du prochain
    fit résiduel est ainsi entièrement remplie par des données volées sous la
    nouvelle policy, quelle que soit la vitesse du backend."""
    t = buffer["t"]
    t0 = t[-1] if t else 0.0
    wall0 = time.time()
    while not flags["quit"]:
        if (t and t[-1] - t0 >= sim_sec) or (time.time() - wall0 >= wall_timeout):
            return
        time.sleep(0.02)


def _recent_hover_err(buffer, goal, n):
    """Erreur de hover moyenne (|pos − goal|) sur les `n` derniers pas du buffer."""
    p = buffer["p"]
    if not p:
        return 0.0
    arr = np.asarray(p[-n:])
    return float(np.mean(np.linalg.norm(arr - np.asarray(goal), axis=1)))


def _start_finetune(agent_ref, buffer, flags, cfg):
    """Adaptation INCRÉMENTALE en vol (papier Sec. III-B / pipeline Gazebo).

    Un déclenchement lance `adapt_steps` petits steps. Chaque step : (1) re-fit le
    résidu sur la fenêtre COURANTE du buffer, (2) BPTT court à partir de la
    DERNIÈRE policy (pas de la base), (3) hot-swap, (4) laisse le vol réaccumuler
    des données fraîches sous la nouvelle policy avant le step suivant. Refaire de
    petits pas sur des données rafraîchies évite le sur-entraînement contre un
    résidu seulement localement valide (cause d'oscillation), contrairement à un
    seul gros tir. Le BPTT réutilise le worker JAX persistant (chaud), donc chaque
    step ne coûte que quelques secondes ; le vol n'est jamais interrompu.
    """
    flags["finetuning"] = True
    flags["request"] = False
    ft_worker = cfg["worker"]
    adapt_steps = cfg["adapt_steps"]
    refill_sec = cfg["refill_sec"]
    goal = cfg["goal"]
    diverge_abs = cfg["diverge_err_m"]
    diverge_factor = cfg["diverge_factor"]
    on_hotswap = cfg.get("on_hotswap")    # callback optionnel : on_hotswap(wall_time) à chaque swap
    n_win = max(1, int(ONLINE_FT_WINDOW_SEC / P.DT))
    warm = "chaud" if ft_worker.ready else "en cours de warmup"
    print(f"[online-ft] start — {adapt_steps} steps incrémentaux (worker JAX {warm}), "
          f"fenêtre {ONLINE_FT_WINDOW_SEC:.0f}s, BPTT {ONLINE_FT_BPTT_EPOCHS} ep/step ; "
          f"vol maintenu avec la policy courante")

    def worker():
        try:
            tmp = TEST_CF / "measurements" / "online_ft"
            tmp.mkdir(exist_ok=True)
            base_pt = tmp / "base.pt"
            log_npz = tmp / "log.npz"
            out_pt = tmp / "ft.pt"
            for step in range(1, adapt_steps + 1):
                if flags["quit"]:
                    break
                # snapshot de la policy COURANTE (= hot-swap du step précédent -> incrémental)
                # et de la fenêtre de log FRAÎCHE (re-remplie sous la nouvelle policy).
                prev_policy = agent_ref[0]
                pre_err = _recent_hover_err(buffer, goal, n_win)
                base_state = {k: v.detach().clone()
                              for k, v in prev_policy.policy.state_dict().items()}
                log_snapshot = {k: np.asarray(list(v)) for k, v in buffer.items()}
                torch.save({"model_state_dict": base_state}, base_pt)
                np.savez_compressed(log_npz, **log_snapshot)

                res = ft_worker.finetune(base_pt, log_npz, out_pt)
                if not res.get("ok"):
                    raise RuntimeError(res.get("error", "échec"))
                with torch.device("cpu"):
                    agent_ref[0] = RLPolicy(str(out_pt))   # hot-swap atomique
                if on_hotswap is not None:
                    on_hotswap(time.time())                # instant du swap (pour marquer les courbes)
                print(f"[online-ft] step {step}/{adapt_steps} en {res.get('sec', '?')}s "
                      f"(résidu sur {res.get('n', '?')} samples, |y|_med={res.get('ymed', '?')}) "
                      f"→ hot-swap (err avant={pre_err:.3f} m)")

                # voler la nouvelle policy pour (a) évaluer sa stabilité, (b) réaccumuler
                # des données fraîches pour le prochain résidu.
                _wait_sim_seconds(buffer, flags, refill_sec, wall_timeout=3 * refill_sec + 5)
                post_err = _recent_hover_err(buffer, goal, n_win)

                # GARDE-FOU anti-divergence : si la nouvelle policy dégrade nettement le
                # hover (sur-adaptation contre un résidu structurel), on REVIENT à la
                # policy précédente et on arrête. Indispensable avant un vrai drone.
                if post_err > max(diverge_abs, diverge_factor * pre_err):
                    agent_ref[0] = prev_policy
                    print(f"[online-ft] ⚠ step {step} DIVERGE (err {pre_err:.3f}→{post_err:.3f} m) "
                          f"→ rollback vers la policy précédente, arrêt de l'adaptation")
                    break
                print(f"[online-ft] step {step} OK (err après={post_err:.3f} m)")
            else:
                print(f"[online-ft] adaptation incrémentale terminée ({adapt_steps} steps)")
        except Exception as e:
            print(f"[online-ft] abandonné, policy = dernier hot-swap réussi : {e}")
        finally:
            flags["finetuning"] = False

    threading.Thread(target=worker, daemon=True).start()
