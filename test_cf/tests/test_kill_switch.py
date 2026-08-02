"""Tests du KILL SWITCH de core/lotf_real_eval.py — SANS drone, SANS radio.

On exécute la VRAIE fonction `main()` (donc la vraie boucle + les vrais handlers
clavier `on_press`/`on_release`) en mockant uniquement la frontière matérielle :
    * `SyncCrazyflie` / `Crazyflie`  → faux context-manager + faux `cf.commander`
      (enregistre les `send_stop_setpoint` / `send_hover_setpoint`).
    * `LotfRealEnv`                  → FakeEnv qui enregistre takeoff/hover_idle/land/
      kill/step (aucune physique, juste la trace des appels).
    * `keyboard.Listener`            → capture les handlers ; le test « tape » les
      touches en appelant directement on_press/on_release depuis un thread driver.
    * `RLPolicy`, `cflib.crtp.init_drivers` → stubs.

On vérifie les 3 comportements critiques avant un vrai vol :
    A. ENTRÉE relâchée  = KILL  → la boucle sort et `send_stop_setpoint()` est appelé
       (et `env.kill()` n'est PAS utilisé : kill via flag + finally, comme l'ancien run).
    B. ESPACE pressé puis relâché = ATTERRISSAGE (et le land l'emporte sur le fail-safe
       hover — le bug d'ordre `if fail_safe ... elif land` est corrigé).
    C. fin de fenêtre RL (t >= RL_END_S) = auto-land.

Lancer :  cd test_cf && RL-real/.venv/bin/python tests/test_kill_switch.py
"""
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
TEST_CF = THIS.parents[1]
sys.path.insert(0, str(TEST_CF / "core"))     # comme quand on lance core/lotf_real_eval.py

import lotf_real_eval as M                     # noqa: E402


# ───────────────────────────── doubles de test ─────────────────────────────
class FakeCommander:
    def __init__(self, calls):
        self.calls = calls

    def send_stop_setpoint(self):
        self.calls.append(("stop",))
        if FakeCommander.on_stop is not None:    # hook test (ex: vérifier quit déjà posé)
            FakeCommander.on_stop()

    on_stop = None

    def send_hover_setpoint(self, r, p, y, z):
        self.calls.append(("hover", z))

    def send_setpoint_manual(self, *a):
        self.calls.append(("manual",) + a)


class FakeParam:
    def add_update_callback(self, group, name, cb):
        cb(None, "1")                          # « flow deck présent » immédiatement


class FakePlatform:
    def __init__(self, calls):
        self.calls = calls

    def send_arming_request(self, on):
        self.calls.append(("arm", on))


class FakeCf:
    def __init__(self, calls):
        self.commander = FakeCommander(calls)
        self.platform = FakePlatform(calls)
        self.param = FakeParam()
        self.log = types.SimpleNamespace(add_config=lambda *a, **k: None)


class FakeScf:
    def __init__(self, calls):
        self.cf = FakeCf(calls)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeState:
    def __init__(self):
        self.RL = False
        self.start_flight_time = 0.0
        self.latest = {"pos": np.zeros(3), "vel": np.zeros(3),
                       "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
        self.history = {}


class FakeEnv:
    """Enregistre la séquence d'appels de la boucle, sans aucune physique."""
    instances = []

    def __init__(self, scf, goal=None):
        self.scf = scf
        self.dt = 0.02
        self.goal = np.asarray(goal if goal is not None else [0, 0, 1.0], float)
        self.state = FakeState()
        self.episode_length = 0
        self.thrust_history = {"time": [], "thrust_N": []}
        self.events = []                       # trace ordonnée des appels
        FakeEnv.instances.append(self)

    def setup_logs(self):
        self.events.append("setup_logs")

    def get_obs(self):
        return np.zeros(27, np.float32)

    def takeoff(self):
        self.events.append("takeoff")

    def hover_idle(self, z=None):
        self.events.append(("hover_idle", round(float(z), 3) if z is not None else None))
        time.sleep(self.dt)

    def land(self):
        self.events.append("land")

    def emergency_stop(self):
        # reproduit le contrat : coupe (synchrone) + marque l'arrêt du thread MC
        self.events.append("emergency_stop")
        self.scf.cf.commander.send_stop_setpoint()

    def step(self, action):
        self.events.append("step")
        self.episode_length += 1
        time.sleep(self.dt)
        info = {"pos": np.zeros(3), "pos_error": 0.05, "thrust_N": 0.27, "pwm_pct": 42.0}
        return self.get_obs(), info


class FakeAgent:
    def get_action(self, obs):
        return torch.tensor([0.27, 0.0, 0.0, 0.0], dtype=torch.float32)


class FakeListener:
    """Capture on_press/on_release ; signale qu'ils sont prêts."""
    last = None

    def __init__(self, on_press=None, on_release=None):
        self.on_press = on_press
        self.on_release = on_release
        self.ready = threading.Event()
        FakeListener.last = self

    def start(self):
        self.ready.set()

    def stop(self):
        pass


class _Key:                                    # imite keyboard.Key.enter / .space
    def __init__(self, name):
        self.name = name


class _CharKey:
    def __init__(self, char):
        self.char = char


# ───────────────────────────── harnais ─────────────────────────────
def run_scenario(argv, drive):
    """Lance M.main() avec argv mocké ; `drive(listener, calls, env_getter)` tape
    les touches depuis un thread une fois les handlers prêts. Renvoie (calls, env)."""
    calls = []
    FakeEnv.instances.clear()
    FakeListener.last = None

    # monkeypatch de la frontière matérielle
    M.SyncCrazyflie = lambda *a, **k: FakeScf(calls)
    M.Crazyflie = lambda *a, **k: None
    M.LotfRealEnv = FakeEnv
    M.RLPolicy = lambda *a, **k: FakeAgent()
    M.keyboard.Listener = FakeListener
    M.keyboard.Key = types.SimpleNamespace(enter=_Key("enter"), space=_Key("space"))
    M.cflib.crtp.init_drivers = lambda *a, **k: None
    # vol court & déterministe
    M.WARMUP_END_S = 0.1
    M.RL_END_S = 100.0                          # ne se termine pas tout seul (sauf scénario C)
    sys.argv = argv

    driver_done = threading.Event()

    def driver():
        # attendre que le listener (et donc les handlers) soient prêts
        for _ in range(500):
            if FakeListener.last is not None and FakeListener.last.ready.is_set():
                break
            time.sleep(0.01)
        time.sleep(1.3)                         # laisse passer l'arming sleep(1.0)+ setup
        drive(FakeListener.last, calls, lambda: FakeEnv.instances[-1])
        driver_done.set()

    th = threading.Thread(target=driver, daemon=True)
    th.start()
    M.main()                                    # bloque jusqu'à quit/land/RL_END
    driver_done.wait(timeout=5)
    return calls, (FakeEnv.instances[-1] if FakeEnv.instances else None)


def _check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return cond


# ───────────────────────────── scénarios ─────────────────────────────
def scenario_A_kill():
    print("Scénario A : ENTRÉE relâchée = KILL INSTANTANÉ depuis le callback clavier")
    probe = {}

    def drive(lst, calls, env_of):
        lst.on_press(M.keyboard.Key.enter)      # vole
        time.sleep(0.6)                          # décolle + quelques pas RL
        stops_avant = sum(c[0] == "stop" for c in calls)
        lst.on_release(M.keyboard.Key.enter)     # KILL (doit couper AVANT de rendre la main)
        # à ce point, on_release a rendu la main : la coupe instantanée a déjà eu lieu
        probe["instant_stop"] = sum(c[0] == "stop" for c in calls) > stops_avant

    calls, env = run_scenario(["lotf_real_eval.py", "--session", "t"], drive)
    ok = True
    ok &= _check("coupe INSTANTANÉE : send_stop déjà émis quand on_release rend la main",
                 probe.get("instant_stop", False))
    ok &= _check("send_stop_setpoint appelé (moteurs coupés)",
                 any(c[0] == "stop" for c in calls))
    ok &= _check("emergency_stop() appelé depuis le callback (coupe + arrêt thread MC)",
                 "emergency_stop" in env.events, f"events={env.events[:6]}...")
    ok &= _check("a bien décollé puis exécuté des pas RL",
                 "takeoff" in env.events and "step" in env.events)
    ok &= _check("PAS d'atterrissage (kill ≠ land)", "land" not in env.events)
    return ok


def scenario_B_space_land():
    print("Scénario B : ESPACE pressé→relâché = ATTERRISSAGE (land prioritaire sur fail-safe)")

    def drive(lst, calls, env_of):
        lst.on_press(M.keyboard.Key.enter)
        time.sleep(0.4)
        lst.on_press(M.keyboard.Key.space)       # fail-safe : hover 0.3
        time.sleep(0.2)
        lst.on_release(M.keyboard.Key.space)     # land
        time.sleep(0.3)

    calls, env = run_scenario(["lotf_real_eval.py", "--session", "t"], drive)
    ok = True
    ok &= _check("fail-safe hover 0.3 m émis pendant ESPACE pressé",
                 ("hover_idle", 0.3) in env.events)
    ok &= _check("env.land() appelé au relâché d'ESPACE", "land" in env.events)
    # land prioritaire : aucun hover_idle(0.3) APRÈS le 1er land
    li = env.events.index("land")
    no_hover_after = all(e != ("hover_idle", 0.3) for e in env.events[li + 1:])
    ok &= _check("land prioritaire : plus de hover 0.3 après le land (bug corrigé)",
                 no_hover_after, f"events={env.events}")
    return ok


def scenario_C_autoland():
    print("Scénario C : t >= RL_END_S = AUTO-LAND")

    def drive(lst, calls, env_of):
        lst.on_press(M.keyboard.Key.enter)
        time.sleep(1.2)                          # dépasse RL_END_S (patché à 0.8)

    # patch local de RL_END_S pour ce scénario
    calls = []
    FakeEnv.instances.clear(); FakeListener.last = None
    M.SyncCrazyflie = lambda *a, **k: FakeScf(calls)
    M.Crazyflie = lambda *a, **k: None
    M.LotfRealEnv = FakeEnv
    M.RLPolicy = lambda *a, **k: FakeAgent()
    M.keyboard.Listener = FakeListener
    M.keyboard.Key = types.SimpleNamespace(enter=_Key("enter"), space=_Key("space"))
    M.cflib.crtp.init_drivers = lambda *a, **k: None
    M.WARMUP_END_S = 0.1
    M.RL_END_S = 0.8
    sys.argv = ["lotf_real_eval.py", "--session", "t"]

    done = threading.Event()

    def driver():
        for _ in range(500):
            if FakeListener.last is not None and FakeListener.last.ready.is_set():
                break
            time.sleep(0.01)
        time.sleep(0.2)
        drive(FakeListener.last, calls, lambda: FakeEnv.instances[-1])
        done.set()

    threading.Thread(target=driver, daemon=True).start()
    M.main()
    done.wait(timeout=5)
    env = FakeEnv.instances[-1]
    ok = _check("env.land() appelé en fin de fenêtre RL", "land" in env.events)
    ok &= _check("send_stop_setpoint appelé en sortie", any(c[0] == "stop" for c in calls))
    return ok


def scenario_D_emergency_stop_real():
    """Test DIRECT du vrai LotfRealEnv.emergency_stop : le thread de setpoint du
    MotionCommander (qui spamme du hover comme cflib) DOIT être tué -> aucun hover
    renvoyé après le kill (sinon les moteurs se rallument, le bug réel)."""
    print("Scénario D : LotfRealEnv.emergency_stop tue le thread MC (pas de rallumage)")
    import lotf_real_env as RENV

    rec = []                                       # (type, wall_time)

    class RecCommander:
        def send_stop_setpoint(self): rec.append(("stop", time.time()))
        def send_hover_setpoint(self, *a): rec.append(("hover", time.time()))
        def send_setpoint_manual(self, *a): rec.append(("manual", time.time()))

    scf = types.SimpleNamespace(cf=types.SimpleNamespace(commander=RecCommander()))
    env = RENV.LotfRealEnv(scf, goal=[0, 0, 1.0])

    # Faux _SetPointThread : renvoie du hover en boucle (comme cflib) jusqu'au stop().
    class FakeSPThread:
        def __init__(self, cmd):
            self._cmd = cmd
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._run, daemon=True)

        def start(self): self._t.start()

        def _run(self):
            while not self._stop.is_set():
                self._cmd.send_hover_setpoint(0, 0, 0, 1.0)
                time.sleep(0.02)                   # ~50 Hz, plus agressif que cflib

        def stop(self):                            # même API que cflib._SetPointThread
            self._stop.set(); self._t.join()

    spt = FakeSPThread(scf.cf.commander); spt.start()
    env.mc._thread = spt; env.mc._is_flying = True

    time.sleep(0.1)                                # le thread « vole » (envoie des hovers)
    hovers_before = sum(r[0] == "hover" for r in rec)

    env.emergency_stop()
    t_kill = time.time()
    time.sleep(0.25)                               # > 1 période MC : un thread vivant rallumerait

    hovers_after = [r for r in rec if r[0] == "hover" and r[1] > t_kill + 0.005]
    ok = True
    ok &= _check("le thread MC volait bien avant le kill", hovers_before > 0,
                 f"{hovers_before} hovers")
    ok &= _check("AUCUN hover renvoyé après le kill (pas de rallumage moteurs)",
                 len(hovers_after) == 0, f"{len(hovers_after)} hovers fantômes")
    ok &= _check("send_stop_setpoint émis au kill", any(r[0] == "stop" for r in rec))
    ok &= _check("thread MC bien marqué arrêté", env.mc._thread is None and not env.mc._is_flying)
    return ok


if __name__ == "__main__":
    results = []
    for fn in (scenario_A_kill, scenario_B_space_land, scenario_C_autoland,
               scenario_D_emergency_stop_real):
        try:
            results.append(fn())
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append(False)
        print()
    print("=" * 60)
    print("RÉSULTAT :", "TOUS OK ✅" if all(results) else "ÉCHEC ❌")
    sys.exit(0 if all(results) else 1)
