"""Test du KILL SWITCH au niveau CLAVIER, SANS le drone (zéro risque).

But : vérifier les 2 choses qui peuvent casser le dead-man de `lotf_real_eval.py` :
  1. le KILL se déclenche-t-il INSTANTANÉMENT au relâché d'ENTRÉE ? (latence callback)
  2. en MAINTENANT ENTRÉE, l'auto-répétition X11 émet-elle de FAUX relâchés ? (ce qui
     couperait les moteurs en plein vol). `xset r off` doit les supprimer.

Reproduit EXACTEMENT la logique clavier de lotf_real_eval (on_press/on_release +
_set_autorepeat), mais sans cflib/genesis. Aucun moteur, aucune radio.

À lancer dans le venv qui a pynput (celui du vrai drone) :
    cd test_cf
    RL-real/.venv/bin/python tests/test_killswitch_keyboard.py

Protocole affiché à l'écran : MAINTENIR ENTRÉE ~5 s puis relâcher, plusieurs fois.
"""
import shutil
import subprocess
import sys
import time

from pynput import keyboard


def _set_autorepeat(on: bool) -> bool:
    """Active/désactive l'auto-répétition clavier via xset (idem lotf_real_eval)."""
    if not shutil.which("xset"):
        return False
    try:
        subprocess.run(["xset", "r", "on" if on else "off"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def main():
    state = {"down": False, "last_release_t": 0.0,
             "kills": 0, "false_releases": 0, "press": 0}

    def now():
        return time.strftime("%H:%M:%S") + f".{int((time.time()%1)*1000):03d}"

    def on_press(key):
        if key == keyboard.Key.enter:
            # un press <120 ms après un release = signature d'AUTO-RÉPÉTITION (faux release)
            if state["down"] is False and state["last_release_t"] \
                    and time.time() - state["last_release_t"] < 0.12:
                state["false_releases"] += 1
                print(f"  [{now()}] ⚠ re-press {1000*(time.time()-state['last_release_t']):.0f} ms "
                      f"après release -> AUTO-RÉPÉTITION (faux release détecté)")
            state["down"] = True
            state["press"] += 1
        if getattr(key, "char", None) == "q":
            return False  # stoppe le listener

    def on_release(key):
        if key == keyboard.Key.enter:
            state["down"] = False
            state["last_release_t"] = time.time()
            state["kills"] += 1
            print(f"  [{now()}] ENTRÉE relâchée -> [KILL] (moteurs coupés sur le vrai drone)")

    autorep = _set_autorepeat(False)
    print("=" * 70)
    print("TEST KILL SWITCH CLAVIER (sans drone)")
    print(f"  auto-répétition désactivée (xset r off) : {'OUI' if autorep else 'NON (xset absent ?)'}")
    print("  -> MAINTIENS ENTRÉE ~5 s puis RELÂCHE. Répète 3-4 fois.")
    print("  -> Chaque relâché doit afficher UN seul [KILL].")
    print("  -> Si tu vois des ⚠ AUTO-RÉPÉTITION PENDANT que tu tiens : DANGER (le drone")
    print("     se couperait seul en vol). 'q' pour finir.")
    print("=" * 70)

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    finally:
        _set_autorepeat(True)

    print("\n" + "=" * 70)
    print(f"RÉSUMÉ : {state['press']} appuis, {state['kills']} [KILL], "
          f"{state['false_releases']} faux relâchés (auto-répétition)")
    if state["false_releases"] == 0:
        print("✅ OK : aucun faux relâché -> kill instantané SÛR à voler (dead-man fiable).")
    else:
        print("❌ DANGER : auto-répétition active -> le kill se déclencherait EN VOL.")
        print("   Corrige avant de voler : `xset r off` manuellement, ou installe x11-xserver-utils.")
    print("=" * 70)


if __name__ == "__main__":
    main()
