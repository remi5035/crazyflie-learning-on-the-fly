"""Harnais jetable : pilote lotf_genesis_eval --online-finetune en headless,
   déclenche 'm' (40 g) puis 'f' (finetune live) et mesure l'erreur avant/après."""
import re, subprocess, sys, threading, time
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"
GENPY = Path(__file__).resolve().parents[2] / "genesis_venv" / "bin" / "python"

cmd = [str(GENPY), str(CORE / "lotf_genesis_eval.py"),
       "--ckpt", "models/model_pretrain.pt", "--online-finetune",
       "--no-viewer", "--mass", "0.027", "--steps", "12000"]
proc = subprocess.Popen(cmd, cwd=str(CORE), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)

state = {"ready": False, "step": 0, "err": None, "ft_done": False, "rollback": False,
         "mass_g": 27, "lines": []}
step_re = re.compile(r"step\s+(\d+).*err=([\d.]+)")

def reader():
    for line in proc.stdout:
        line = line.rstrip("\n")
        state["lines"].append(line)
        if "worker JAX chaud" in line:
            state["ready"] = True
        m = step_re.search(line)
        if m:
            state["step"] = int(m.group(1)); state["err"] = float(m.group(2))
        if "adaptation incrémentale terminée" in line:
            state["ft_done"] = True
        if "rollback" in line:
            state["rollback"] = True
        if any(k in line for k in ["[online-ft]", "[mass]", "[worker]", "@", "résumé",
                                    "erreur position", "DIVERGE", "moyenne résidu"]):
            print("  |", line, flush=True)
        elif m and int(m.group(1)) % 200 == 0:
            print("  | step", m.group(1), "err", m.group(2), "mass", state["mass_g"], "g", flush=True)

threading.Thread(target=reader, daemon=True).start()

def wait_until(pred, timeout, label):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        if proc.poll() is not None:
            print(f"[harness] process terminé prématurément pendant '{label}'"); return False
        time.sleep(0.2)
    print(f"[harness] TIMEOUT en attendant '{label}'"); return False

def send(c):
    try:
        proc.stdin.write(c + "\n"); proc.stdin.flush()
    except (BrokenPipeError, ValueError):
        print(f"[harness] pipe fermé en envoyant '{c}'")

print("[harness] attente warmup worker JAX...")
wait_until(lambda: state["ready"], 120, "worker ready")
print("[harness] worker chaud. On vole ~3 s à 27 g puis on passe à 40 g.")
s0 = state["step"]; wait_until(lambda: state["step"] - s0 > 150, 60, "vol 27g")
# Avec le fix UX, un seul 'm' saute le no-op et va direct à 40 g.
send("m"); state["mass_g"] = 40
print("[harness] >>> 'm' envoyé (cible 40 g). Vol ~4 s pour remplir la fenêtre résidu.")
s1 = state["step"]; wait_until(lambda: state["step"] - s1 > 200, 60, "drift 40g")
err_before = state["err"]
print(f"[harness] erreur AVANT finetune (à 40 g) = {err_before}")
send("f")
print("[harness] >>> 'f' envoyé (finetune live).")
wait_until(lambda: state["ft_done"] or state["rollback"], 180, "finetune")
# laisser voler ~4 s pour stabiliser après le dernier hot-swap
s2 = state["step"]; wait_until(lambda: state["step"] - s2 > 200, 60, "post-finetune")
err_after = state["err"]
send("q"); time.sleep(2)
try: proc.wait(timeout=10)
except Exception: proc.kill()

print("\n========== RÉSULTAT LIVE ==========")
print(f"err à 40 g AVANT finetune : {err_before}")
print(f"err à 40 g APRÈS finetune : {err_after}")
print(f"rollback déclenché        : {state['rollback']}")
print(f"finetune terminé          : {state['ft_done']}")
