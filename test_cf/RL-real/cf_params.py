"""Crazyflie 2.x parameters + LOTF env constants (single source of truth).

These constants are shared by lotf_sim.py, drone_state.py, drone_controller.py,
pretrain_lotf.py and finetune_lotf.py so changing the platform only requires
editing this file.

Matches the layout of `scripts/rl_controller_lotf.py` + `lotf/envs/hovering_state_env.py`.
"""
import numpy as np

# ───────────────────────────── Physical platform ─────────────────────────────
MASS = 0.027                  # kg     — Crazyflie 2.1 nominal
G = 9.81                      # m/s^2
THRUST_MAX_PER_ROTOR = 0.15   # N      — ~max thrust per rotor at 100% PWM
THRUST_MIN_PER_ROTOR = 0.0    # N
OMEGA_MAX = 0.5              # rad/s  — per-axis body-rate command bound (policy/sim)

# Safety clamp applied ONLY when sending to the real Crazyflie (first flights).
# The policy still sees / logs its full-authority action; only the radio setpoint
# is softened. Reference: the working RL-real cascade-1 setup commanded ±0.2 rad/s
# (±11.5 deg/s). 0.5 rad/s ≈ 28.6 deg/s gives a bit more room while staying gentle.
# Set to OMEGA_MAX to disable the clamp.
SEND_OMEGA_MAX = 0.5          # rad/s  — per-axis body-rate cap on the radio setpoint

# Derived (must match the LOTF env)
THRUST_MIN_TOTAL = 4 * THRUST_MIN_PER_ROTOR     # 0.0 N
THRUST_MAX_TOTAL = 4 * THRUST_MAX_PER_ROTOR     # 0.6 N
T_HOVER = MASS * G    
HOVER_PWM_PCT = 65.0    # %PWM mesuré au hover sur le vrai drone (= ta mesure)                          # ≈ 0.265 N

# Action bounds (SI, matches HoveringStateEnv.action_space)
ACTION_LOW = np.array([THRUST_MIN_TOTAL, -OMEGA_MAX, -OMEGA_MAX, -OMEGA_MAX], dtype=np.float32)
ACTION_HIGH = np.array([THRUST_MAX_TOTAL, OMEGA_MAX, OMEGA_MAX, OMEGA_MAX], dtype=np.float32)
HOVERING_ACTION = np.array([T_HOVER, 0.0, 0.0, 0.0], dtype=np.float32)

# ───────────────────────────── Env / control loop ────────────────────────────
DT = 0.02                       # 50 Hz — matches LOTF sim
DELAY = 0.04                    # s
NUM_LAST_ACTIONS = int(np.ceil(DELAY / DT)) + 1   # = 3
HOVER_GOAL = np.array([0.0, 0.0, 0.5], dtype=np.float32)
WORLD_BOX_HALF = 1.5            # m   — pos bounds = goal ± WORLD_BOX_HALF

# Two-phase scripted goal (immediate switch, like the legacy RL-real controller).
# Time is measured from the start of the RL window (t_rl = 0 when RL takes over).
GOAL_PHASE1 = np.array([0.0, 0.0, 0.5], dtype=np.float32)   # used while t_rl <  GOAL_SWITCH_T
GOAL_PHASE2 = np.array([0.0, 0.0, 0.5], dtype=np.float32)   # used while t_rl >= GOAL_SWITCH_T
GOAL_SWITCH_T = 10.0           # s — instant of the immediate switch
V_BOUND = 5.0                   # m/s — vel obs bound

# Observation layout: pos(3) + R_flat(9) + vel(3) + last_actions(NUM_LAST_ACTIONS*4)
NUM_OBS = 3 + 9 + 3 + 4 * NUM_LAST_ACTIONS    # = 27
NUM_ACTIONS = 4

# MLP shape (LOTF base policy)
HIDDEN_DIMS = [512, 512]

# ──────────────────────────── Crazyflie cflib mapping ────────────────────────
# Thrust SI → setpoint percentage : map HONNÊTE proportionnelle à la force commandée
# (0 N → 0 %, THRUST_MAX_TOTAL → 100 %). Même philosophie que l'actionneur honnête de
# Genesis : la commande %PWM est STRICTEMENT proportionnelle à T_N, sans recentrage
# autour du hover. L'ancienne affine (hover→82 %) envoyait ~68 % même pour 0 N, ce qui
# découplait la poussée de la commande autour du hover (cf calibrate_thrust.py).
# ⚠ Conséquence : le hover (T_HOVER ≈ 0.265 N) tombe désormais à ~44 % PWM — à VÉRIFIER
# au sol/vol court sur le vrai drone (si le hover réel exige un autre %, recalibrer
# THRUST_MAX_TOTAL ou passer à un fit honnête mesuré, cf option « calibrer le vrai drone »).
def thrust_N_to_pwm_pct(T_N: float) -> float:
    pct = HOVER_PWM_PCT * (T_N / T_HOVER)   # proportionnel, passe par (T_HOVER, 82%)
    return float(np.clip(pct, 0.0, 100.0))

# Body-rate setpoint: send_setpoint_manual expects deg/s for roll/pitch and a
# yaw angle (LOTF stays in rate mode for yaw too — we keep the same convention).
RAD_TO_DEG = 180.0 / np.pi
