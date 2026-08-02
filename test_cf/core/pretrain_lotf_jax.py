"""Pré-entraînement BPTT de la politique LOTF (lotf-JAX) — CONFIGURABLE.

Porte fidèlement `examples/state_hovering/1_train_base_policy.ipynb` (envs,
wrappers, modules, algos de `lotf`), mais expose la MASSE du quad et un RÉSIDU
de masse optionnel pendant l'entraînement, pour comparer plusieurs façons de
gérer l'écart de masse sim↔réel (le vrai Crazyflie pèse ~33 g, pas 27 g).

╔══════════════════════════════════════════════════════════════════════════╗
║  TROIS CONFIGS PRÊTES (--preset) — voir README "Comparer des politiques"   ║
║                                                                            ║
║  base27  : quad 27 g, sans résidu  (= politique de base actuelle ; à       ║
║            finetuner ensuite -> c'est la config 3, "pretrain 27 g + FT").  ║
║  mass33  : quad 33 g, sans résidu  (config 1 : la masse réelle est BAKÉE   ║
║            dans la dynamique du pretrain).                                  ║
║  res27   : quad 27 g + RÉSIDU de masse a_res=coeff·f_d·R_z émulant 33 g     ║
║            (config 2 : le pretrain voit la dynamique lourde via le résidu, ║
║            la MÊME formule que le finetune en ligne).                       ║
╚══════════════════════════════════════════════════════════════════════════╝

Pourquoi ces trois : si une politique pré-adaptée à la masse réelle (mass33 /
res27) vole MIEUX que "base27 + finetune en ligne", alors la dégradation
observée en réel vient du FINETUNE en vol (sur-adaptation / sim-to-real du
résidu), pas d'une incapacité à corriger la masse.

Fix important vs l'ancienne version : on sauve `action_bias = env.hovering_action`
(= 9.81·masse_du_quad), PAS un 0.265 codé en dur. Pour un quad 33 g la poussée de
hover est 0.324 N ; figer 0.265 cassait la politique. `Actor` (RL_policy.py) charge
ce buffer en strict=True -> la bonne poussée de hover part sur le drone.

Usage (depuis ../.venv qui contient jax+lotf+torch) :
    python core/pretrain_lotf_jax.py --preset base27 --out models/model_pretrain.pt
    python core/pretrain_lotf_jax.py --preset mass33 --out models/model_33g.pt
    python core/pretrain_lotf_jax.py --preset res27  --out models/model_res33.pt
    # ou réglages manuels :
    python pretrain_lotf_jax.py --mass 0.030 --residual-m-real 0.035 --out ...
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import jax
import jax.numpy as jnp
import optax
import yaml
from flax.training.train_state import TrainState

THIS = Path(__file__).resolve().parent                  # core/
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())   # racine du dépôt
TEST_CF = REPO / "test_cf"
sys.path.insert(0, str(TEST_CF / "RL-real"))
sys.path.insert(0, str(REPO))

import cf_params as P                                   # noqa: E402  (DT, DELAY, HOVER_GOAL, MASS, ...)
from lotf_config import CFG                              # noqa: E402  (uniquement pour le cache XLA)

# Cache de compilation XLA (comme le finetune) : warmup ~15 s -> ~6 s ensuite.
_cc = CFG.get("jax", {}).get("compilation_cache_dir") if hasattr(CFG, "get") else None
if _cc:
    _cc = _cc if Path(_cc).is_absolute() else str(TEST_CF / _cc)
    jax.config.update("jax_compilation_cache_dir", _cc)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from lotf import LOTF_PATH                               # noqa: E402
from lotf.objects import Quadrotor                       # noqa: E402
from lotf.envs import HoveringStateEnv                   # noqa: E402
from lotf.envs.wrappers import MinMaxObservationWrapper, LogWrapper, VecEnv  # noqa: E402
from lotf.algos import bptt                              # noqa: E402
from lotf.utils.residual_dynamics import create_vec_funcs  # noqa: E402
from lotf_jax_bridge import make_lotf_mlp, flax_to_torch_sd  # noqa: E402

# ── Hyperparamètres = examples/state_hovering/1_train_base_policy.ipynb (codés en dur) ──
SIM_DT          = 0.02         # = cf_params.DT
DELAY           = 0.04         # = cf_params.DELAY
# Horizon d'épisode BPTT. Le notebook utilise 3.0 s, mais à 3 s la dérive LENTE due au biais
# de rate cuit (cf genesis-33g-drift-is-policy) n'entre pas dans le coût -> la policy 33g
# dérive (~1.7 m @10s). Passer à 6 s + grad-clip 0.5 (DEFAULT_GRAD_CLIP) ÉLIMINE la dérive
# (33g 1.7 m -> ~1 cm, robuste sur 3 seeds) en gardant la 27g propre. Fix plus robuste que la
# pénalité L1 (OMEGA_CMD_PENALTY) qui, elle, a une zone morte. Override : --max-sim-time 3.
MAX_SIM_TIME      = 6.0        # s -> 300 pas / épisode (notebook : 3.0)
DEFAULT_GRAD_CLIP = 0.5        # requis sur horizon long (BPTT explosif sur plant instable)
NUM_ENVS        = 200
MAX_EPOCHS      = 200
LR              = 2e-3         # cosine_decay_schedule(LR, MAX_EPOCHS) ; adam, sans grad-clip.
                              # 2e-3 donne le meilleur transfert Genesis @27g (err ~8 cm vs ~38 cm à 5e-3).
# env (randomisation + reward) — valeurs EXACTES du notebook
YAW_SCALE       = 1.0
PITCH_ROLL      = 0.1
VELOCITY_STD    = 0.1
OMEGA_STD       = 0.1
REWARD_SHARP    = 3.0
ACTION_PENALTY  = 0.5
# Pénalité L1 OPTIONNELLE sur le RATE COMMANDÉ (action[1:]). Le smooth_l1 d'action_penalty
# est quadratique près de 0 -> gradient nul en 0 -> un biais ω stationnaire (~0.01 rad/s)
# est quasi gratuit et se fait cuire dans la policy au pretrain. |ω_cmd| a un gradient
# CONSTANT en 0 -> pince le rate de hover à zéro.
# ⚠ OFF PAR DÉFAUT : le fix d'horizon long (MAX_SIM_TIME=6 + grad-clip) est plus robuste.
# Le L1 a une FENÊTRE ÉTROITE (0.002 réduit la dérive 33g 1.7->0.03 m, >=0.005 DÉSTABILISE
# via une zone morte qui supprime les petites corrections) et reste sensible au seed.
OMEGA_CMD_PENALTY = 0.0
# Pénalité sur le TAUX de variation de l'action (Δ entre 2 commandes consécutives). À horizon
# long la policy serre le suivi -> commandes de poussée vives -> excite le lag d'actionnement
# de Genesis/du vrai drone -> OSCILLATION VERTICALE en début de mouvement. smooth_l1 du Δ
# l'amortit (quadratique -> pas de zone morte). ⚠ ne se reproduit PAS en low-fi (point-masse
# sans lag) -> à régler dans Genesis. 0 = off ; essayer 0.1-0.3 si oscillation z.
ACTION_RATE_PENALTY = 0.0
# Lag d'actionneur 1er ordre sur la poussée [s] (cste de temps spin-up moteur), EN PLUS du
# delay pur (latence radio/calcul). La low-fi appliquait la poussée instantanément -> policy
# trop vive sur z -> oscillation verticale quand elle pilote la poussée laggée de Genesis.
# Ajouter ~40 ms rapproche la low-fi de Genesis -> la policy apprend l'amortissement. 0 = off.
ACTUATOR_TAU = 0.04
MARGIN          = 0.5
QUAD            = "crazyflie_quad"   # (exemple : example_quad)

# Masses de référence (kg). Le vrai Crazyflie mesuré ≈ 33.4 g (m_eff de real_run_lotf.npz,
# coeff ∂a_z/∂T = -7.10). 33 g = arrondi "théorique". Nominal sim = 27 g (cf_params.MASS).
MASS_NOMINAL = P.MASS          # 0.027
MASS_REAL    = 0.033           # cible "masse réelle" (théorique) ; mesuré ~0.0334

# Presets nommés : (masse du quad, m_real émulée par le résidu ou None = pas de résidu).
PRESETS = {
    "base27": (MASS_NOMINAL, None),        # config 3 (base à finetuner ensuite)
    "mass33": (MASS_REAL,    None),        # config 1 (masse réelle bakée)
    "res27":  (MASS_NOMINAL, MASS_REAL),   # config 2 (résidu de masse pendant le pretrain)
}


def dummy_residual_params(num_models=3):
    """Pytree résiduel de la BONNE structure pour satisfaire `bptt.train`/`env.step`.
    En résidu exact (exact_mass_residual) les params sont IGNORÉS ; sans résidu
    (use_forward_residual=False) jamais utilisés. Sert juste à satisfaire l'API."""
    init_fn, _, _ = create_vec_funcs()
    _, states = init_fn(1e-2, jnp.arange(num_models, dtype=jnp.int32))
    return states.params


def build_quad(mass, residual_m_real):
    """Quad crazyflie avec masse surchargée et, si `residual_m_real`, un résidu de
    masse EXACT a_res = (1/m_real - 1/mass)·f_d·R_z (émule un quad plus lourd sans
    changer la masse intégrée). m_nominal du résidu = masse du quad (le sim divise
    f_d par cette masse), pour que la correction reproduise bien `residual_m_real`."""
    with open(os.path.join(LOTF_PATH, "objects/quadrotor_files/crazyflie_quad.yaml")) as f:
        cf = yaml.safe_load(f)
    cf = dict(cf); cf["mass"] = float(mass)
    if residual_m_real is None:
        dyn = {"use_high_fidelity": False, "use_forward_residual": False}
    else:
        dyn = {"use_high_fidelity": False, "use_forward_residual": True,
               "exact_mass_residual": {"m_nominal": float(mass), "m_real": float(residual_m_real)}}
    return Quadrotor.from_dict(cf, dyn), (residual_m_real is not None)


def train(target, epochs, num_envs, lr, seed, mass, residual_m_real, omega_cmd_penalty,
          max_sim_time, grad_clip, action_rate_penalty, actuator_tau):
    # ── 2-3. quad + env (mêmes wrappers/ordre que le notebook) ──
    quad, has_res = build_quad(mass, residual_m_real)
    env = HoveringStateEnv(
        max_steps_in_episode=int(max_sim_time / SIM_DT),
        dt=SIM_DT, delay=DELAY,
        yaw_scale=YAW_SCALE, pitch_roll_scale=PITCH_ROLL,
        velocity_std=VELOCITY_STD, omega_std=OMEGA_STD,
        quad_obj=quad, reward_sharpness=REWARD_SHARP,
        action_penalty_weight=ACTION_PENALTY,
        omega_cmd_penalty_weight=omega_cmd_penalty,
        action_rate_penalty_weight=action_rate_penalty,
        actuator_tau=actuator_tau, margin=MARGIN,
        hover_target=list(target),
    )
    env = MinMaxObservationWrapper(env)
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    hovering_action = np.asarray(env.hovering_action)      # = 9.81·masse (poussée de hover)
    env = LogWrapper(env)
    env = VecEnv(env)
    res_str = f"+résidu masse->{residual_m_real*1e3:.1f}g" if has_res else "sans résidu"
    print(f"[pretrain-jax] env=HoveringStateEnv({QUAD} @ {mass*1e3:.1f}g {res_str})  "
          f"obs={obs_dim} act={action_dim}  goal={list(target)}  hover_T={hovering_action[0]:.3f} N  "
          f"omega_cmd_penalty(L1)={omega_cmd_penalty}  horizon={max_sim_time}s  grad_clip={grad_clip}  "
          f"action_rate_penalty={action_rate_penalty}  actuator_tau={actuator_tau}s")

    # ── 4. réseau (TANH pour torch) + optim (adam + cosine ; grad-clip optionnel) ──
    policy_net = make_lotf_mlp(obs_dim, action_dim, hovering_action, initial_scale=0.01)
    key = jax.random.key(seed)
    key_init, key_bptt = jax.random.split(key, 2)
    policy_params = policy_net.initialize(key_init)

    scheduler = optax.cosine_decay_schedule(lr, epochs)
    # Sur horizon long le BPTT traverse un plant instable -> gradient explosif : clip de la
    # norme globale (comme le finetune/le bloc pretrain de lotf_config). 0 = off (= notebook).
    tx = (optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(scheduler))
          if grad_clip and grad_clip > 0 else optax.adam(scheduler))
    train_state = TrainState.create(apply_fn=policy_net.apply, params=policy_params, tx=tx)

    # ── 6. train (init envs puis bptt.train, signature identique au notebook) ──
    key_bptt, key_ = jax.random.split(key_bptt)
    init_env_state, init_obs = env.reset(jax.random.split(key_, num_envs), None)

    t0 = time.time()
    res = bptt.train(
        env, init_env_state, init_obs, train_state,
        num_epochs=epochs, num_steps_per_epoch=env.max_steps_in_episode,
        num_envs=num_envs, res_model_params=dummy_residual_params(), key=key_bptt,
    )
    returns = -np.asarray(res["metrics"])
    print(f"[pretrain-jax] {epochs} epochs en {time.time()-t0:.1f}s  "
          f"| return: début={returns[0]:.2f}  final={returns[-1]:.2f}  max={returns.max():.2f}")
    return res["runner_state"].train_state.params, hovering_action


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="config prête : base27 (27g) | mass33 (33g) | res27 (27g+résidu->33g)")
    ap.add_argument("--mass", type=float, default=None, help="masse du quad [kg] (sinon --preset/27g)")
    ap.add_argument("--residual-m-real", type=float, default=None,
                    help="si fixé : résidu de masse émulant ce m_real [kg] (sinon --preset/aucun)")
    ap.add_argument("--out", default="models/model_pretrain_jax.pt",
                    help="Sortie .pt torch (chargeable par RL_policy.RLPolicy).")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--num-envs", type=int, default=NUM_ENVS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--omega-cmd-penalty", type=float, default=OMEGA_CMD_PENALTY,
                    help="poids de la pénalité L1 sur le rate commandé (pince le biais ω de hover ; 0 = off)")
    ap.add_argument("--max-sim-time", type=float, default=MAX_SIM_TIME,
                    help="horizon d'épisode BPTT [s] (3=notebook ; ↑ pour que la dérive lente entre dans le coût)")
    ap.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP,
                    help="clip de la norme globale du gradient BPTT (0=off ; ~0.5 requis sur horizon long)")
    ap.add_argument("--action-rate-penalty", type=float, default=ACTION_RATE_PENALTY,
                    help="pénalise le Δ d'action entre commandes (amortit l'oscillation z due au lag ; 0=off, essayer 0.1-0.3)")
    ap.add_argument("--actuator-tau", type=float, default=ACTUATOR_TAU,
                    help="lag actionneur 1er ordre sur la poussée [s] (rapproche la low-fi de Genesis ; 0=off)")
    ap.add_argument("--target", type=float, nargs=3, default=list(P.HOVER_GOAL))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # priorité : flags explicites > preset > défaut (base27)
    p_mass, p_res = PRESETS[args.preset] if args.preset else PRESETS["base27"]
    mass = args.mass if args.mass is not None else p_mass
    residual_m_real = args.residual_m_real if args.residual_m_real is not None else (
        None if args.mass is not None else p_res)   # --mass seul désactive le résidu du preset

    flax_params, hovering_action = train(
        args.target, args.epochs, args.num_envs, args.lr, args.seed, mass, residual_m_real,
        args.omega_cmd_penalty, args.max_sim_time, args.grad_clip, args.action_rate_penalty,
        args.actuator_tau)

    # FIX : sauver le biais de hover RÉEL du quad (= 9.81·masse), pas P.HOVERING_ACTION fixe.
    sd = flax_to_torch_sd(flax_params, hovering_action)
    out = TEST_CF / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": sd}, out)
    print(f"[pretrain-jax] politique torch -> {out}  (hover_bias={hovering_action[0]:.3f} N ; "
          f"revolable dans Genesis / vrai drone)")


if __name__ == "__main__":
    main()
