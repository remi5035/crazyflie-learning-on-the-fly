"""DIAGNOSTIC : rejoue base.pt et ft.pt DANS l'env de finetune (HoveringStateEnv
+ résidu), pas dans Genesis, pour savoir si l'oscillation observée dans Genesis
vient du résidu/BPTT ou d'un mismatch sim-to-sim.

On reconstruit EXACTEMENT le pipeline du worker : résidu re-fit depuis le log du
dernier finetune en ligne (measurements/online_ft/log.npz), env augmenté `use_forward_residual`.
Puis on déroule chaque politique sous les MÊMES graines et on mesure, en régime
établi : le jitter de thrust (écart-type + variation pas-à-pas de l'action[0]) et
la dérive xy (obs rel-pos).

Lecture :
  - ft LISSE ici mais oscille dans Genesis  -> mismatch sim-to-sim / delay
  - ft OSCILLE déjà ici (≫ base)             -> le résidu a déstabilisé le modèle

À lancer dans .venv (jax+lotf+torch) :
    ../.venv/bin/python diag_replay_ft_env.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import jax
import jax.numpy as jnp

THIS = Path(__file__).resolve().parent                  # tests/
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())   # racine du dépôt
TEST_CF = REPO / "test_cf"
sys.path.insert(0, str(TEST_CF / "core"))
sys.path.insert(0, str(TEST_CF / "RL-real"))
sys.path.insert(0, str(REPO))

import cf_params as P                                   # noqa: E402
from lotf_config import CFG                              # noqa: E402
from lotf_residual import build_residual_dataset        # noqa: E402
from lotf_jax_bridge import torch_sd_to_flax, make_lotf_mlp  # noqa: E402
import finetune_lotf_jax as F                           # noqa: E402
from finetune_lotf_worker import resample               # noqa: E402  (même ré-échantillonnage)
from lotf.objects import Quadrotor                       # noqa: E402
from lotf.envs import HoveringStateEnv                   # noqa: E402
from lotf.envs.wrappers import MinMaxObservationWrapper, LogWrapper, VecEnv  # noqa: E402

ONLINE_FT_WINDOW_SEC = CFG["online"]["window_sec"]
ONLINE_FT_BPTT_EPOCHS = CFG["online"]["bptt_epochs"]
NUM_SAMPLES = CFG["online"]["num_samples"]
NUM_ENVS = 8
STEPS = 250                  # 5 s : assez pour voir un éventuel limit cycle lent


def build_replay_ctx(target, noise=True):
    """Comme `F.build_bptt_context` mais avec un interrupteur de bruit de process.

    `noise=False` (velocity_std=omega_std=0) -> rollout quasi déterministe : on
    observe le comportement INTRINSÈQUE de la politique (limit cycle ou non),
    sans le jitter de réaction au bruit qui masque tout."""
    std = 0.1 if noise else 0.0
    quad = Quadrotor.from_name(F.QUAD, {"use_high_fidelity": False, "use_forward_residual": True})
    env = HoveringStateEnv(
        max_steps_in_episode=STEPS + 1, dt=P.DT, delay=P.DELAY,
        yaw_scale=1.0, pitch_roll_scale=0.1, velocity_std=std, omega_std=std,
        quad_obj=quad, reward_sharpness=3.0, action_penalty_weight=0.5,
        margin=0.5, hover_target=list(target))
    env = MinMaxObservationWrapper(env)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    env = VecEnv(LogWrapper(env))
    policy_net = make_lotf_mlp(obs_dim, action_dim, np.asarray(env.hovering_action))
    return {"env": env, "policy_net": policy_net, "obs_dim": obs_dim}


def load_flax(pt_path):
    ckpt = torch.load(pt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    return torch_sd_to_flax(sd)


def rollout(ctx, policy_params, res_params, seed=0):
    """Déroule la politique dans l'env augmenté (mêmes appels que bptt.train).
    Renvoie actions (STEPS, NUM_ENVS, 4) et obs (STEPS, NUM_ENVS, obs_dim)."""
    env = ctx["env"]
    apply_fn = ctx["policy_net"].apply
    key = jax.random.key(seed)
    key, kr = jax.random.split(key)
    env_state, obs = env.reset(jax.random.split(kr, NUM_ENVS), None)

    # Mêmes clips d'action que la frontière Genesis (drone_controller._send_action /
    # lotf_genesis_env.step) : sans quoi l'env LOTF laisse passer des ω bien plus
    # agressifs que ce que le vrai actionneur/Genesis autorise (ω à ±SEND_OMEGA_MAX).
    a_lo = jnp.asarray(P.ACTION_LOW); a_hi = jnp.asarray(P.ACTION_HIGH)

    def clip_action(a):
        a = jnp.clip(a, a_lo, a_hi)
        om = jnp.clip(a[..., 1:], -P.SEND_OMEGA_MAX, P.SEND_OMEGA_MAX)
        return jnp.concatenate([a[..., :1], om], axis=-1)

    def step_fn(carry, _):
        env_state, obs, key = carry
        action = clip_action(apply_fn(policy_params, obs))
        key, k_ = jax.random.split(key)
        env_state, obs2, *_ = env.step(env_state, action, res_params, jax.random.split(k_, NUM_ENVS))
        return (env_state, obs2, key), (action, obs)

    _, (actions, obses) = jax.lax.scan(step_fn, (env_state, obs, key), None, STEPS)
    return np.asarray(actions), np.asarray(obses)


def metrics(actions, obses, tag):
    """Stats en régime établi (2e moitié du rollout), moyennées sur les envs."""
    s = STEPS // 2
    thrust = actions[s:, :, 0]                       # (T, E)
    omega = actions[s:, :, 1:]                       # (T, E, 3)
    relxy = obses[s:, :, 0:2]                         # rel-pos x,y (normalisée)
    relz = obses[s:, :, 2]

    thr_std = thrust.std(axis=0).mean()
    thr_p2p = (thrust.max(axis=0) - thrust.min(axis=0)).mean()
    thr_jit = np.abs(np.diff(thrust, axis=0)).mean()  # variation pas-à-pas (jitter HF)
    om_std = omega.std(axis=0).mean()
    xy_std = relxy.std(axis=0).mean()
    xy_wander = np.linalg.norm(relxy - relxy.mean(axis=0, keepdims=True), axis=2).mean()
    z_mean = relz.mean()

    print(f"\n── {tag} (régime établi, {NUM_ENVS} envs) ──")
    print(f"  thrust   : std={thr_std:.4f} N   peak-to-peak={thr_p2p:.4f} N   "
          f"jitter(|Δ pas|)={thr_jit:.4f} N")
    print(f"  omega cmd: std={om_std:.4f} rad/s")
    print(f"  xy        : std={xy_std:.4f} (norm.)   wander={xy_wander:.4f} (norm.)")
    print(f"  z rel moy : {z_mean:+.4f} (norm.)")
    return dict(thr_std=thr_std, thr_p2p=thr_p2p, thr_jit=thr_jit, xy_std=xy_std)


def main():
    # 1) résidu re-fit depuis le log du dernier finetune en ligne (même pipeline)
    log_path = TEST_CF / "measurements" / "online_ft" / "log.npz"
    log = {k: v for k, v in np.load(log_path).items()}
    win = None if ONLINE_FT_WINDOW_SEC <= 0 else ONLINE_FT_WINDOW_SEC
    X, y = build_residual_dataset(log, window_sec=win)
    np.random.seed(0)                                # ré-échantillonnage reproductible
    Xr, yr = resample(X, y, NUM_SAMPLES)
    print(f"[diag] résidu re-fit depuis {log_path.name}: {X.shape[0]} samples "
          f"-> N={NUM_SAMPLES}, |y|_med={np.median(np.linalg.norm(y, axis=1)):.3f} m/s²")
    res_params = F.fit_residual_ensemble(Xr, yr)

    base = load_flax(TEST_CF / "measurements" / "online_ft" / "base.pt")
    ft = load_flax(TEST_CF / "measurements" / "online_ft" / "ft.pt")
    res_zero = jax.tree_util.tree_map(lambda x: x * 0.0, res_params)  # contrôle = dyn. nominale

    # Tout SANS bruit de process -> comportement intrinsèque des politiques.
    print("\n########## ENV SANS BRUIT (comportement intrinsèque) ##########")
    # Contrôle : base avec résidu=0 (même chemin de code = dynamique nominale).
    ctx0 = build_replay_ctx(list(P.HOVER_GOAL), noise=False)
    m_ctrl = metrics(*rollout(ctx0, base, res_zero, seed=0), tag="BASE, résidu=0 (nominal)")
    mb = metrics(*rollout(ctx0, base, res_params, seed=0), tag="BASE + résidu")
    mf = metrics(*rollout(ctx0, ft, res_params, seed=0), tag="FINETUNÉE (ft) + résidu")

    print("\n=== VERDICT ===")
    nominal = m_ctrl["thr_jit"]
    print(f"  jitter thrust : nominal={nominal:.4f}  base+résidu={mb['thr_jit']:.4f}  "
          f"ft+résidu={mf['thr_jit']:.4f} N/pas")
    r_res = mb["thr_jit"] / max(nominal, 1e-9)        # effet du RÉSIDU sur une politique lisse
    r_ft = mf["thr_jit"] / max(nominal, 1e-9)         # effet sur la politique FINETUNÉE
    print(f"  ratios vs nominal : base+résidu={r_res:.2f}x   ft+résidu={r_ft:.2f}x")
    if r_ft > 2.0 and r_res < 1.6:
        print("  -> Le RÉSIDU seul ajoute peu de jitter (base+résidu ≈ nominal), mais la")
        print("     politique FINETUNÉE est ~%.1fx plus jittery EN THRUST, dans son PROPRE" % r_ft)
        print("     env d'entraînement. Le jitter est donc BAKÉ DANS ft par le BPTT (il se")
        print("     reproduit hors Genesis) -> ce n'est pas un pur mismatch sim-to-sim, ni")
        print("     une instabilité boucle-ouverte du modèle augmenté, mais une solution")
        print("     BPTT à thrust haute fréquence. Leviers : pénaliser Δaction, brider le")
        print("     résidu (reg spectrale ↑ / capacité ↓), BPTT moins agressif.")
    elif r_res > 2.0:
        print("  -> Le résidu déstabilise déjà une politique lisse (base+résidu ≫ nominal).")
        print("     Cause = résidu fantôme / régularisation trop faible.")
    else:
        print("  -> ft ~aussi lisse que base ici : l'oscillation Genesis vient plutôt d'un")
        print("     mismatch sim-to-sim (delay/actionneur env≠Genesis).")


if __name__ == "__main__":
    main()
