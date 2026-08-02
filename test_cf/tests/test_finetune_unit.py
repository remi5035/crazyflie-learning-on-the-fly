"""Tests UNITAIRES du finetuning LOTF (résidu + BPTT + pont torch↔flax).

Chaque test a une VÉRITÉ TERRAIN connue (données synthétiques ou résidu exact),
donc un échec pointe une régression précise — contrairement aux notebooks qui
valident "de bout en bout". À lancer dans `.venv` (jax + lotf + torch) :

    cd test_cf
    ../.venv/bin/python tests/test_finetune_unit.py          # tout, avec PASS/FAIL
    ../.venv/bin/python -m pytest tests/test_finetune_unit.py -q   # via pytest

Couvre :
  1. pont torch↔flax : aller-retour identité (sinon le .pt revolé est faux).
  2. resample : PAS de bootstrap-upsampling (la régression qui faisait diverger le live).
  3. résidu : récupère un offset CONSTANT et GÉNÉRALISE (train/test).
  4. résidu : apprend le bon GRADIENT ∂a_z/∂T sur un offset de masse (ce que le BPTT
     exploite) quand les données couvrent une plage de thrust — et AVERTIT que sur un
     hover statique (thrust quasi constant) ce gradient est mal contraint.
  5. BPTT : sur un résidu EXACT de masse, l'offset de hover DIMINUE (brique d'optim).
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import torch
import yaml

THIS = Path(__file__).resolve().parent
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())
TEST_CF = REPO / "test_cf"
for p in (TEST_CF / "core", TEST_CF / "RL-real", REPO):
    sys.path.insert(0, str(p))

import cf_params as P                                          # noqa: E402
from lotf import LOTF_PATH                                     # noqa: E402
from lotf.objects import Quadrotor                             # noqa: E402
from lotf.envs import HoveringStateEnv, rollout                # noqa: E402
from lotf.envs.wrappers import MinMaxObservationWrapper, LogWrapper, VecEnv  # noqa: E402
from lotf.algos import bptt                                    # noqa: E402
import optax                                                   # noqa: E402
from flax.training.train_state import TrainState               # noqa: E402

import finetune_lotf_jax as F                                  # noqa: E402
from finetune_lotf_worker import resample                      # noqa: E402
from lotf_jax_bridge import make_lotf_mlp, torch_sd_to_flax, flax_to_torch_sd  # noqa: E402

M_NOM, M_REAL = 0.027, 0.040
COEFF = 1.0 / M_REAL - 1.0 / M_NOM         # ∂a_z/∂T du résidu exact de masse (≈ -12.0)


# ───────────────────────── helpers synthétiques ─────────────────────────
def _synth_X(n, t_lo=0.30, t_hi=0.45, seed=0):
    """Features 19-d ~hover : p~0, R=I, v petit, T dans [t_lo,t_hi], ω petit."""
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.05, (n, 3))
    R = np.tile(np.eye(3).reshape(1, 9), (n, 1))
    v = rng.normal(0, 0.05, (n, 3))
    T = rng.uniform(t_lo, t_hi, (n, 1))
    om = rng.normal(0, 0.05, (n, 3))
    return np.concatenate([p, R, v, T, om], axis=1).astype(np.float32)


def _predict_mean(params, X):
    _, _, predict_fn = F._vec_funcs()
    pred = np.asarray(predict_fn(params, jnp.asarray(X)))   # (num_models, n, 3)
    return pred.mean(axis=0)                                # (n, 3) moyenne d'ensemble


# ───────────────────────── 1. pont torch↔flax ─────────────────────────
def test_bridge_roundtrip():
    rng = np.random.default_rng(0)
    sd = {
        "action_bias": torch.tensor(np.asarray(P.HOVERING_ACTION, np.float32)),
        "actor.0.weight": torch.tensor(rng.normal(0, .1, (512, 27)).astype(np.float32)),
        "actor.0.bias":   torch.tensor(rng.normal(0, .1, (512,)).astype(np.float32)),
        "actor.2.weight": torch.tensor(rng.normal(0, .1, (512, 512)).astype(np.float32)),
        "actor.2.bias":   torch.tensor(rng.normal(0, .1, (512,)).astype(np.float32)),
        "actor.4.weight": torch.tensor(rng.normal(0, .1, (4, 512)).astype(np.float32)),
        "actor.4.bias":   torch.tensor(rng.normal(0, .1, (4,)).astype(np.float32)),
    }
    flax = torch_sd_to_flax(sd)
    sd2 = flax_to_torch_sd(flax, P.HOVERING_ACTION)
    for k in ["actor.0.weight", "actor.2.weight", "actor.4.weight",
              "actor.0.bias", "actor.2.bias", "actor.4.bias"]:
        err = np.abs(sd[k].numpy() - sd2[k].numpy()).max()
        assert err < 1e-6, f"pont non-identité sur {k}: {err}"
    print("  [1] pont torch↔flax : aller-retour identité  ✓")


# ───────────────────────── 2. resample (régression) ─────────────────────────
def test_resample_no_upsample_bootstrap():
    rng = np.random.default_rng(0)
    X = np.arange(99 * 19, dtype=np.float32).reshape(99, 19)
    y = np.arange(99 * 3, dtype=np.float32).reshape(99, 3)
    # m<n : NE DOIT PAS bootstrap-upsampler (cause de la divergence live). Tous les
    # 99 échantillons doivent être présents (au plus 1 doublon), de façon déterministe.
    Xr, yr = resample(X, y, 100, rng=np.random.default_rng(0))
    assert Xr.shape == (100, 19)
    uniq = np.unique(Xr, axis=0).shape[0]
    assert uniq >= 99, f"resample m<n perd des échantillons (uniques={uniq}/99)"
    Xr2, _ = resample(X, y, 100, rng=np.random.default_rng(0))
    assert np.array_equal(Xr, Xr2), "resample non déterministe à seed fixe"
    # m>n : sous-échantillon SANS remise (100 distincts depuis 200)
    Xb = np.arange(200 * 19, dtype=np.float32).reshape(200, 19)
    yb = np.zeros((200, 3), np.float32)
    Xs, _ = resample(Xb, yb, 100, rng=np.random.default_rng(0))
    assert np.unique(Xs, axis=0).shape[0] == 100, "downsample devrait être sans remise"
    print("  [2] resample : pas de bootstrap-upsampling, downsample sans remise  ✓")


# ───────────────────────── 3. résidu : offset constant + généralisation ─────────────────────────
def test_residual_recovers_constant_and_generalizes():
    c = np.array([0.05, 0.02, -2.0], np.float32)             # résidu vrai = constante
    Xtr = _synth_X(300, seed=1)
    ytr = np.tile(c, (300, 1)) + np.random.default_rng(2).normal(0, 0.02, (300, 3)).astype(np.float32)
    params = F.fit_residual_ensemble(Xtr, ytr)
    Xte = _synth_X(150, seed=99)                              # données JAMAIS vues
    pred_te = _predict_mean(params, Xte)
    mean_err = np.abs(pred_te.mean(0) - c)
    spread = pred_te.std(0)                                   # doit être ~0 (constante)
    assert (mean_err < 0.1).all(), f"moyenne hors-échantillon fausse: {mean_err}"
    assert (spread < 0.15).all(), f"résidu pas constant hors-échantillon (std={spread})"
    print(f"  [3] résidu constant : moyenne test={pred_te.mean(0).round(3)} (vrai {c}) "
          f"std={spread.round(3)}  ✓")


# ───────────────────────── 4. résidu : gradient ∂a_z/∂T ─────────────────────────
def _learned_dadT(params, T0, dT=0.02):
    """Pente ∂a_z/∂T du résidu appris autour de T0 (R=I, hover), par diff. finie."""
    base = _synth_X(1, seed=7); base[0, 12:15] = 0.0
    xp = base.copy(); xp[0, 15] = T0 + dT
    xm = base.copy(); xm[0, 15] = T0 - dT
    return float((_predict_mean(params, xp)[0, 2] - _predict_mean(params, xm)[0, 2]) / (2 * dT))


def test_residual_thrust_gradient():
    # données = offset de masse EXACT : y_z = COEFF * T (R=I). Thrust couvrant une PLAGE.
    X = _synth_X(400, t_lo=0.25, t_hi=0.50, seed=3)
    y = np.zeros((400, 3), np.float32)
    y[:, 2] = COEFF * X[:, 15]
    params = F.fit_residual_ensemble(X, y)
    slope = _learned_dadT(params, T0=0.39)
    rel = abs(slope - COEFF) / abs(COEFF)
    assert rel < 0.4, f"gradient ∂a_z/∂T faux: appris {slope:.2f} vs exact {COEFF:.2f} (rel {rel:.0%})"
    print(f"  [4] gradient ∂a_z/∂T (plage de thrust) : appris {slope:.2f} ≈ exact {COEFF:.2f}  ✓")
    print(f"      ⚠ rappel : sur un hover STATIQUE (T~constant), ce gradient n'est PAS contraint "
          f"par les données -> risque pour le BPTT.")


# ───────────────────────── 5. BPTT : corrige un offset de masse exact ─────────────────────────
def _eval_offset(policy_params, policy_net, m_real, target, steps=200, n=4, seed=0):
    with open(LOTF_PATH + "/objects/quadrotor_files/crazyflie_quad.yaml") as f:
        cf = yaml.safe_load(f)
    cf = dict(cf); cf["mass"] = m_real
    quad = Quadrotor.from_dict(cf, {"use_high_fidelity": False, "use_forward_residual": False})
    env = MinMaxObservationWrapper(HoveringStateEnv(
        max_steps_in_episode=steps + 1, dt=P.DT, delay=P.DELAY, quad_obj=quad,
        margin=0.5, hover_target=list(target)))

    def policy(obs, key):
        return policy_net.apply(policy_params, obs)
    keys = jax.random.split(jax.random.key(seed), n)
    tr = jax.vmap(rollout, in_axes=(None, 0, None, None))(env, keys, policy, {})
    z = np.asarray(tr.state.quadrotor_state.p[:, :, 2])
    done = np.asarray(tr.terminated) | np.asarray(tr.truncated)
    fd = np.where(done.any(1), done.argmax(1), z.shape[1] - 1)
    finals = [zi[max(0, k - 50):k + 1].mean() for zi, k in zip(z, fd)]
    return float(np.mean(finals)) - target[2]


def test_bptt_reduces_offset_exact_residual():
    target = list(P.HOVER_GOAL)
    ckpt = torch.load(TEST_CF / "models" / "model_pretrain.pt", map_location="cpu")
    base = torch_sd_to_flax(ckpt.get("model_state_dict", ckpt))

    # env d'entraînement = crazyflie nominal + résidu EXACT de masse (≡ quad lourd)
    quad = Quadrotor.from_name("crazyflie_quad", {
        "use_high_fidelity": False, "use_forward_residual": True,
        "exact_mass_residual": {"m_nominal": M_NOM, "m_real": M_REAL}})
    env = HoveringStateEnv(max_steps_in_episode=int(3.0 / P.DT), dt=P.DT, delay=P.DELAY,
                           yaw_scale=1.0, pitch_roll_scale=0.1, velocity_std=0.1, omega_std=0.1,
                           quad_obj=quad, reward_sharpness=3.0, action_penalty_weight=0.5,
                           margin=0.5, hover_target=target)
    env = MinMaxObservationWrapper(env)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    env = VecEnv(LogWrapper(env))
    net = make_lotf_mlp(obs_dim, act_dim, P.HOVERING_ACTION)

    off_before = _eval_offset(base, net, M_REAL, target)
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(optax.cosine_decay_schedule(1e-3, 120)))
    ts = TrainState.create(apply_fn=net.apply, params=base, tx=tx)
    k = jax.random.key(0); kb, kr = jax.random.split(k)
    ies, io = env.reset(jax.random.split(kr, 10), None)
    res = bptt.train(env, ies, io, ts, num_epochs=120, num_steps_per_epoch=env.max_steps_in_episode,
                     num_envs=10, res_model_params={}, key=kb)
    ft = res["runner_state"].train_state.params
    off_after = _eval_offset(ft, net, M_REAL, target)
    assert abs(off_after) < abs(off_before) - 0.02, \
        f"BPTT n'a pas réduit l'offset: {off_before:+.3f} -> {off_after:+.3f} m"
    print(f"  [5] BPTT sur résidu exact : offset {off_before:+.3f} -> {off_after:+.3f} m  ✓")


if __name__ == "__main__":
    tests = [test_bridge_roundtrip, test_resample_no_upsample_bootstrap,
             test_residual_recovers_constant_and_generalizes, test_residual_thrust_gradient,
             test_bptt_reduces_offset_exact_residual]
    fails = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            fails += 1; print(f"  ❌ {t.__name__}: {e}")
        except Exception as e:                                # noqa: BLE001
            fails += 1; print(f"  ❌ {t.__name__} (erreur): {e}")
    print(f"\n{'TOUS OK ✅' if fails == 0 else f'{fails} ÉCHEC(S) ❌'}")
    sys.exit(1 if fails else 0)
