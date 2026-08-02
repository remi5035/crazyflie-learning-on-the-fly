"""Test UNITAIRE du chemin de prod `residual_mode="constant"` — le seul utilisé
en vol (lotf_config.yaml: online.residual_mode=constant) mais qu'AUCUN test ne
couvrait. test_finetune_unit.py:test_bptt_reduces_offset_exact_residual valide le
BPTT via le résidu EXACT injecté dans la CONFIG du quad (`exact_mass_residual`),
PAS via le chemin réel : en prod le quad est construit avec `use_forward_residual`
seul -> il applique le VRAI MLP d'ensemble (get_residual_dyn_model_apply_fn) à des
params `res_model_params`. Le mode "constant" fabrique ces params à la main
(_constant_residual_params) pour que le MLP sorte une CONSTANTE. Si ce hack est
faux (archi MLP, broadcast du biais, gradient non nul), le BPTT s'entraîne contre
un résidu erroné et la finetune "ne marche pas" sans aucun test rouge.

Couvre :
  A. _constant_residual_params : le MLP d'ensemble sort bien mean_y POUR TOUTE
     entrée (std≈0 sur X variés) et ∂a_z/∂T ≈ 0 (toute la raison d'être du mode
     constant : zéro gradient parasite, cf. docstring fit_residual_ensemble).
  B. Pipeline de prod complet (fit mode="constant" -> build_bptt_context ->
     run_bptt) : l'offset de hover sur un quad LOURD diminue. Compare les epochs
     de prod en vol (online.bptt_epochs) à un budget plus large, pour exposer si
     le nb d'epochs du vol est simplement trop faible pour corriger.

À lancer dans `.venv` (jax + lotf + torch) :
    cd test_cf
    ../.venv/bin/python tests/test_finetune_constant.py
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import torch

THIS = Path(__file__).resolve().parent
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())
TEST_CF = REPO / "test_cf"
for p in (TEST_CF / "core", TEST_CF / "RL-real", REPO):
    sys.path.insert(0, str(p))

import cf_params as P                                          # noqa: E402
from lotf_config import CFG                                    # noqa: E402
import finetune_lotf_jax as F                                  # noqa: E402
from lotf_jax_bridge import torch_sd_to_flax                   # noqa: E402

# réutilise les helpers/constantes du test unitaire principal
from test_finetune_unit import _synth_X, _predict_mean, _eval_offset, M_NOM, M_REAL, COEFF  # noqa: E402

# Constante de résidu de masse au hover (R≈I) : y_z = COEFF · T_hover.
T_HOVER = float(P.HOVERING_ACTION[0])
MEAN_Y = np.array([0.0, 0.0, COEFF * T_HOVER], np.float32)     # ≈ [0,0,-3.2]


# ───────── A. params constants : sortie constante + gradient nul ─────────
def test_constant_params_are_truly_constant():
    """Le hack _constant_residual_params doit faire sortir EXACTEMENT mean_y au
    MLP d'ensemble, indépendamment de l'entrée, et avec ∂a/∂x ≡ 0."""
    params = F._constant_residual_params(MEAN_Y)

    # sortie sur un nuage d'entrées TRÈS variées (thrust large, p/v/ω non nuls)
    X = _synth_X(200, t_lo=0.15, t_hi=0.55, seed=11)
    pred = _predict_mean(params, X)                            # (n, 3)
    mean_err = np.abs(pred.mean(0) - MEAN_Y).max()
    spread = pred.std(0).max()                                 # doit être ~0
    assert mean_err < 1e-4, f"sortie ≠ mean_y : err={mean_err:.2e} (mean={pred.mean(0)})"
    assert spread < 1e-4, f"sortie NON constante sur X variés : std={spread:.2e}"

    # gradient ∂a_z/∂T : doit être nul (sinon le BPTT exploite un gradient parasite)
    base = _synth_X(1, seed=7); base[0, 12:15] = 0.0
    xp = base.copy(); xp[0, 15] += 0.05
    xm = base.copy(); xm[0, 15] -= 0.05
    dadT = float((_predict_mean(params, xp)[0, 2] - _predict_mean(params, xm)[0, 2]) / 0.10)
    assert abs(dadT) < 1e-4, f"∂a_z/∂T parasite non nul : {dadT:.2e}"
    print(f"  [A] params constants : sortie={pred.mean(0).round(3)} (cible {MEAN_Y}), "
          f"std={spread:.1e}, ∂a_z/∂T={dadT:.1e}  ✓")


def test_fit_constant_mode_matches_mean():
    """fit_residual_ensemble(mode='constant') doit ignorer le MLP et ne renvoyer
    QUE la moyenne empirique de y (les data peuvent être bruitées/variées)."""
    X = _synth_X(300, seed=5)
    y = np.tile(MEAN_Y, (300, 1)) + np.random.default_rng(1).normal(0, 0.3, (300, 3)).astype(np.float32)
    params = F.fit_residual_ensemble(X, y, mode="constant")
    out = _predict_mean(params, _synth_X(50, seed=42)).mean(0)
    assert np.abs(out - y.mean(0)).max() < 1e-4, f"mode constant ≠ mean(y): {out} vs {y.mean(0)}"
    print(f"  [A'] fit mode=constant : sortie={out.round(3)} = mean(y)={y.mean(0).round(3)}  ✓")


# ───────── B. constant (chemin de prod) vs exact : le modèle du résidu importe ─────────
# Leçon encodée : un écart de MASSE est intrinsèquement PROPORTIONNEL À LA POUSSÉE
# (a_res = (1/m_real - 1/m_nom)·f_d·R_z, cf. exact_mass_residual). Un résidu CONSTANT
# ne matche la vraie dynamique qu'à LA poussée où il a été mesuré (~T_hover nominal) ;
# au nouvel équilibre lourd (T plus grand) il sous-estime la correction -> le BPTT
# converge vers une MAUVAISE poussée de hover -> l'offset n'est PAS corrigé. Le résidu
# EXACT scale avec T -> il corrige. Ce test fige cette différence.
EPOCHS_B = 100   # assez pour que le BPTT CONVERGE (à 30 epochs rien ne bouge, cf. diag)


def _base_and_net():
    ckpt = torch.load(TEST_CF / "models" / "model_pretrain.pt", map_location="cpu")
    return torch_sd_to_flax(ckpt.get("model_state_dict", ckpt))


def _bptt_constant(base):
    """Chemin de prod EXACT du worker en vol : fit mode='constant' -> contexte BPTT
    de prod (build_bptt_context, use_forward_residual seul, MLP-params constants)."""
    X = _synth_X(CFG["online"]["num_samples"], seed=0)
    y = np.tile(MEAN_Y, (X.shape[0], 1)).astype(np.float32)
    res_params = F.fit_residual_ensemble(X, y, mode="constant")
    ctx = F.build_bptt_context(list(P.HOVER_GOAL), EPOCHS_B)
    return F.run_bptt(ctx, base, res_params), ctx["policy_net"]


def _bptt_exact_mass(base):
    """Référence physique : résidu EXACT de masse injecté dans la CONFIG du quad
    (proportionnel à la poussée). Construit son propre contexte (quad différent)."""
    from lotf.objects import Quadrotor
    from lotf.envs import HoveringStateEnv
    from lotf.envs.wrappers import MinMaxObservationWrapper, LogWrapper, VecEnv
    from lotf.algos import bptt
    import optax
    from flax.training.train_state import TrainState
    from lotf_jax_bridge import make_lotf_mlp

    quad = Quadrotor.from_name("crazyflie_quad", {
        "use_high_fidelity": False, "use_forward_residual": True,
        "exact_mass_residual": {"m_nominal": M_NOM, "m_real": M_REAL}})
    e = CFG["hovering_env"]
    env = HoveringStateEnv(max_steps_in_episode=int(CFG["bptt"]["max_sim_time"] / P.DT),
                           dt=P.DT, delay=P.DELAY, yaw_scale=e["yaw_scale"],
                           pitch_roll_scale=e["pitch_roll_scale"], velocity_std=e["velocity_std"],
                           omega_std=e["omega_std"], quad_obj=quad, reward_sharpness=e["reward_sharpness"],
                           action_penalty_weight=e["action_penalty_weight"], margin=e["margin"],
                           hover_target=list(P.HOVER_GOAL))
    env = MinMaxObservationWrapper(env)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    env = VecEnv(LogWrapper(env))
    net = make_lotf_mlp(od, ad, P.HOVERING_ACTION)
    tx = optax.chain(optax.clip_by_global_norm(CFG["bptt"]["grad_clip"]),
                     optax.adam(optax.cosine_decay_schedule(CFG["bptt"]["lr"], EPOCHS_B)))
    kb, kr = jax.random.split(jax.random.key(0))
    ies, io = env.reset(jax.random.split(kr, CFG["bptt"]["num_envs"]), None)
    ts = TrainState.create(apply_fn=net.apply, params=base, tx=tx)
    res = bptt.train(env, ies, io, ts, num_epochs=EPOCHS_B,
                     num_steps_per_epoch=env.max_steps_in_episode,
                     num_envs=CFG["bptt"]["num_envs"], res_model_params={}, key=kb)
    return res["runner_state"].train_state.params, net


def test_constant_residual_is_wrong_model_for_mass():
    target = list(P.HOVER_GOAL)
    base = _base_and_net()
    ft_const, net = _bptt_constant(base)
    ft_exact, _ = _bptt_exact_mass(base)

    off_before = _eval_offset(base, net, M_REAL, target)
    off_const = _eval_offset(ft_const, net, M_REAL, target)
    off_exact = _eval_offset(ft_exact, net, M_REAL, target)

    print(f"\n  [B] offset hover quad lourd ({M_REAL*1e3:.0f} g), {EPOCHS_B} epochs BPTT :")
    print(f"      base (non adaptée)        {off_before:+.3f} m")
    print(f"      résidu CONSTANT (prod)    {off_const:+.3f} m")
    print(f"      résidu EXACT masse        {off_exact:+.3f} m")

    # 1) le résidu EXACT (proportionnel à T) corrige bien -> la brique BPTT est saine
    assert abs(off_exact) < abs(off_before) - 0.02, \
        f"résidu EXACT n'a pas corrigé: {off_before:+.3f} -> {off_exact:+.3f} m (brique BPTT cassée ?)"
    # 2) le résidu CONSTANT corrige nettement MOINS (la racine du 'finetuning ne marche pas')
    assert abs(off_const) > abs(off_exact) + 0.02, \
        (f"résidu CONSTANT aussi bon qu'EXACT ({off_const:+.3f} vs {off_exact:+.3f}) — "
         f"le modèle synthétique ne reproduit plus la limitation attendue ?")
    print(f"  [B] CONFIRMÉ : un écart de masse exige un résidu PROPORTIONNEL À LA POUSSÉE ; "
          f"le résidu constant converge vers une mauvaise poussée de hover  ✓")


# ───────── C. mode mass_estimation : ∝ poussée, coeff estimé en ligne ─────────
def test_mass_estimation_reduces_offset():
    """Le mode de prod `mass_estimation` : (1) estime coeff = mean(y_z)/mean(T) ≈ COEFF
    exact ; (2) via le chemin de prod (build_bptt_context(res_mode=...) + run_bptt avec
    coeff porté par res_model_params) il corrige l'offset comme le résidu exact."""
    target = list(P.HOVER_GOAL)
    # données = offset de masse EXACT sur une plage de poussée (R≈I) -> coeff identifiable
    X = _synth_X(CFG["online"]["num_samples"], t_lo=0.30, t_hi=0.45, seed=0)
    y = np.zeros((X.shape[0], 3), np.float32)
    y[:, 2] = COEFF * X[:, 15]
    coeff = F.fit_residual_ensemble(X, y, mode="mass_estimation")    # scalaire jnp
    rel = abs(float(coeff) - COEFF) / abs(COEFF)
    assert rel < 0.1, f"coeff estimé faux: {float(coeff):.2f} vs exact {COEFF:.2f} (rel {rel:.0%})"

    base = _base_and_net()
    ctx = F.build_bptt_context(target, EPOCHS_B, res_mode="mass_estimation")
    ft = F.run_bptt(ctx, base, coeff)
    off_before = _eval_offset(base, ctx["policy_net"], M_REAL, target)
    off_after = _eval_offset(ft, ctx["policy_net"], M_REAL, target)
    print(f"\n  [C] mass_estimation : coeff={float(coeff):.2f} (exact {COEFF:.2f}) ; "
          f"offset {off_before:+.3f} -> {off_after:+.3f} m")
    assert abs(off_after) < abs(off_before) - 0.02, \
        f"mass_estimation n'a pas corrigé l'offset: {off_before:+.3f} -> {off_after:+.3f} m"
    print(f"  [C] mode mass_estimation CORRIGE l'offset (résidu ∝ poussée, coeff en ligne)  ✓")


if __name__ == "__main__":
    tests = [test_constant_params_are_truly_constant, test_fit_constant_mode_matches_mean,
             test_constant_residual_is_wrong_model_for_mass, test_mass_estimation_reduces_offset]
    fails = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            fails += 1; print(f"  ❌ {t.__name__}: {e}")
        except Exception as e:                                # noqa: BLE001
            import traceback; traceback.print_exc()
            fails += 1; print(f"  ❌ {t.__name__} (erreur): {e}")
    print(f"\n{'TOUS OK ✅' if fails == 0 else f'{fails} ÉCHEC(S) ❌'}")
    sys.exit(1 if fails else 0)
