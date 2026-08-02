"""Finetune LOTF de la pipeline Genesis en réutilisant lotf-JAX — IDENTIQUE à
la pipeline Gazebo (`scripts/finetune_lotf.py`), mais hors ROS : consomme un
rollout Genesis `_lotf.npz` et ressort une politique torch `.pt` revolable dans
Genesis. Architecturalement, la finetune reste un PROCESSUS SÉPARÉ (comme le nœud
ROS Gazebo), donc le venv Genesis (torch) n'a pas besoin de JAX.

Étapes (toutes en lotf, sauf build_residual_dataset qui partage la formule) :
  1. dataset résiduel y=a_mes-a_nom            (lotf_residual.build_residual_dataset)
  2. fit ENSEMBLE de 3 résidus                  (lotf.utils.residual_dynamics.create_vec_funcs)
  3. BPTT court contre HoveringStateEnv(crazyflie_quad, use_forward_residual) (lotf.algos.bptt)
  4. politique flax -> .pt torch                (lotf_jax_bridge)

Usage (depuis .venv qui contient jax+lotf+torch) :
  python core/finetune_lotf_jax.py --base models/model_pretrain.pt --log measurements/genesis/rollout.npz \
      --out models/model_ft_jax.pt --target 0 0 0.5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import jax, jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

THIS = Path(__file__).resolve().parent                  # core/
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())   # racine du dépôt
TEST_CF = REPO / "test_cf"
sys.path.insert(0, str(TEST_CF / "RL-real"))
sys.path.insert(0, str(REPO))

import cf_params as P                                   # noqa: E402
from lotf_config import CFG                              # noqa: E402  (source unique des hyperparams)

# Cache de compilation XLA persistant (à régler AVANT toute compilation, donc ici) :
# garde sur disque les binaires du BPTT -> pas de recompilation aux lancements suivants
# (warmup ~15 s -> ~6 s). Voir lotf_config.yaml section `jax`.
_cc = CFG.get("jax", {}).get("compilation_cache_dir") if hasattr(CFG, "get") else None
if _cc:
    _cc = _cc if Path(_cc).is_absolute() else str(TEST_CF / _cc)
    jax.config.update("jax_compilation_cache_dir", _cc)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

from lotf_residual import build_residual_dataset        # noqa: E402  (formule partagée)
from lotf.objects import Quadrotor                      # noqa: E402
from lotf.envs import HoveringStateEnv                  # noqa: E402
from lotf.envs.wrappers import MinMaxObservationWrapper, LogWrapper, VecEnv  # noqa: E402
from lotf.algos import bptt                             # noqa: E402
from lotf.utils.residual_dynamics import create_vec_funcs  # noqa: E402
from lotf_jax_bridge import make_lotf_mlp, torch_sd_to_flax, flax_to_torch_sd  # noqa: E402

# Hyperparams = lotf_config.yaml (aligné scripts/finetune_lotf.py / papier Sec. III-B)
_R, _B = CFG["residual_fit"], CFG["bptt"]
FT_NUM_MODELS, FT_RES_LR, FT_RES_LAMBDA = _R["num_models"], _R["lr"], _R["lambda_reg"]
FT_RES_EPOCHS, FT_RES_EVAL = _R["epochs"], _R["eval_every"]
FT_NUM_ENVS, FT_MAX_SIM_TIME, FT_LR, FT_SEED = _B["num_envs"], _B["max_sim_time"], _B["lr"], _B["seed"]
FT_DEFAULT_EPOCHS, QUAD, FT_GRAD_CLIP = _B["epochs_per_step"], _B["quad"], _B["grad_clip"]


# Les fonctions vmappées du fit résiduel sont créées UNE fois puis réutilisées :
# `train` (lotf) est @jit avec sample-count traceable -> recréer le wrapper à
# chaque appel reste correct mais on évite toute re-trace en le mémorisant.
_VEC_FUNCS = None


def _vec_funcs():
    global _VEC_FUNCS
    if _VEC_FUNCS is None:
        _VEC_FUNCS = create_vec_funcs()
    return _VEC_FUNCS


def _constant_residual_params(mean_y):
    """Params d'ensemble MLP produisant une SORTIE CONSTANTE = mean_y pour tout x.

    On part de la structure initialisée (mêmes clés/shapes que le MLP fitté, donc
    le cache JIT du BPTT reste chaud), puis : tous les NOYAUX -> 0, tous les biais
    CACHÉS -> 0, et le biais de SORTIE (dim 3) -> mean_y. Avec entrée nulle à chaque
    couche, tanh(0)=0 se propage et la dernière couche linéaire renvoie mean_y :
        net(x) = mean_y   ∀x   ->   ∂a/∂x ≡ 0 (aucun gradient parasite, dont ∂a/∂T).

    C'est l'équivalent par-params de lotf.get_constant_residual_apply_fn : pour un
    décalage de MASSE le résidu est ~un offset constant (cf. tests/test_residual_real),
    donc inutile (et nuisible) d'exciter la poussée pour contraindre ∂a/∂T du MLP."""
    init_fn, _, _ = _vec_funcs()
    _, states = init_fn(FT_RES_LR, jnp.arange(FT_NUM_MODELS, dtype=jnp.int32))
    mean_y = jnp.asarray(mean_y, dtype=jnp.float32)

    def fix(p):
        # biais de sortie de l'ensemble : (num_models, 3) -> ndim 2, dernière dim 3.
        # (les noyaux de sortie sont (num_models, 128, 3) -> ndim 3 ; biais cachés
        #  (num_models, 128) -> dernière dim 128 ; tout le reste passe à 0.)
        if p.ndim == 2 and p.shape[-1] == mean_y.shape[0]:
            return jnp.broadcast_to(mean_y, p.shape)
        return jnp.zeros_like(p)

    return jax.tree_util.tree_map(fix, states.params)


def fit_residual_ensemble(X, y, mode="mlp"):
    """Fit de l'ensemble résiduel. N.B. la shape de X (nb d'échantillons) est
    traceable par le JIT : appeler toujours avec le MÊME N garde le cache chaud
    (cf. finetune_lotf_worker, qui ré-échantillonne à un N fixe).

    mode :
      'mlp'             -> résidu MLP appris (capte une dynamique qui VARIE dans le temps,
                           mais ∂a/∂T non contraint en hover statique -> peut nécessiter
                           l'excitation de poussée pour rester stable au BPTT).
      'constant'        -> résidu CONSTANT = moyenne du résidu mesuré sur la fenêtre. ∂a/∂x ≡ 0,
                           stable sans excitation MAIS ne matche la dynamique qu'à LA poussée
                           mesurée -> sur un écart de MASSE le BPTT vise une mauvaise poussée de
                           hover et l'offset n'est PAS corrigé (cf. tests/test_finetune_constant).
      'mass_estimation' -> résidu de MASSE estimé, PROPORTIONNEL À LA POUSSÉE :
                           a_res = coeff·f_d·R_z avec coeff = mean(y_z)/mean(T) = 1/m_eff - 1/m_nom.
                           Scale avec T -> matche le quad lourd au nouvel équilibre -> CORRIGE
                           l'offset (≈ résidu exact, mais m_eff estimé en ligne). Renvoie le
                           SCALAIRE coeff (porté par res_model_params -> trace JIT fixe)."""
    init_fn, train_fn, predict_fn = _vec_funcs()
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    mean_y = np.asarray(jnp.mean(yj, axis=0))

    if mode == "constant":
        print(f"[ft-jax] résidu CONSTANT (offset masse) = {mean_y.round(3)} m/s²  "
              f"(∂a/∂x ≡ 0 -> stable sans excitation)")
        return _constant_residual_params(mean_y)

    if mode == "mass_estimation":
        T_mean = float(np.mean(np.asarray(Xj[:, 15])))
        coeff = float(mean_y[2] / T_mean) if abs(T_mean) > 1e-8 else 0.0
        m_eff = 1.0 / (1.0 / P.MASS + coeff) if (1.0 / P.MASS + coeff) > 1e-6 else float("inf")
        print(f"[ft-jax] résidu MASSE ∝ poussée : coeff={coeff:.3f} (1/kg) -> m_eff={m_eff*1e3:.1f} g "
              f"(nominal {P.MASS*1e3:.0f} g) ; a_res = coeff·f_d·R_z")
        return jnp.asarray(coeff, dtype=jnp.float32)

    _, states = init_fn(FT_RES_LR, jnp.arange(FT_NUM_MODELS, dtype=jnp.int32))
    states = train_fn(states, Xj, yj, FT_RES_LAMBDA, FT_RES_EPOCHS, FT_RES_EVAL)

    # Diagnostic : moyenne de la prédiction de l'ensemble (résidu APPRIS) sur le
    # dataset, à comparer à la moyenne de la cible y (= a_mes - a_nom). Sur un pur
    # décalage de masse, le terme z domine (perte de poussée). Cf. lotf_residual_check.py.
    mean_mlp = np.asarray(jnp.mean(predict_fn(states.params, Xj), axis=(0, 1)))
    print(f"[ft-jax] moyenne résidu appris (MLP) = {mean_mlp.round(3)} m/s²  "
          f"(cible données y = {mean_y.round(3)})")
    return states.params


def build_bptt_context(target, epochs, res_mode="mlp"):
    """Construit env + réseau + état initial du BPTT, parties INVARIANTES d'une
    finetune à l'autre. `bptt.train` JIT-compile avec env/epochs/num_envs en
    static_argnums : réutiliser ce contexte (même objet env, mêmes epochs) fait
    que les finetunes suivantes touchent le cache compilé au lieu de recompiler.

    `res_mode` fixe l'apply-fn du résidu DANS le quad (donc la trace JIT) :
      'mlp'/'constant'  -> MLP d'ensemble (res_model_params = poids MLP).
      'mass_estimation' -> a_res = coeff·f_d·R_z, coeff porté par res_model_params
                           (trace fixe ; cf. get_mass_residual_apply_fn).
    """
    e = CFG["hovering_env"]
    quad_cfg = {"use_high_fidelity": False, "use_forward_residual": True}
    if res_mode == "mass_estimation":
        quad_cfg["mass_residual"] = True
    quad = Quadrotor.from_name(QUAD, quad_cfg)
    env = HoveringStateEnv(
        max_steps_in_episode=int(FT_MAX_SIM_TIME / P.DT), dt=P.DT, delay=P.DELAY,
        yaw_scale=e["yaw_scale"], pitch_roll_scale=e["pitch_roll_scale"],
        velocity_std=e["velocity_std"], omega_std=e["omega_std"],
        quad_obj=quad, reward_sharpness=e["reward_sharpness"],
        action_penalty_weight=e["action_penalty_weight"],
        margin=e["margin"], hover_target=list(target))
    env = MinMaxObservationWrapper(env)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    env = VecEnv(LogWrapper(env))

    policy_net = make_lotf_mlp(obs_dim, action_dim, np.asarray(env.hovering_action))
    scheduler = optax.cosine_decay_schedule(FT_LR, epochs)
    # grad-clip global comme le pretrain : borne l'exploding-gradient du BPTT à travers
    # un quad instable (sinon un pic éjecte la policy -> divergence aléatoire du 1er step).
    tx = optax.chain(optax.clip_by_global_norm(FT_GRAD_CLIP), optax.adam(scheduler))
    key_bptt, key_reset = jax.random.split(jax.random.key(FT_SEED))
    init_env_state, init_obs = env.reset(jax.random.split(key_reset, FT_NUM_ENVS), None)
    return {
        "env": env, "policy_net": policy_net, "tx": tx,
        "epochs": epochs, "key_bptt": key_bptt,
        "init_env_state": init_env_state, "init_obs": init_obs,
        "obs_dim": obs_dim, "action_dim": action_dim,
    }


def run_bptt(ctx, base_flax_params, residual_params):
    """Lance le BPTT à partir d'un contexte `build_bptt_context` (réutilisable)."""
    env = ctx["env"]
    train_state = TrainState.create(apply_fn=ctx["policy_net"].apply,
                                    params=base_flax_params, tx=ctx["tx"])
    res = bptt.train(env, ctx["init_env_state"], ctx["init_obs"], train_state,
                     num_epochs=ctx["epochs"], num_steps_per_epoch=env.max_steps_in_episode,
                     num_envs=FT_NUM_ENVS, res_model_params=residual_params, key=ctx["key_bptt"])
    return res["runner_state"].train_state.params


def bptt_finetune(base_flax_params, residual_params, target, epochs, res_mode="mlp"):
    """One-shot (CLI) : construit le contexte puis lance le BPTT."""
    return run_bptt(build_bptt_context(target, epochs, res_mode=res_mode),
                    base_flax_params, residual_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help=".pt torch de la politique de base")
    ap.add_argument("--log", default=None, help="rollout Genesis _lotf.npz (requis sauf si --sim-mass)")
    ap.add_argument("--out", default="models/model_ft_jax.pt")
    ap.add_argument("--epochs", type=int, default=FT_DEFAULT_EPOCHS, help="epochs BPTT")
    ap.add_argument("--target", type=float, nargs=3, default=list(P.HOVER_GOAL))
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--residual-mode", choices=["mlp", "constant", "mass_estimation"], default="mlp",
                    help="mode de fit du résidu depuis --log (défaut mlp, comme avant)")
    ap.add_argument("--sim-mass", type=float, default=None,
                    help="résidu de masse THÉORIQUE (sans log) : a_res=coeff·f_d·R_z avec "
                         "coeff=1/sim_mass-1/27g, simulant un quad de cette masse [kg]. "
                         "Force mass_estimation et rend --log inutile. Ex: --sim-mass 0.033")
    args = ap.parse_args()

    if args.sim_mass is not None:
        # Résidu analytique exact : ce qu'un log parfait de pur décalage de masse fitterait
        # en mass_estimation (cf. fit_residual_ensemble). Pas besoin de vol.
        res_mode = "mass_estimation"
        coeff = 1.0 / args.sim_mass - 1.0 / P.MASS
        m_eff = 1.0 / (1.0 / P.MASS + coeff)
        print(f"[ft-jax] résidu de masse THÉORIQUE : sim_mass={args.sim_mass*1e3:.1f} g -> "
              f"coeff={coeff:.3f} (1/kg), m_eff={m_eff*1e3:.1f} g ; a_res=coeff·f_d·R_z (pas de log)")
        res_params = jnp.asarray(coeff, dtype=jnp.float32)
    else:
        if args.log is None:
            ap.error("--log est requis quand --sim-mass n'est pas fourni")
        res_mode = args.residual_mode
        log = {k: v for k, v in np.load(TEST_CF / args.log if not Path(args.log).is_absolute()
                                       else Path(args.log)).items()}
        win = None if args.window_sec <= 0 else args.window_sec
        X, y = build_residual_dataset(log, window_sec=win)
        print(f"[ft-jax] résidu ({res_mode}) : {X.shape[0]} samples, |y|_median="
              f"{np.median(np.linalg.norm(y, axis=1)):.3f} m/s² -> ensemble {FT_NUM_MODELS} nets")
        res_params = fit_residual_ensemble(X, y, mode=res_mode)

    ckpt = torch.load(TEST_CF / args.base if not Path(args.base).is_absolute() else Path(args.base),
                      map_location="cpu")
    sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    base_flax = torch_sd_to_flax(sd)

    print(f"[ft-jax] BPTT lotf : HoveringStateEnv({QUAD}) + bptt.train, {args.epochs} epochs...")
    t0 = time.time()
    new_flax = bptt_finetune(base_flax, res_params, args.target, args.epochs, res_mode=res_mode)
    print(f"[ft-jax] BPTT fait en {time.time()-t0:.1f}s")

    new_sd = flax_to_torch_sd(new_flax, P.HOVERING_ACTION)
    out = TEST_CF / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': new_sd}, out)
    print(f"[ft-jax] politique torch -> {out}  (revolable dans Genesis)")


if __name__ == "__main__":
    main()
