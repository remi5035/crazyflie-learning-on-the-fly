"""Test DÉDIÉ AU RÉSIDU sur un vrai log — pendant réel de lotf_residual_check.py.

Compare, sur un log du VRAI drone, le résidu MESURÉ (MLP appris, pipeline) au résidu
EXACT CONSTANT qui ne compense QUE l'erreur de masse, puis regarde si le finetuning
est STABLE avec chacun.

Hypothèse : près du hover (R≈I, f_d≈const) le vrai résidu est ~un décalage de masse,
donc ~CONSTANT en monde. On en déduit une masse EFFECTIVE m_eff telle que
    mean(y_z) = (1/m_eff - 1/m_nom) · T_hover
et on construit le résidu exact analytique `R·[0,0,T(1/m_eff-1/m_nom)]`. Si le MLP
appris s'en écarte (surtout son gradient ∂a/∂T, non contraint par un hover statique),
le BPTT peut l'exploiter -> instable. On le mesure.

À lancer dans `.venv` (jax + lotf + torch) :
    cd test_cf
    ../.venv/bin/python tests/test_residual_real.py --log measurements/real/real_run_lotf.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import torch
import optax
from flax.training.train_state import TrainState

THIS = Path(__file__).resolve().parent
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())
TEST_CF = REPO / "test_cf"
for p in (TEST_CF / "core", TEST_CF / "RL-real", REPO):
    sys.path.insert(0, str(p))

import cf_params as P                                          # noqa: E402
from lotf.objects import Quadrotor                             # noqa: E402
from lotf.envs import HoveringStateEnv, rollout                # noqa: E402
from lotf.envs.wrappers import MinMaxObservationWrapper, LogWrapper, VecEnv  # noqa: E402
from lotf.algos import bptt                                    # noqa: E402
from lotf.utils.residual_dynamics import exact_mass_residual   # noqa: E402

import finetune_lotf_jax as F                                  # noqa: E402
from lotf_residual import build_residual_dataset              # noqa: E402
from lotf_jax_bridge import make_lotf_mlp, torch_sd_to_flax    # noqa: E402

M_NOM = P.MASS                                                 # masse que le modèle nominal suppose


# ── construction d'un contexte BPTT pour un quad/résidu donné (≈ build_bptt_context) ──
def build_ctx(quad_cfg, epochs, target):
    quad = Quadrotor.from_name("crazyflie_quad", quad_cfg)
    env = HoveringStateEnv(max_steps_in_episode=int(3.0 / P.DT), dt=P.DT, delay=P.DELAY,
                           yaw_scale=1.0, pitch_roll_scale=0.1, velocity_std=0.1, omega_std=0.1,
                           quad_obj=quad, reward_sharpness=3.0, action_penalty_weight=0.5,
                           margin=0.5, hover_target=list(target))
    env = MinMaxObservationWrapper(env)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    env = VecEnv(LogWrapper(env))
    net = make_lotf_mlp(obs_dim, act_dim, P.HOVERING_ACTION)
    tx = optax.chain(optax.clip_by_global_norm(0.5),
                     optax.adam(optax.cosine_decay_schedule(1e-3, epochs)))
    kb, kr = jax.random.split(jax.random.key(0))
    ies, io = env.reset(jax.random.split(kr, 10), None)
    return dict(env=env, net=net, tx=tx, epochs=epochs, kb=kb, ies=ies, io=io)


def run_bptt(ctx, base, res_params):
    ts = TrainState.create(apply_fn=ctx["net"].apply, params=base, tx=ctx["tx"])
    res = bptt.train(ctx["env"], ctx["ies"], ctx["io"], ts, num_epochs=ctx["epochs"],
                     num_steps_per_epoch=ctx["env"].max_steps_in_episode, num_envs=10,
                     res_model_params=res_params, key=ctx["kb"])
    return res["runner_state"].train_state.params


# ── éval dans un monde de RÉFÉRENCE (résidu exact de masse) ──
def eval_ref(params, net, m_eff, target, n=8):
    quad = Quadrotor.from_name("crazyflie_quad", {
        "use_high_fidelity": False, "use_forward_residual": True,
        "exact_mass_residual": {"m_nominal": M_NOM, "m_real": m_eff}})
    env = MinMaxObservationWrapper(HoveringStateEnv(
        max_steps_in_episode=251, dt=P.DT, delay=P.DELAY, quad_obj=quad,
        margin=0.5, hover_target=list(target)))

    def policy(obs, key):
        return net.apply(params, obs)
    tr = jax.vmap(rollout, in_axes=(None, 0, None, None))(
        env, jax.random.split(jax.random.key(0), n), policy, {})
    p = np.asarray(tr.state.quadrotor_state.p)
    term = np.asarray(tr.terminated); done = term | np.asarray(tr.truncated)
    fd = np.where(done.any(1), done.argmax(1), p.shape[1] - 1)
    err = np.mean([np.linalg.norm(p[i, max(0, k-50):k+1] - np.asarray(target), axis=1).mean()
                   for i, k in enumerate(fd)])
    z_off = np.mean([p[i, max(0, k-50):k+1, 2].mean() for i, k in enumerate(fd)]) - target[2]
    return float(z_off), float(err), int(term.any(1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="measurements/real/real_run_lotf.npz")
    ap.add_argument("--base", default="models/model_pretrain.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--window-sec", type=float, default=2.0)
    ap.add_argument("--target", type=float, nargs=3, default=list(P.HOVER_GOAL))
    args = ap.parse_args()
    target = args.target

    log_path = (TEST_CF / args.log) if not Path(args.log).is_absolute() else Path(args.log)
    log = {k: v for k, v in np.load(log_path).items()}
    X, y = build_residual_dataset(log, window_sec=None if args.window_sec <= 0 else args.window_sec)
    T = X[:, 15]; R = X[:, 3:12].reshape(-1, 3, 3)
    mean_y = y.mean(0)

    # masse EFFECTIVE qui explique le résidu z mesuré (near hover, R≈I)
    coeff = mean_y[2] / T.mean()                 # = 1/m_eff - 1/m_nom
    m_eff = 1.0 / (1.0 / M_NOM + coeff)
    print(f"[diag] log={log_path.name}  samples={len(X)}  T_moy={T.mean():.3f} N "
          f"(std={T.std():.4f})")
    print(f"[diag] résidu z moyen mesuré = {mean_y[2]:.3f} m/s²  ->  masse EFFECTIVE "
          f"m_eff={m_eff*1000:.1f} g  (nominal {M_NOM*1000:.0f} g)")
    print(f"       ∂a_z/∂T exact (masse) = {coeff:.2f}  (1/kg)")

    # ── 1. RÉSIDU MESURÉ (MLP) vs EXACT-CONSTANT (masse) : ajustement aux données ──
    res_mlp = F.fit_residual_ensemble(X, y)
    _, _, predict_fn = F._vec_funcs()
    pred_mlp = np.asarray(predict_fn(res_mlp, jnp.asarray(X))).mean(0)        # (n,3)
    pred_exact = np.stack([np.asarray(exact_mass_residual(jnp.asarray(X[i]), M_NOM, m_eff))
                           for i in range(len(X))])                            # (n,3)
    mse_mlp = float(np.mean((pred_mlp - y) ** 2))
    mse_exact = float(np.mean((pred_exact - y) ** 2))
    mse_const = float(np.mean((mean_y[None, :] - y) ** 2))
    print("\n── 1. AJUSTEMENT AUX DONNÉES (MSE résidu vs y mesuré) ──")
    print(f"  MLP appris        : {mse_mlp:.4f}")
    print(f"  exact masse (m_eff): {mse_exact:.4f}")
    print(f"  constante (moyenne): {mse_const:.4f}")
    print(f"  -> {'le MLP fait à peine mieux que la constante : le résidu EST ~un offset de masse'
            if mse_mlp > 0.5*mse_const else 'le MLP capte une structure au-delà de la constante'}")

    # ── 2. GRADIENT ∂a_z/∂T : MLP (potentiellement parasite) vs exact (physique) ──
    base_x = X[np.argmin(np.abs(T - T.mean()))].copy(); base_x[12:15] = 0.0
    def mlp_dadT(dT=0.02):
        xp = base_x.copy(); xp[15] += dT; xm = base_x.copy(); xm[15] -= dT
        pp = np.asarray(predict_fn(res_mlp, jnp.asarray(xp[None]))).mean(0)[0, 2]
        pm = np.asarray(predict_fn(res_mlp, jnp.asarray(xm[None]))).mean(0)[0, 2]
        return float((pp - pm) / (2 * dT))
    print("\n── 2. GRADIENT ∂a_z/∂T (ce que le BPTT exploite) ──")
    print(f"  MLP appris : {mlp_dadT():+.2f}   |  exact masse : {coeff:+.2f}   "
          f"(données T std={T.std():.4f} N -> {'plage OK' if T.std()>0.01 else '⚠ NON CONTRAINT'})")

    # ── 3. STABILITÉ DU FINETUNE avec chaque résidu (éval dans le monde exact de réf.) ──
    ckpt = torch.load((TEST_CF / args.base) if not Path(args.base).is_absolute() else Path(args.base),
                      map_location="cpu")
    base = torch_sd_to_flax(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)

    ctx_mlp = build_ctx({"use_high_fidelity": False, "use_forward_residual": True}, args.epochs, target)
    ctx_const = build_ctx({"use_high_fidelity": False, "use_forward_residual": True,
                           "constant_residual": [float(v) for v in mean_y]}, args.epochs, target)
    ctx_exact = build_ctx({"use_high_fidelity": False, "use_forward_residual": True,
                           "exact_mass_residual": {"m_nominal": M_NOM, "m_real": m_eff}}, args.epochs, target)

    ft_mlp = run_bptt(ctx_mlp, base, res_mlp)
    ft_const = run_bptt(ctx_const, base, {})
    ft_exact = run_bptt(ctx_exact, base, {})

    print("\n── 3. STABILITÉ DU FINETUNE (éval dans le monde EXACT de réf., m_eff) ──")
    print(f"  {'résidu utilisé pour finetune':32s} {'offset z':>9s} {'err pos':>9s} {'crashs/8':>9s}")
    for name, params in [("base (non adaptée)", base), ("MLP appris (mesuré)", ft_mlp),
                         ("constante (moyenne)", ft_const), ("exact masse (m_eff)", ft_exact)]:
        zo, er, nc = eval_ref(params, ctx_mlp["net"], m_eff, target)
        print(f"  {name:32s} {zo:+9.3f} {er:9.3f} {nc:9d}")
    print("\n  Lecture : si 'MLP appris' a PLUS de crashs / err que 'constante'/'exact', son")
    print("  gradient ∂a/∂T parasite (hover statique) déstabilise -> préférer le résidu CONSTANT.")


if __name__ == "__main__":
    main()
