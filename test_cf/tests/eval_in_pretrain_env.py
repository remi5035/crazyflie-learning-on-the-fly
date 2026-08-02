"""Évalue une politique pretrain DANS SA PROPRE DYNAMIQUE D'ENTRAÎNEMENT.

But : trancher si res27 est instable PARCE QUE le pretrain a échoué (mauvais optimum)
ou seulement au DÉPLOIEMENT. On rejoue chaque .pt dans l'env low-fi exact du pretrain :
  - base27 : crazyflie 27 g, sans résidu
  - mass33 : crazyflie 33 g, sans résidu
  - res27  : crazyflie 27 g + résidu de masse EXACT émulant 33 g (a_res=R·[0,0,f_d·(1/0.033-1/0.027)])
Si res27 hover ICI proprement -> le biais EST appris, le crash Genesis = déploiement.
Si res27 hover mal ICI -> l'entraînement lui-même a raté.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import jax

THIS = Path(__file__).resolve().parent
REPO = next(p for p in THIS.parents if (p / "lotf").is_dir())
TEST_CF = REPO / "test_cf"
for p in (TEST_CF / "core", TEST_CF / "RL-real", REPO):
    sys.path.insert(0, str(p))

import cf_params as P                                          # noqa: E402
from lotf.envs import HoveringStateEnv, rollout                # noqa: E402
from lotf.envs.wrappers import MinMaxObservationWrapper        # noqa: E402
from lotf_jax_bridge import make_lotf_mlp, torch_sd_to_flax    # noqa: E402
from pretrain_lotf_jax import build_quad, dummy_residual_params  # noqa: E402


def eval_policy(pt_path, mass, residual_m_real, target, steps=500):
    """Départ HOVER EXACT (scales reset à 0, margin=1.5 -> p=goal), pas d'auto-reset.
    Isole le biais de hover et la dérive xy de l'instabilité de récupération."""
    quad, _ = build_quad(mass, residual_m_real)
    env = MinMaxObservationWrapper(HoveringStateEnv(
        max_steps_in_episode=steps, dt=P.DT, delay=P.DELAY, quad_obj=quad,
        yaw_scale=0.0, pitch_roll_scale=0.0, velocity_std=0.0, omega_std=0.0,
        margin=1.5, hover_target=list(target)))
    hover = np.asarray(env.hovering_action)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    net = make_lotf_mlp(obs_dim, act_dim, hover)
    ckpt = torch.load(pt_path, map_location="cpu")
    params = torch_sd_to_flax(ckpt["model_state_dict"])

    def policy(obs, key):
        return net.apply(params, obs)
    tr = rollout(env, jax.random.key(0), policy, dummy_residual_params(),
                 real_step=False, num_steps=steps)
    p = np.asarray(tr.state.quadrotor_state.p)              # (T,3)
    term = np.asarray(tr.terminated)
    left = bool(term.any())
    kleft = int(term.argmax()) if left else steps - 1
    last = p[max(0, kleft-100):kleft+1]                    # ~2 dernières s avant sortie/fin
    z_off = float(last[:, 2].mean() - target[2])
    xy_drift = float(np.linalg.norm(p[:kleft+1, :2] - np.asarray(target)[:2], axis=1).max())
    z_min = float(p[:kleft+1, 2].min())
    return float(hover[0]), z_off, xy_drift, z_min, left, kleft


def main():
    target = list(P.HOVER_GOAL)
    cases = [
        ("base27 (27g, sans résidu)",      "models/model_pretrain.pt", 0.027, None),
        ("mass33 (33g, sans résidu)",      "models/model_33g.pt",      0.033, None),
        ("res27  (27g + résidu->33g)",     "models/model_res33.pt",    0.027, 0.033),
    ]
    print(f"DÉPART HOVER EXACT, 500 pas (10 s), goal={target}  — chacune dans SON env d'entraînement\n")
    print(f"  {'politique':30s} {'hover_bias':>10s} {'offset z':>9s} {'xy drift':>9s} {'z min':>8s} {'sortie boîte':>13s}")
    for name, rel, mass, mres in cases:
        hb, z, xy, zmin, left, k = eval_policy(str(TEST_CF / rel), mass, mres, target)
        out = f"pas {k} ({k*P.DT:.1f}s)" if left else "non"
        print(f"  {name:30s} {hb:9.3f}N {z:+8.3f}m {xy:8.3f}m {zmin:7.2f}m {out:>13s}")


if __name__ == "__main__":
    main()
