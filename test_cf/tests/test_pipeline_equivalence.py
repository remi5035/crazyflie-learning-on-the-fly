"""Test d'équivalence : le pipeline résiduel+BPTT de `test_cf/RL-real` reproduit
la démonstration de `examples/residual_dynamics/learned_residual_offset_demo.ipynb`.

On ne touche PAS à Genesis : on utilise le simulateur différentiable maison
`CrazyflieDiffSim` à la fois comme « quad lourd réel » (masse M_REAL) et comme
modèle nominal augmenté du résidu (masse M_NOM), exactement comme le notebook
utilise le `Quadrotor` léger/lourd de `lotf`.

Déroulé (miroir des cellules du notebook) :
  0. pré-entraîne une politique de hover nominale (BPTT sans résidu).
  A. vole le quad LOURD sous la politique nominale -> dataset (build_residual_dataset)
     -> fit_residual -> compare au résidu analytique EXACT R·[0,0,T(1/m_real-1/m_nom)].
  B. finetune la politique sur (i) résidu EXACT, (ii) résidu APPRIS -> évalue sur le
     quad lourd -> l'offset en z doit être corrigé dans les deux cas comme la baseline.

Usage :
  python test_pipeline_equivalence.py            # pipeline tel quel (residual .detach())
  python test_pipeline_equivalence.py --fix-detach  # version corrigée (grad à travers le résidu)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = Path(__file__).resolve().parent              # tests/
REPO = next(p for p in THIS_DIR.parents if (p / "lotf").is_dir())   # racine du dépôt
REAL_DIR = REPO / "test_cf" / "RL-real"
sys.path.insert(0, str(REAL_DIR))

import cf_params as P                                   # noqa: E402
import lotf_sim                                         # noqa: E402
from lotf_sim import CrazyflieDiffSim                   # noqa: E402
from lotf_residual import build_residual_dataset, fit_residual  # noqa: E402
from RL_policy import Actor                             # noqa: E402
from finetune_lotf import bptt_finetune                 # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

M_NOM = P.MASS            # 0.027 kg (Crazyflie nominal)
M_REAL = 0.045           # kg — quad « lourd » (analogue 0.192 -> 0.5 du notebook)
DEVICE = "cpu"
GOAL_Z = float(P.HOVER_GOAL[2])


# ───────────────────────── helpers ─────────────────────────
def rollout_log(sim: CrazyflieDiffSim, policy, horizon, mass):
    """Vole `sim` (masse forcée à `mass`) sous `policy`, renvoie un log par env
    au format finetune_lotf (clés t,p,R,v,T_N,omega) + le z final moyen."""
    old_mass = lotf_sim.MASS
    lotf_sim.MASS = mass                 # « vrai » quad : on change sa masse
    try:
        obs = sim.reset()
        n = sim.num_envs
        rec = {k: [] for k in ('t', 'p', 'R', 'v', 'T_N', 'omega')}
        with torch.no_grad():
            for i in range(horizon):
                a = policy(obs)
                a = torch.maximum(sim.action_low, torch.minimum(sim.action_high, a))
                rec['t'].append(np.full(n, i * P.DT))
                rec['p'].append(sim.p_pos.cpu().numpy().copy())
                rec['R'].append(sim.R.reshape(n, 9).cpu().numpy().copy())
                rec['v'].append(sim.v.cpu().numpy().copy())
                rec['T_N'].append(a[:, 0].cpu().numpy().copy())
                rec['omega'].append(a[:, 1:].cpu().numpy().copy())
                obs, _ = sim.step(a)
    finally:
        lotf_sim.MASS = old_mass
    # -> array (horizon, n, ...) puis on renvoie un log par env
    rec = {k: np.stack(v) for k, v in rec.items()}
    logs = []
    for e in range(n):
        logs.append({
            't': rec['t'][:, e], 'p': rec['p'][:, e], 'R': rec['R'][:, e],
            'v': rec['v'][:, e], 'T_N': rec['T_N'][:, e], 'omega': rec['omega'][:, e],
        })
    z_final = rec['p'][-50:, :, 2].mean()
    return logs, z_final


def exact_mass_residual(X):
    """Résidu analytique exact de décalage de masse : R·[0,0,T(1/m_real-1/m_nom)].
    X = [p(3), R_flat(9), v(3), T_N(1), omega(3)]  (19-d)."""
    R = X[:, 3:12].reshape(-1, 3, 3)
    T = X[:, 15]
    body = np.zeros((X.shape[0], 3), np.float32)
    body[:, 2] = T * (1.0 / M_REAL - 1.0 / M_NOM)
    return np.einsum("nij,nj->ni", R, body)


class ExactResidual(nn.Module):
    """Module résidu renvoyant le résidu analytique exact (pour le finetune (i))."""
    def forward(self, feats):
        R = feats[:, 3:12].reshape(-1, 3, 3)
        T = feats[:, 15]
        body = torch.zeros((feats.shape[0], 3), dtype=feats.dtype, device=feats.device)
        body[:, 2] = T * (1.0 / M_REAL - 1.0 / M_NOM)
        return torch.einsum("nij,nj->ni", R, body)


# ───────────────────── 0. politique nominale ─────────────────────
def pretrain_nominal(epochs=200, num_envs=64, horizon=100):
    print("\n[0] pré-entraînement politique nominale (BPTT, quad nominal, sans résidu)")
    sim = CrazyflieDiffSim(num_envs=num_envs, residual_model=None, device=DEVICE)
    policy = Actor().to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_sd = -1e9, None
    import copy
    for ep in range(epochs):
        obs = sim.reset()
        total = torch.zeros(num_envs, device=DEVICE)
        for _ in range(horizon):
            a = policy(obs)
            obs, r = sim.step(a)
            total = total + r
        loss = -total.mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step(); sched.step()
        if total.mean().item() > best:
            best = total.mean().item(); best_sd = copy.deepcopy(policy.state_dict())
        if ep % 40 == 0 or ep == epochs - 1:
            print(f"    ep={ep:3d}  return={total.mean().item():9.2f}")
    policy.load_state_dict(best_sd)
    return policy


# ────────────────────── A. résidu appris vs exact ──────────────────────
def test_A_residual(policy):
    print("\n[A] dataset (quad LOURD sous politique nominale) -> fit résidu -> vs exact")
    sim = CrazyflieDiffSim(num_envs=16, residual_model=None, device=DEVICE)
    logs, z_heavy = rollout_log(sim, policy, horizon=150, mass=M_REAL)
    print(f"    z final quad lourd sous politique nominale = {z_heavy:.3f} m "
          f"(cible {GOAL_Z}) -> offset {z_heavy - GOAL_Z:+.3f} m  [doit sagger]")

    Xs, ys = [], []
    for lg in logs:
        X, y = build_residual_dataset(lg, window_sec=None)
        if len(X):
            Xs.append(X); ys.append(y)
    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(f"    dataset : X{X.shape} y{y.shape}")
    print(f"    biais moyen y         = {y.mean(0).round(3)} m/s²  (gros terme z = perte de portance)")
    print(f"    exact moyen attendu   = {exact_mass_residual(X).mean(0).round(3)} m/s²")

    model = fit_residual(X, y, epochs=300, device=DEVICE, verbose=False)
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).cpu().numpy()
    exact = exact_mass_residual(X)

    print(f"    moyenne modèle        = {pred.mean(0).round(3)}")
    print(f"    moyenne exact         = {exact.mean(0).round(3)}")
    for ax, nm in enumerate("xyz"):
        rmse = np.sqrt(np.mean((pred[:, ax] - exact[:, ax]) ** 2))
        denom = exact[:, ax].std()
        corr = np.corrcoef(pred[:, ax], exact[:, ax])[0, 1] if denom > 1e-6 else float('nan')
        print(f"      axe {nm}: RMSE(modèle vs exact)={rmse:6.3f}  corr={corr:+.3f}")
    rmse_g = np.sqrt(np.mean((pred - exact) ** 2))
    print(f"    RMSE global modèle vs exact = {rmse_g:.3f} m/s²  "
          f"(|exact| médian = {np.median(np.linalg.norm(exact, axis=1)):.3f})")
    return model


# ─────────────── B. finetune exact vs appris -> offset z ───────────────
def test_B_finetune(policy_nominal, learned_residual, fix_detach):
    print("\n[B] finetune BPTT (résidu EXACT vs APPRIS) puis évaluation sur quad LOURD")
    import copy

    def finetune(res_model):
        pol = Actor().to(DEVICE)
        pol.load_state_dict(copy.deepcopy(policy_nominal.state_dict()))
        return bptt_finetune(pol, res_model, epochs=120, num_envs=64, horizon=100,
                             lr=5e-4, grad_clip=0.5, device=DEVICE)

    # mute les prints internes de bptt_finetune
    import builtins, contextlib, io
    def quiet_ft(res_model):
        with contextlib.redirect_stdout(io.StringIO()):
            return finetune(res_model)

    pol_exact = quiet_ft(ExactResidual())
    pol_learned = quiet_ft(learned_residual)

    # évaluation sur le quad lourd
    sim = CrazyflieDiffSim(num_envs=32, residual_model=None, device=DEVICE)
    _, z_base = rollout_log(sim, policy_nominal, horizon=150, mass=M_REAL)
    _, z_exact = rollout_log(sim, pol_exact, horizon=150, mass=M_REAL)
    _, z_learn = rollout_log(sim, pol_learned, horizon=150, mass=M_REAL)

    print(f"    {'baseline (nominale @ lourd)':35s} z_final={z_base:.3f}  offset {z_base-GOAL_Z:+.3f} m")
    print(f"    {'finetune résidu EXACT  @ lourd':35s} z_final={z_exact:.3f}  offset {z_exact-GOAL_Z:+.3f} m")
    print(f"    {'finetune résidu APPRIS @ lourd':35s} z_final={z_learn:.3f}  offset {z_learn-GOAL_Z:+.3f} m")
    return z_base, z_exact, z_learn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-detach", action="store_true",
                    help="Retire le .detach() du résidu dans lotf_sim.step (grad à travers le résidu).")
    args = ap.parse_args()

    if args.fix_detach:
        _patch_remove_detach()
        print(">>> mode CORRIGÉ : gradient propagé à travers le résidu (pas de .detach())")
    else:
        print(">>> mode TEL QUEL : résidu .detach() dans lotf_sim.step")

    print(f"M_NOM={M_NOM} kg   M_REAL={M_REAL} kg   "
          f"résidu z attendu @hover ≈ {P.T_HOVER*(1/M_REAL-1/M_NOM):+.2f} m/s²")

    policy_nominal = pretrain_nominal()
    learned = test_A_residual(policy_nominal)
    test_B_finetune(policy_nominal, learned, args.fix_detach)


def _patch_remove_detach():
    """Réécrit CrazyflieDiffSim.step pour NE PAS détacher le résidu (version correcte)."""
    import torch as _t

    def step(self, action):
        action = _t.maximum(self.action_low, _t.minimum(self.action_high, action))
        T_N = action[:, 0]; omega_cmd = action[:, 1:]
        self.omega = self.omega + self.dt * (omega_cmd - self.omega) / lotf_sim.TAU_W
        thrust_body = _t.zeros_like(self.v); thrust_body[:, 2] = T_N / lotf_sim.MASS
        a = self.g_vec + _t.einsum("nij,nj->ni", self.R, thrust_body)
        if self.residual_model is not None:
            a = a + self.residual_model(self._residual_features(T_N))   # <-- pas de .detach()
        self.v = self.v + self.dt * a
        self.p_pos = self.p_pos + self.dt * self.v
        self.R = lotf_sim._rot_step(self.R, self.omega, self.dt)
        self.last_actions = _t.cat([self.last_actions[:, 1:], action.unsqueeze(1)], dim=1)
        return self.get_obs(), self._reward(action)

    CrazyflieDiffSim.step = step


if __name__ == "__main__":
    main()
