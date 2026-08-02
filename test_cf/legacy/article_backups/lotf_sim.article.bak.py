"""Differentiable Crazyflie hover simulator — LOTF-aligned (SI, 50 Hz, 27-d obs).

Matches `lotf/envs/hovering_state_env.py` + `scripts/rl_controller_lotf.py`:
    action      = [thrust_total_N, ωx, ωy, ωz]   in SI, NOT clipped to [-1,1]
    obs (27)    = [rel_pos_norm(3), R_flat(9), v_norm(3), last_actions_norm(12)]
    last-action buffer length = NUM_LAST_ACTIONS = 3  (simulates command delay)
    dt          = 0.02 s (50 Hz)
    integrator  = rate-control cascade with first-order body-rate tracking
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cf_params import (MASS, G, DT, NUM_LAST_ACTIONS, HOVER_GOAL,
                       WORLD_BOX_HALF, V_BOUND,
                       ACTION_LOW, ACTION_HIGH, HOVERING_ACTION)

TAU_W = 0.06   # body-rate first-order tracking constant (s)


# ──────────────────────────── helpers ────────────────────────────
def _skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([
        torch.stack([z, -w[..., 2], w[..., 1]], dim=-1),
        torch.stack([w[..., 2], z, -w[..., 0]], dim=-1),
        torch.stack([-w[..., 1], w[..., 0], z], dim=-1),
    ], dim=-2)


def _reortho(R):
    """Réorthonormalise une matrice ~SO(3) par Gram-Schmidt (dérive numérique RK4)."""
    c0 = R[..., :, 0]; c0 = c0 / (c0.norm(dim=-1, keepdim=True) + 1e-8)
    c1 = R[..., :, 1]
    c1 = c1 - (c1 * c0).sum(-1, keepdim=True) * c0
    c1 = c1 / (c1.norm(dim=-1, keepdim=True) + 1e-8)
    c2 = torch.cross(c0, c1, dim=-1)
    return torch.stack([c0, c1, c2], dim=-1)


def _minmax(x, lo, hi):
    return (2.0 * (x - lo) / (hi - lo) - 1.0).clamp(-1.0, 1.0)


# ──────────────────────────── env ────────────────────────────
class CrazyflieDiffSim:
    """Batched torch differentiable sim. Residual model is plugged in dynamically."""

    def __init__(self, num_envs: int, residual_model: nn.Module | None = None,
                 device: str = "cpu", goal=None, detach_residual: bool = True):
        self.num_envs = num_envs
        self.dt = DT
        self.device = device
        self.residual_model = residual_model
        # Article (Sec III-E) : le gradient de policy ne passe QUE par le modèle
        # analytique, pas par le réseau résiduel. On détache donc sa sortie pendant
        # le BPTT (sa VALEUR contribue au modèle hybride, mais pas son gradient).
        self.detach_residual = detach_residual

        # Precompute tensors
        self.goal = torch.tensor(goal if goal is not None else HOVER_GOAL,
                                 device=device, dtype=torch.float32)
        self.action_low = torch.tensor(ACTION_LOW, device=device)
        self.action_high = torch.tensor(ACTION_HIGH, device=device)
        self.act_buf_low = self.action_low.repeat(NUM_LAST_ACTIONS)
        self.act_buf_high = self.action_high.repeat(NUM_LAST_ACTIONS)
        self.hovering_action = torch.tensor(HOVERING_ACTION, device=device)
        self.g_vec = torch.tensor([0.0, 0.0, -G], device=device)

        self.reset()

    # ─────────────────────────── reset ───────────────────────────
    def reset(self):
        n, d = self.num_envs, self.device
        # randomise initial pose around the goal (small box, near-upright)
        self.p_pos = self.goal + 0.5 * (torch.rand(n, 3, device=d) - 0.5)
        self.v = 0.1 * torch.randn(n, 3, device=d)
        self.R = torch.eye(3, device=d).expand(n, 3, 3).contiguous()
        # ω = ω_cmd à chaque pas (pas d'état de taux : article éq. 1, Ṙ = R[ω_cmd]×).
        # On garde self.omega = dernier ω commandé, seulement pour le terme de reward.
        self.omega = torch.zeros(n, 3, device=d)
        # last-action FIFO seeded with hovering action
        self.last_actions = self.hovering_action.expand(n, NUM_LAST_ACTIONS, 4).contiguous()
        return self.get_obs()

    # ────────────────────────── observation ──────────────────────
    def get_obs(self) -> torch.Tensor:
        rel_pos = ((self.p_pos - self.goal) / WORLD_BOX_HALF).clamp(-1.0, 1.0)
        R_flat = self.R.reshape(self.num_envs, 9)
        v_norm = (self.v / V_BOUND).clamp(-1.0, 1.0)
        act_flat = self.last_actions.reshape(self.num_envs, -1)
        act_norm = _minmax(act_flat, self.act_buf_low, self.act_buf_high)
        return torch.cat([rel_pos, R_flat, v_norm, act_norm], dim=-1)

    # ───────────────── dynamique continue (article éq. 1 + résiduel) ─────────────
    def _residual_accel(self, p, R, v, T_N, omega_cmd):
        """â_res(p,R,v,c,ω) ; détachée si detach_residual (gradient analytique seul)."""
        if self.residual_model is None:
            return torch.zeros_like(v)
        feats = torch.cat([p, R.reshape(self.num_envs, 9), v,
                           (T_N / MASS).unsqueeze(-1), omega_cmd], dim=-1)
        out = self.residual_model(feats)
        return out.detach() if self.detach_residual else out

    def _deriv(self, p, R, v, T_N, omega_cmd):
        """ẋ = [v, R[ω_cmd]×, g + R·[0,0,c] + â_res]  (modèle hybride, article Sec III)."""
        dp = v
        dR = R @ _skew(omega_cmd)
        thrust_body = torch.zeros_like(v)
        thrust_body[:, 2] = T_N / MASS
        dv = (self.g_vec + torch.einsum("nij,nj->ni", R, thrust_body)
              + self._residual_accel(p, R, v, T_N, omega_cmd))
        return dp, dR, dv

    # ───────────────────────────── step (RK4) ────────────────────
    def step(self, action: torch.Tensor):
        """action: (n, 4) in SI units = [thrust_N, ωx, ωy, ωz]. Returns (obs, reward).

        Intégration Runge-Kutta 4 à 50 Hz (article Sec III-E), commande tenue
        constante (zero-order hold) sur le pas. La poussée est suivie sans retard
        (ω = ω_cmd), les écarts réels étant capturés par le résiduel."""
        action = torch.maximum(self.action_low, torch.minimum(self.action_high, action))
        T_N = action[:, 0]
        omega_cmd = action[:, 1:]
        self.omega = omega_cmd                         # dernier ω commandé (pour la reward)

        dt = self.dt
        p, R, v = self.p_pos, self.R, self.v
        k1p, k1R, k1v = self._deriv(p, R, v, T_N, omega_cmd)
        k2p, k2R, k2v = self._deriv(p + 0.5*dt*k1p, R + 0.5*dt*k1R, v + 0.5*dt*k1v, T_N, omega_cmd)
        k3p, k3R, k3v = self._deriv(p + 0.5*dt*k2p, R + 0.5*dt*k2R, v + 0.5*dt*k2v, T_N, omega_cmd)
        k4p, k4R, k4v = self._deriv(p + dt*k3p, R + dt*k3R, v + dt*k3v, T_N, omega_cmd)

        self.p_pos = p + (dt/6.0) * (k1p + 2*k2p + 2*k3p + k4p)
        self.v = v + (dt/6.0) * (k1v + 2*k2v + 2*k3v + k4v)
        self.R = _reortho(R + (dt/6.0) * (k1R + 2*k2R + 2*k3R + k4R))

        # FIFO push (oldest dropped)
        self.last_actions = torch.cat(
            [self.last_actions[:, 1:], action.unsqueeze(1)], dim=1
        )

        return self.get_obs(), self._reward(action)

    # ──────────────────────────── reward (article Sec IV-A-1) ─────────────────────
    def _reward(self, action):
        """r = r_pos + r_vel + r_act, pertes de Huber (article, hover stabilisé) :
            r_pos = -1.0·L_H(5·(p−p_des))
            r_vel = -0.1·L_H(v) − 0.1·L_H(ω)
            r_act = -0.5·L_H(u − u_hover),  u = [c, ω],  u_hover = [g, 0, 0, 0]."""
        def huber(x):
            return torch.nn.functional.huber_loss(
                x, torch.zeros_like(x), reduction="none", delta=1.0).sum(-1)

        r_pos = -1.0 * huber(5.0 * (self.p_pos - self.goal))
        r_vel = -0.1 * huber(self.v) - 0.1 * huber(self.omega)

        c_t = action[:, 0:1] / MASS                    # poussée mass-normalisée
        u = torch.cat([c_t, action[:, 1:]], dim=-1)
        u_hover = torch.tensor([G, 0.0, 0.0, 0.0], device=u.device, dtype=u.dtype)
        r_act = -0.5 * huber(u - u_hover)

        return r_pos + r_vel + r_act
