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


def _rot_step(R, omega, dt):
    R_new = R + dt * (R @ _skew(omega))
    c0 = R_new[..., :, 0]; c0 = c0 / (c0.norm(dim=-1, keepdim=True) + 1e-8)
    c1 = R_new[..., :, 1]
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
                 device: str = "cpu", goal=None):
        self.num_envs = num_envs
        self.dt = DT
        self.device = device
        self.residual_model = residual_model

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
        self.omega = 0.1 * torch.randn(n, 3, device=d)
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

    # ───────────────────────────── step ──────────────────────────
    def step(self, action: torch.Tensor):
        """action: (n, 4) in SI units = [thrust_N, ωx, ωy, ωz]. Returns (obs, reward)."""
        action = torch.maximum(self.action_low, torch.minimum(self.action_high, action))
        T_N = action[:, 0]
        omega_cmd = action[:, 1:]

        # first-order body-rate tracking
        self.omega = self.omega + self.dt * (omega_cmd - self.omega) / TAU_W

        # world-frame acceleration: g + R · [0, 0, T/m] (+ residual)
        thrust_body = torch.zeros_like(self.v)
        thrust_body[:, 2] = T_N / MASS
        a = self.g_vec + torch.einsum("nij,nj->ni", self.R, thrust_body)
        if self.residual_model is not None:
            feats = self._residual_features(T_N)
            a = a + self.residual_model(feats).detach()

        # integrate
        self.v = self.v + self.dt * a
        self.p_pos = self.p_pos + self.dt * self.v
        self.R = _rot_step(self.R, self.omega, self.dt)

        # FIFO push (oldest dropped)
        self.last_actions = torch.cat(
            [self.last_actions[:, 1:], action.unsqueeze(1)], dim=1
        )

        return self.get_obs(), self._reward(action)

    # ──────────────────────────── reward ─────────────────────────
    def _reward(self, action):
        pos_err = (self.p_pos - self.goal).pow(2).sum(-1)
        vel_pen = self.v.pow(2).sum(-1)
        omega_pen = self.omega.pow(2).sum(-1)
        action_pen = (action - self.hovering_action).pow(2).sum(-1)
        upright = self.R[:, 2, 2]
        return (-1.0 * pos_err
                - 0.05 * vel_pen
                - 0.01 * omega_pen
                - 0.05 * action_pen
                + 0.2 * upright)

    # ─────────────────── residual MLP feature builder ───────────
    def _residual_features(self, T_N):
        return torch.cat([
            self.p_pos, self.R.reshape(self.num_envs, 9), self.v,
            T_N.unsqueeze(-1), self.omega,
        ], dim=-1)
