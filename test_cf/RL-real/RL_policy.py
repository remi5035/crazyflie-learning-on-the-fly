"""LOTF hover policy (27-dim obs, 4-dim SI action, MLP 512×512 + action_bias).

Mirrors `lotf/modules/mlp.MLP` + the policy loaded by `scripts/rl_controller_lotf.py`.
The MLP outputs an action *delta* from a fixed `HOVERING_ACTION` bias; the
controller clips to the physical action bounds before sending to the drone.
"""
import torch
import torch.nn as nn

from cf_params import (NUM_OBS, NUM_ACTIONS, HIDDEN_DIMS,
                       HOVERING_ACTION, ACTION_LOW, ACTION_HIGH)


class Actor(nn.Module):
    """[27, 512, 512, 4] tanh MLP with constant additive bias = hovering action.

    `initial_scale` shrinks the last layer so the untrained net outputs ≈
    HOVERING_ACTION (i.e. the drone hovers from epoch 0 of BPTT).
    """
    def __init__(self, initial_scale: float = 0.01):
        super().__init__()
        layers = []
        last = NUM_OBS
        for h in HIDDEN_DIMS:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        head = nn.Linear(last, NUM_ACTIONS)
        with torch.no_grad():
            head.weight.mul_(initial_scale)
            head.bias.zero_()
        layers.append(head)
        self.actor = nn.Sequential(*layers)
        self.register_buffer("action_bias", torch.tensor(HOVERING_ACTION))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x) + self.action_bias


class RLPolicy:
    def __init__(self, model_path: str):
        self.device = torch.device("cpu")
        ckpt = torch.load(model_path, map_location=self.device)
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        self.policy = Actor()
        self.policy.load_state_dict(sd, strict=True)
        self.policy.eval()
        self._low = torch.tensor(ACTION_LOW)
        self._high = torch.tensor(ACTION_HIGH)
        print(f"[RLPolicy] loaded {model_path}")

    @classmethod
    def from_state_dict(cls, state_dict):
        """Build a ready-to-run policy from weights only (used for in-flight hot-swap)."""
        obj = cls.__new__(cls)
        obj.device = torch.device("cpu")
        obj.policy = Actor()
        obj.policy.load_state_dict(state_dict, strict=True)
        obj.policy.eval()
        obj._low = torch.tensor(ACTION_LOW)
        obj._high = torch.tensor(ACTION_HIGH)
        return obj

    def get_action(self, obs_vec: torch.Tensor) -> torch.Tensor:
        """obs_vec: 27-dim torch tensor (already normalized by DroneState)."""
        with torch.no_grad():
            a = self.policy(obs_vec)
            a = torch.maximum(self._low, torch.minimum(self._high, a))
        return a
