"""Residual dynamics MLP (19 → 128 → 128 → 3) + offline dataset builder.

Same role as `lotf/utils/residual_dynamics.py`, but in PyTorch and consuming
the `*_lotf.npz` files saved by `drone_controller.save_lotf_log`.

Input features (19): [pos(3), R_flat(9), vel(3), thrust_N, omega(3)]
Target (3):         a_meas - a_nominal  (centred FD on v;  a_nom = g + R·[0,0,T/m])
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from cf_params import MASS, G


class ResidualDynamicsMLP(nn.Module):
    def __init__(self, in_dim=19, hidden=128, out_dim=3, initial_scale=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(initial_scale)
            self.net[-1].bias.mul_(initial_scale)

    def forward(self, x):
        return self.net(x)
    
    def spectral_norm_reg(self):
        """Σ_l ‖Wˡ‖₂ — somme des normes spectrales (plus grande valeur singulière)
        de chaque couche Linear, comme dans la loss résiduelle LOTF."""
        reg = 0.0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                reg = reg + torch.linalg.matrix_norm(layer.weight, ord=2)
        return reg
    



def build_residual_dataset(log: dict, window_sec: float | None = None):
    """log keys: ['t','p','R','v','T_N','omega'] → (X[N,19], y[N,3]).

    window_sec: if set, only keep the last `window_sec` seconds of the log
    (LOTF paper fits the residual on a short trailing window, ~2 s, so it tracks
    the *recent* dynamics — battery sag, local disturbances — instead of
    averaging the whole flight). None = use the entire log.
    """
    t = np.asarray(log['t']).reshape(-1)
    p = np.asarray(log['p'])
    R = np.asarray(log['R']).reshape(-1, 3, 3)
    v = np.asarray(log['v'])
    T_N = np.asarray(log['T_N']).reshape(-1)
    omega = np.asarray(log['omega'])

    # Trailing-window selection (done before finite differences so the kept
    # samples have valid neighbours inside the window).
    if window_sec is not None and t.size:
        keep = t >= (t[-1] - window_sec)
        t, p, R, v, T_N, omega = t[keep], p[keep], R[keep], v[keep], T_N[keep], omega[keep]

    a_meas = np.zeros_like(v)
    a_meas[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])[:, None]

    thrust_body = np.zeros_like(v); thrust_body[:, 2] = T_N / MASS
    a_nom = np.array([0.0, 0.0, -G]) + np.einsum("nij,nj->ni", R, thrust_body)
    y = a_meas - a_nom

    # clean: drop edges, dt jitter, NaN/inf, MAD outliers
    dt = np.diff(t)
    median_dt = np.median(dt) if len(dt) > 0 else 0.02
    bad_dt = np.concatenate([[True], np.abs(dt - median_dt) > 0.5 * median_dt])
    mask = np.ones(len(t), dtype=bool)
    mask[[0, -1]] = False
    mask &= ~bad_dt
    mask &= np.all(np.isfinite(y), axis=1)

    y_norm = np.linalg.norm(y, axis=1)
    if mask.any():
        med = np.median(y_norm[mask])
        mad = np.median(np.abs(y_norm[mask] - med)) + 1e-6
        mask &= y_norm < med + 6.0 * 1.4826 * mad

    X = np.concatenate([p, R.reshape(-1, 9), v, T_N[:, None], omega], axis=1)[mask]
    return X.astype(np.float32), y[mask].astype(np.float32)


def fit_residual(X, y, *, epochs=200, lr=1e-2, beta=1e-3,
                 hidden=128, device="cpu", verbose=True):
    model = ResidualDynamicsMLP(hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr) 
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    for ep in range(epochs):
        pred = model(Xt)
        mse = (pred - yt).pow(2).mean()
        reg = model.spectral_norm_reg()
        loss = mse + beta * reg
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % max(1, epochs // 10) == 0:
            print(f"  [residual] ep={ep:4d}  mse={loss.item():.4f}")
    if verbose:
        print(f"  [residual] final mse={loss.item():.4f}  on {len(X)} samples")
    return model


def load_log_npz(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}
