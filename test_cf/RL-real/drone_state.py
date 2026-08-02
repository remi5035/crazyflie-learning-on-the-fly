"""DroneState — assembles the 27-dim LOTF observation from cflib logs.

Layout (matches `scripts/rl_controller_lotf.py._build_obs`):
    rel_pos_norm (3)  : (drone - goal) / WORLD_BOX_HALF, clipped [-1, 1]
    R_flat       (9)  : world-from-body rotation matrix, flattened
    vel_norm     (3)  : vel / V_BOUND, clipped [-1, 1]
    last_actions_norm (4 * NUM_LAST_ACTIONS) : per-element min-max normalised
"""
import time
from collections import deque

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_scipy

from cf_params import (NUM_LAST_ACTIONS, HOVERING_ACTION, ACTION_LOW, ACTION_HIGH,
                       WORLD_BOX_HALF, V_BOUND, HOVER_GOAL)


def _minmax(x, lo, hi):
    """Scale to [-1, 1] given physical bounds [lo, hi]."""
    return np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def _quat_wxyz_to_R(q):
    """w, x, y, z -> 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


class DroneState:
    def __init__(self, goal=None):
        self.goal = np.array(goal if goal is not None else HOVER_GOAL, dtype=np.float64)
        self.latest = {
            'pos':  np.zeros(3),
            'vel':  np.zeros(3),
            'quat_wxyz': np.array([1.0, 0.0, 0.0, 0.0]),
            'gyro': np.zeros(3),   # logged only for plotting / residual fit
        }
        self.last_actions = deque(
            [HOVERING_ACTION.copy() for _ in range(NUM_LAST_ACTIONS)],
            maxlen=NUM_LAST_ACTIONS,
        )
        # Plotting history (kept from the legacy file so the post-flight plots work)
        self.history = {'time': [], 'x_pos': [], 'y_pos': [], 'z_pos': [],
                        'roll': [], 'pitch': []}
        self.start_flight_time = 0.0
        self.RL = False

        # Precomputed bounds for last-actions normalisation
        self._act_buf_low = np.tile(ACTION_LOW, NUM_LAST_ACTIONS)
        self._act_buf_high = np.tile(ACTION_HIGH, NUM_LAST_ACTIONS)

    # ───────────────────────────── from cflib logs ─────────────────────────────
    def update_from_log(self, data):
        if 'stateEstimate.x' in data:
            self.latest['pos'] = np.array([
                data['stateEstimate.x'], data['stateEstimate.y'], data['stateEstimate.z']])
            if self.RL:
                if self.start_flight_time == 0.0:
                    self.start_flight_time = time.time()
                t = time.time() - self.start_flight_time
                self.history['time'].append(t)
                self.history['x_pos'].append(data['stateEstimate.x'])
                self.history['y_pos'].append(data['stateEstimate.y'])
                self.history['z_pos'].append(data['stateEstimate.z'])

        if 'stateEstimate.vx' in data:
            self.latest['vel'] = np.array([
                data['stateEstimate.vx'], data['stateEstimate.vy'], data['stateEstimate.vz']])

        if 'gyro.x' in data:  # deg/s, body frame
            self.latest['gyro'] = np.array([data['gyro.x'], data['gyro.y'], data['gyro.z']])

        if 'stateEstimate.qx' in data:
            qw, qx, qy, qz = (data['stateEstimate.qw'], data['stateEstimate.qx'],
                              data['stateEstimate.qy'], data['stateEstimate.qz'])
            self.latest['quat_wxyz'] = np.array([qw, qx, qy, qz])
            if self.RL:
                rpy = R_scipy.from_quat([qx, qy, qz, qw]).as_euler('xyz', degrees=True)
                self.history['roll'].append(rpy[0])
                self.history['pitch'].append(rpy[1])

    # ───────────────────────────── observation ─────────────────────────────
    def get_obs(self) -> torch.Tensor:
        """Build the 27-dim normalised obs and return as torch.float32."""
        rel_pos = (self.latest['pos'] - self.goal) / WORLD_BOX_HALF
        rel_pos = np.clip(rel_pos, -1.0, 1.0)

        R = _quat_wxyz_to_R(self.latest['quat_wxyz']).flatten()

        v_norm = np.clip(self.latest['vel'] / V_BOUND, -1.0, 1.0)

        act_flat = np.concatenate(list(self.last_actions))
        act_norm = _minmax(act_flat, self._act_buf_low, self._act_buf_high)

        obs = np.concatenate([rel_pos, R, v_norm, act_norm]).astype(np.float32)
        return torch.from_numpy(obs)

    # ───────────────────────────── housekeeping ─────────────────────────────
    def set_goal(self, goal):
        """Update the hover target at runtime (world frame, metres)."""
        self.goal = np.asarray(goal, dtype=np.float64)

    def push_action(self, action_si: np.ndarray):
        """Append the latest action (already in SI units) to the FIFO."""
        self.last_actions.append(np.asarray(action_si, dtype=np.float32))
