"""Shared helpers for time-to-collision (TTC) labels and baselines."""
from __future__ import annotations

import numpy as np

# Label / evaluation defaults (also documented in docs/ttc_labels.md)
TTC_CAP_S = 10.0
MIN_CLOSING_SPEED_M_S = 0.5


def compute_ttc_xy_global(
    obj_xy: np.ndarray,
    ego_xy: np.ndarray,
    vel_xy_global: np.ndarray,
    ttc_cap: float = TTC_CAP_S,
    min_closing_speed: float = MIN_CLOSING_SPEED_M_S,
):
    """TTC in the global horizontal (XY) plane.

    Closing speed is the rate at which range decreases:
    ``closing = -dot(v, r_hat)`` with ``r_hat`` the unit vector from ego to
    object in XY. Only defined for ``closing >= min_closing_speed``.

    Returns:
        tuple: (ttc: float or None, distance: float, closing_speed: float)
    """
    obj_xy = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    ego_xy = np.asarray(ego_xy, dtype=np.float64).reshape(2)
    vel_xy_global = np.asarray(vel_xy_global, dtype=np.float64).reshape(2)

    dr = obj_xy - ego_xy
    dist = float(np.linalg.norm(dr))
    if dist < 1e-6:
        return None, dist, 0.0

    r_hat = dr / dist
    closing = float(-np.dot(vel_xy_global, r_hat))
    if closing < min_closing_speed or not np.isfinite(closing):
        return None, dist, closing

    ttc = min(dist / closing, ttc_cap)
    if not np.isfinite(ttc):
        return None, dist, closing
    return ttc, dist, closing


def lidar_velocity_to_global(
    lidar2global: np.ndarray,
    vel_xy_lidar: np.ndarray,
) -> np.ndarray:
    """Rotate lidar horizontal velocity into global frame (3-vector)."""
    lidar2global = np.asarray(lidar2global, dtype=np.float64).reshape(4, 4)
    R = lidar2global[:3, :3]
    v_l = np.array([vel_xy_lidar[0], vel_xy_lidar[1], 0.0], dtype=np.float64)
    return R @ v_l


def ego_xy_global_from_info(info: dict) -> np.ndarray:
    """Ego XY in global frame (matches ego_pose on the keyframe LIDAR sample)."""
    return np.asarray(info["ego2global_translation"], dtype=np.float64)[:2]


def build_lidar2global(info: dict) -> np.ndarray:
    from pyquaternion import Quaternion

    l2e_r = info["lidar2ego_rotation"]
    l2e_t = np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    e2g_r = info["ego2global_rotation"]
    e2g_t = np.asarray(info["ego2global_translation"], dtype=np.float64)

    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    lidar2global = np.eye(4, dtype=np.float64)
    lidar2global[:3, :3] = e2g_r_mat @ l2e_r_mat
    lidar2global[:3, 3] = e2g_t + e2g_r_mat @ l2e_t
    return lidar2global
