#!/usr/bin/env python3
"""Heuristic and linear TTC baselines vs. StreamPETR box/velocity outputs.

Requires:
- Pkl from "tools/test.py --out"
- Pkl from "tools/generate_ttc_labels.py"
"""

from __future__ import annotations

import argparse
import os.path as osp
import pickle
import sys

import numpy as np

_ROOT = osp.dirname(osp.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttc_utils import (
    MIN_CLOSING_SPEED_M_S,
    TTC_CAP_S,
    build_lidar2global,
    compute_ttc_xy_global,
    ego_xy_global_from_info,
    lidar_velocity_to_global,
)

CLASS_NAMES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def parse_args():
    p = argparse.ArgumentParser(description="TTC heuristic + linear baselines")
    p.add_argument("--data-root", required=True, help="nuScenes dataroot")
    p.add_argument(
        "--info",
        required=True,
        help="Temporal infos pkl (must match evaluation split, e.g. *val*.pkl)",
    )
    p.add_argument("--ttc-labels", required=True, help="Output of generate_ttc_labels.py")
    p.add_argument(
        "--pred-results",
        required=True,
        help="list of outputs from tools/test.py --out (one entry per dataset index)",
    )
    p.add_argument("--version", default="v1.0-mini", help="nuScenes version for sample lookups")
    p.add_argument("--score-thr", type=float, default=0.2)
    p.add_argument("--match-dist", type=float, default=2.5, help="BEV L2 match threshold (m)")
    p.add_argument("--train-frac", type=float, default=0.8, help="Train fraction for linear baselines")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--lasso-alpha", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _boxes_tensor_to_numpy(boxes_3d):
    if hasattr(boxes_3d, "tensor"):
        return boxes_3d.tensor.detach().cpu().numpy()
    return np.asarray(boxes_3d)


def _pred_state_from_row(box_row: np.ndarray):
    """LiDAR-frame center and horizontal velocity from decoded StreamPETR row."""
    x, y, z = box_row[0], box_row[1], box_row[2]
    w, l, h = box_row[3], box_row[4], box_row[5]
    if box_row.shape[0] >= 9:
        vx, vy = box_row[7], box_row[8]
    else:
        vx, vy = 0.0, 0.0
    return np.array([x, y, z], dtype=np.float64), np.array([vx, vy], dtype=np.float64), (
        w,
        l,
        h,
    )


def pred_geometry_global(box_row: np.ndarray, lidar2global: np.ndarray):
    center_l, vel_l, whl = _pred_state_from_row(box_row)
    pos_g = (lidar2global @ np.array([center_l[0], center_l[1], center_l[2], 1.0], dtype=np.float64))[:3]
    vel_g = lidar_velocity_to_global(lidar2global, vel_l)
    return pos_g[:2], vel_g[:2], whl


def heuristic_ttc_pred(box_row: np.ndarray, lidar2global: np.ndarray, ego_xy: np.ndarray):
    obj_xy, vel_xy, _ = pred_geometry_global(box_row, lidar2global)
    return compute_ttc_xy_global(
        obj_xy,
        ego_xy,
        vel_xy,
        ttc_cap=TTC_CAP_S,
        min_closing_speed=MIN_CLOSING_SPEED_M_S,
    )


def recall_at_3s(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of GT risk objects (TTC < 3s) with predicted TTC < 3s."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true < 3.0
    if not np.any(mask):
        return float("nan")
    sub = y_pred[mask]
    hit = np.isfinite(sub) & (sub < 3.0)
    return float(np.mean(hit))


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.nanmean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))
    return mae, rmse


def match_gt_pred(
    pred_centers: np.ndarray,
    pred_labels: np.ndarray,
    gt_centers: np.ndarray,
    gt_labels: np.ndarray,
    max_center_m: float,
):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        linear_sum_assignment = None

    n_p, n_g = pred_labels.shape[0], gt_labels.shape[0]
    if n_p == 0 or n_g == 0:
        return {}

    cost = np.full((n_p, n_g), 1e6, dtype=np.float64)
    for i in range(n_p):
        for j in range(n_g):
            if pred_labels[i] != gt_labels[j]:
                continue
            cost[i, j] = np.linalg.norm(pred_centers[i] - gt_centers[j])

    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        pairs = {}
        for i, j in zip(r, c):
            if cost[i, j] < max_center_m:
                pairs[int(j)] = int(i)
        return pairs

    # Greedy fallback
    pairs = {}
    used_p, used_g = set(), set()
    flat = [(float(cost[i, j]), i, j) for i in range(n_p) for j in range(n_g) if cost[i, j] < 1e5]
    flat.sort()
    for d, i, j in flat:
        if d >= max_center_m:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        pairs[j] = i
    return pairs


def add_bias_column(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)


def fit_ols_ridge(X: np.ndarray, y: np.ndarray, ridge_alpha: float = 0.0):
    """OLS when ridge_alpha=0, else ridge closed-form."""
    Xb = add_bias_column(X)
    n_feat = Xb.shape[1]
    a = ridge_alpha * np.eye(n_feat, dtype=np.float64)
    a[-1, -1] = 0.0  # do not regularize intercept
    coef = np.linalg.lstsq(Xb.T @ Xb + a, Xb.T @ y, rcond=None)[0]
    return coef


def predict_linear(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    Xb = add_bias_column(X)
    return Xb @ coef


def unpack_pts_bbox(pbox: dict):
    """Support mmdet3d bbox3d2result key naming variants."""
    if "boxes_3d" in pbox:
        boxes = pbox["boxes_3d"]
    elif "bboxes" in pbox:
        boxes = pbox["bboxes"]
    else:
        raise KeyError(f"No 3D boxes in pts_bbox; keys={pbox.keys()}")

    scores = pbox.get("scores_3d", pbox.get("scores"))
    labels = pbox.get("labels_3d", pbox.get("labels"))
    if scores is None or labels is None:
        raise KeyError(f"Missing scores/labels in pts_bbox; keys={pbox.keys()}")
    return boxes, scores, labels


def build_feature_matrix(dist: np.ndarray, closing: np.ndarray, cls_ids: np.ndarray, sizes: np.ndarray, num_classes: int):
    oh = np.zeros((cls_ids.shape[0], num_classes), dtype=np.float64)
    oh[np.arange(cls_ids.shape[0]), cls_ids] = 1.0
    return np.concatenate(
        [
            dist.reshape(-1, 1),
            closing.reshape(-1, 1),
            sizes,
            oh,
        ],
        axis=1,
    )


def print_results_table(rows):
    headers = ["Method", "MAE", "RMSE", "Recall@3s"]
    print("\n| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for name, mae, rmse, rec in rows:
        rec_s = f"{rec:.4f}" if np.isfinite(rec) else "n/a"
        print(f"| {name} | {mae:.4f} | {rmse:.4f} | {rec_s} |")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    from mmcv import load as mmcv_load
    from nuscenes.nuscenes import NuScenes

    infos = mmcv_load(args.info)["infos"]
    with open(args.ttc_labels, "rb") as f:
        ttc_payload = pickle.load(f)
    if isinstance(ttc_payload, dict) and "labels" in ttc_payload:
        labels = ttc_payload["labels"]
    else:
        labels = ttc_payload

    with open(args.pred_results, "rb") as f:
        pred_results = pickle.load(f)

    if len(pred_results) != len(infos):
        raise ValueError(
            f"pred_results length {len(pred_results)} != infos {len(infos)}. "
            "Regenerate predictions on the same ann_file / split."
        )

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    pairs_data = []

    for idx, info in enumerate(infos):
        sample = nusc.get("sample", info["token"])
        ann_tokens = sample["anns"]
        valid = info["valid_flag"].astype(bool)
        gt_boxes = info["gt_boxes"]
        gt_names = info["gt_names"]

        res = pred_results[idx]
        if isinstance(res, dict) and "pts_bbox" in res:
            pbox = res["pts_bbox"]
        elif isinstance(res, dict) and ("boxes_3d" in res or "bboxes" in res):
            pbox = res
        else:
            pbox = res

        boxes_raw, scores_raw, labels_raw = unpack_pts_bbox(pbox)
        boxes = _boxes_tensor_to_numpy(boxes_raw)
        scores = np.asarray(scores_raw)
        plabs = np.asarray(labels_raw)

        keep = scores >= args.score_thr
        boxes, scores, plabs = boxes[keep], scores[keep], plabs[keep]

        lidar2global = build_lidar2global(info)
        ego_xy = ego_xy_global_from_info(info)

        gt_centers = []
        gt_labels_arr = []
        gt_ann_tokens = []
        for j, tok in enumerate(ann_tokens):
            if not valid[j]:
                continue
            if tok not in labels:
                continue
            try:
                cls_i = CLASS_NAMES.index(gt_names[j])
            except ValueError:
                continue
            gt_centers.append(gt_boxes[j, :2].copy())
            gt_labels_arr.append(cls_i)
            gt_ann_tokens.append(tok)
        if not gt_centers:
            continue
        gt_centers = np.asarray(gt_centers, dtype=np.float64)
        gt_labels_arr = np.asarray(gt_labels_arr, dtype=np.int64)

        pred_centers = boxes[:, :2] if len(boxes) else np.zeros((0, 2), dtype=np.float64)

        mp = match_gt_pred(
            pred_centers,
            plabs,
            gt_centers,
            gt_labels_arr,
            args.match_dist,
        )

        for gti, pi in mp.items():
            tok = gt_ann_tokens[gti]
            y = labels[tok]["ttc"]
            row = boxes[pi]
            pos_xy, vel_xy, whl = pred_geometry_global(row, lidar2global)
            dist = float(np.linalg.norm(pos_xy - ego_xy))
            r_hat = (pos_xy - ego_xy) / max(dist, 1e-6)
            closing = float(-np.dot(vel_xy, r_hat))

            ttc_h, _, _ = heuristic_ttc_pred(row, lidar2global, ego_xy)

            w, l, h = whl
            pairs_data.append(
                dict(
                    gt_ttc=y,
                    heuristic_ttc=ttc_h,
                    pred_dist=dist,
                    pred_closing=closing,
                    pred_class=int(plabs[pi]),
                    pred_wlh=np.array([w, l, h], dtype=np.float64),
                )
            )

    if not pairs_data:
        raise RuntimeError("No matched GT/pred pairs. Check score threshold, split, and labels.")

    y = np.array([d["gt_ttc"] for d in pairs_data], dtype=np.float64)
    h_list = [d["heuristic_ttc"] for d in pairs_data]
    h = np.array([np.nan if v is None else float(v) for v in h_list], dtype=np.float64)

    dist = np.array([d["pred_dist"] for d in pairs_data], dtype=np.float64)
    closing = np.array([d["pred_closing"] for d in pairs_data], dtype=np.float64)
    cls_ids = np.array([d["pred_class"] for d in pairs_data], dtype=np.int64)
    sizes = np.stack([d["pred_wlh"] for d in pairs_data], axis=0)
    X = build_feature_matrix(dist, closing, cls_ids, sizes, len(CLASS_NAMES))

    n = X.shape[0]
    n_pairs = n
    if n < 2:
        raise RuntimeError("Need at least 2 matched pairs for linear baselines (train/test split).")
    perm = np.random.permutation(n)
    split = max(1, int(n * args.train_frac))
    if split >= n:
        split = n - 1
    tr, te = perm[:split], perm[split:]

    y_tr, y_te = y[tr], y[te]
    h_tr, h_te = h[tr], h[te]
    X_tr, X_te = X[tr], X[te]

    mae_h_te, rmse_h_te = mae_rmse(y_te, h_te)
    rec_h_te = recall_at_3s(y_te, h_te)
    n_h_fin_te = int(np.sum(np.isfinite(h_te)))
    results = [
        (
            f"Heuristic (StreamPETR box + vel; test split; finite pred {n_h_fin_te}/{len(te)})",
            mae_h_te,
            rmse_h_te,
            rec_h_te,
        )
    ]
    mae_h_all, rmse_h_all = mae_rmse(y, h)
    rec_h_all = recall_at_3s(y, h)
    n_h_fin_all = int(np.sum(np.isfinite(h)))

    coef_ols = fit_ols_ridge(X_tr, y_tr, ridge_alpha=0.0)
    p_ols = predict_linear(X_te, coef_ols)
    results.append(("Linear regression (OLS)", *mae_rmse(y_te, p_ols), recall_at_3s(y_te, p_ols)))

    coef_r = fit_ols_ridge(X_tr, y_tr, ridge_alpha=args.ridge_alpha)
    p_r = predict_linear(X_te, coef_r)
    results.append(
        (f"Ridge (L2, alpha={args.ridge_alpha})", *mae_rmse(y_te, p_r), recall_at_3s(y_te, p_r))
    )

    lasso_row = None
    try:
        from sklearn.linear_model import Lasso

        las = Lasso(alpha=args.lasso_alpha, max_iter=10000, fit_intercept=True)
        las.fit(X_tr, y_tr)
        p_l = las.predict(X_te)
        lasso_name = f"Lasso (L1, alpha={args.lasso_alpha})"
        lasso_row = (lasso_name, *mae_rmse(y_te, p_l), recall_at_3s(y_te, p_l))
    except ImportError:
        print("scikit-learn not installed; skipping Lasso baseline.")
    except Exception as e:
        print(f"Lasso failed ({e}); skipping.")

    if lasso_row:
        results.append(lasso_row)

    print(f"\nMatched objects (train {len(tr)}, eval {len(te)} of {n} total pairs).")
    print(
        "Heuristic on all matched pairs (reference, not comparable to test-only rows): "
        f"MAE={mae_h_all:.4f}, RMSE={rmse_h_all:.4f}, Recall@3s={rec_h_all:.4f}, "
        f"finite pred {n_h_fin_all}/{n_pairs}"
    )
    print("\nAll methods below use the **same** held-out test indices.\n")
    print_results_table(results)


if __name__ == "__main__":
    main()
