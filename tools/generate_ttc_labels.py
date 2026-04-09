"""Generate ground-truth TTC labels from nuScenes annotations (mini or full).

Design choices are summarized in docs/ttc_labels.md.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
import sys

import numpy as np

_ROOT = osp.dirname(osp.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ttc_utils import MIN_CLOSING_SPEED_M_S, TTC_CAP_S, compute_ttc_xy_global


def parse_args():
    p = argparse.ArgumentParser(description="Build annotation-token TTC pickle")
    p.add_argument("--data-root", required=True, help="nuScenes root (contains maps/, v1.0-*, ...)")
    p.add_argument(
        "--version",
        default="v1.0-mini",
        choices=["v1.0-mini", "v1.0-trainval", "v1.0-test"],
        help="nuScenes version",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output pickle path (default: <root>/ttc_gt_labels_<version>.pkl)",
    )
    p.add_argument(
        "--flat",
        action="store_true",
        help="If set, pickle only the annotation_token -> fields dict (no metadata wrapper).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    labels = {}
    stats = {
        "n_samples": 0,
        "n_annotations": 0,
        "n_skipped_low_closing": 0,
        "n_skipped_nan_vel": 0,
        "n_skipped_zero_range": 0,
    }

    for sample in nusc.sample:
        stats["n_samples"] += 1
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = nusc.get("sample_data", lidar_token)
        pose = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        ego_xy = np.asarray(pose["translation"][:2], dtype=np.float64)

        for ann_token in sample["anns"]:
            stats["n_annotations"] += 1
            ann = nusc.get("sample_annotation", ann_token)
            obj_xy = np.asarray(ann["translation"][:2], dtype=np.float64)

            vel = nusc.box_velocity(ann_token)
            vel = np.asarray(vel, dtype=np.float64)
            if vel.shape[0] < 2 or not np.all(np.isfinite(vel[:2])):
                stats["n_skipped_nan_vel"] += 1
                continue
            vel_xy = vel[:2]

            ttc, dist, closing = compute_ttc_xy_global(
                obj_xy,
                ego_xy,
                vel_xy,
                ttc_cap=TTC_CAP_S,
                min_closing_speed=MIN_CLOSING_SPEED_M_S,
            )
            if ttc is None:
                if dist < 1e-6:
                    stats["n_skipped_zero_range"] += 1
                else:
                    stats["n_skipped_low_closing"] += 1
                continue

            labels[ann_token] = {
                "ttc": float(ttc),
                "distance_xy": float(dist),
                "closing_speed": float(closing),
                "sample_token": sample["token"],
                "category": ann["category_name"],
            }

    metadata = {
        "nuscenes_version": args.version,
        "data_root": osp.abspath(args.data_root),
        "ttc_cap_s": TTC_CAP_S,
        "min_closing_speed_m_s": MIN_CLOSING_SPEED_M_S,
        "coordinate_frame": "global_xy_velocity_global_xy",
        "stats": stats,
    }
    payload = {"labels": labels, "metadata": metadata}

    out_path = args.out
    if out_path is None:
        out_path = osp.join(args.data_root, f"ttc_gt_labels_{args.version.replace('.', '_')}.pkl")

    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    to_write = labels if args.flat else payload
    with open(out_path, "wb") as f:
        pickle.dump(to_write, f)

    print(f"Wrote {len(labels)} labeled annotations to {out_path}")
    print(f"Metadata stats: {stats}")
    if args.flat:
        print("(--flat) Pickle is only the token->record dict; ttc_heuristic_baseline.py accepts this format.")


if __name__ == "__main__":
    main()
