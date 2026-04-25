#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Bird's-eye-view PNGs: ego at origin, objects colored by GT TTC (label pickle).

Optionally combine with the six nuScenes camera keyframes (2×3 grid + BEV).

Example (BEV only):
  python tools/visualize_ttc_bev.py \\
    --data-root ./data/nuscenes \\
    --ttc-labels ./data/nuscenes/ttc_gt_labels_v1_0_mini.pkl \\
    --out-dir ./work_dirs/ttc_bev_gt

Example (cameras left, TTC BEV right):
  python tools/visualize_ttc_bev.py ... --with-cameras

Example (one keyframe per listed **scene** — qualitative table):
  python tools/visualize_ttc_bev.py \\
    --data-root ./data/nuscenes --ttc-labels ./data/nuscenes/ttc_gt_labels_v1_0_mini.pkl \\
    --version v1.0-mini --out-dir ./work_dirs/ttc_bev_pick --with-cameras \\
    --scene-tokens scene-0061,scene-0103
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
import sys

_ROOT = osp.dirname(osp.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# nuScenes keyframe order: two rows × three columns
CAM_GRID = [
    ("CAM_FRONT_LEFT", "FL"),
    ("CAM_FRONT", "F"),
    ("CAM_FRONT_RIGHT", "FR"),
    ("CAM_BACK_LEFT", "BL"),
    ("CAM_BACK", "B"),
    ("CAM_BACK_RIGHT", "BR"),
]


def ttc_to_rgba(ttc: float) -> tuple:
    """Red < 3s, yellow 3–6s, green > 6s (aligned with Phase 5 plan)."""
    if ttc < 3.0:
        return (0.9, 0.15, 0.15, 1.0)
    if ttc < 6.0:
        return (0.95, 0.85, 0.2, 1.0)
    return (0.2, 0.75, 0.35, 1.0)


def collect_bev_points(sample, boxes, labels: dict, show_no_ttc: bool):
    xs_l, ys_l, cs_l = [], [], []
    xs_u, ys_u = [], []
    assert len(sample["anns"]) == len(boxes), (len(sample["anns"]), len(boxes))
    for ann_token, box in zip(sample["anns"], boxes):
        x, y, _ = box.center
        if ann_token in labels:
            xs_l.append(x)
            ys_l.append(y)
            cs_l.append(ttc_to_rgba(labels[ann_token]["ttc"]))
        elif show_no_ttc:
            xs_u.append(x)
            ys_u.append(y)
    return xs_l, ys_l, cs_l, xs_u, ys_u


def draw_bev_on_ax(
    ax,
    token: str,
    range_m: float,
    xs_l,
    ys_l,
    cs_l,
    xs_u,
    ys_u,
    title: str | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if xs_u:
        ax.scatter(xs_u, ys_u, c="#c0c0c0", s=12, alpha=0.7, label="no TTC label")
    if xs_l:
        ax.scatter(
            xs_l,
            ys_l,
            c=cs_l,
            s=36,
            edgecolors="black",
            linewidths=0.4,
            label="GT TTC",
        )

    ax.scatter([0.0], [0.0], c="blue", s=80, marker="^", zorder=5, label="ego")
    ax.plot([0.0, 8.0], [0.0, 0.0], "b-", linewidth=2, alpha=0.8)

    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m] LiDAR / ego forward")
    ax.set_ylabel("y [m] LiDAR / ego left")
    ax.set_title(title or f"BEV GT TTC (sample {token[:8]}…)")

    handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="blue", markersize=9, label="ego"),
    ]
    if xs_u:
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0c0c0", markersize=7, label="no TTC label")
        )
    handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ttc_to_rgba(1.0), markersize=8, label="TTC < 3 s"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ttc_to_rgba(4.0), markersize=8, label="TTC 3–6 s"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ttc_to_rgba(8.0), markersize=8, label="TTC > 6 s"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7)


def parse_args():
    p = argparse.ArgumentParser(description="Export BEV plots colored by TTC")
    p.add_argument("--data-root", required=True)
    p.add_argument("--ttc-labels", required=True, help="Pickle from generate_ttc_labels.py")
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--out-dir", default="./work_dirs/ttc_bev_gt")
    p.add_argument("--max-samples", type=int, default=80, help="0 = all samples")
    p.add_argument(
        "--range-m",
        type=float,
        default=50.0,
        help="Plot half-extent in meters around ego (square BEV).",
    )
    p.add_argument(
        "--show-no-ttc",
        action="store_true",
        help="Plot objects without a TTC label (min closing speed, etc.) in light gray.",
    )
    p.add_argument(
        "--with-cameras",
        action="store_true",
        help="Save a wide figure: 2×3 camera grid + BEV (filename *_ttc_panel.png).",
    )
    p.add_argument(
        "--also-bev-only",
        action="store_true",
        help="When --with-cameras, also write the BEV-only PNG (*_ttc_bev.png).",
    )
    p.add_argument(
        "--sample-tokens",
        default=None,
        help="Comma-separated **sample** tokens to render (overrides default loop).",
    )
    p.add_argument(
        "--scene-tokens",
        default=None,
        help="Comma-separated **scene** tokens; render keyframes from each scene. "
        "By default only the **first** keyframe per scene; use --scene-all-keyframes for every keyframe.",
    )
    p.add_argument(
        "--scene-all-keyframes",
        action="store_true",
        help="With --scene-tokens, walk the full scene (can be many PNGs).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from nuscenes.nuscenes import NuScenes

    with open(args.ttc_labels, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "labels" in payload:
        labels = payload["labels"]
    else:
        labels = payload

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    os.makedirs(args.out_dir, exist_ok=True)

    def _iter_samples_from_scenes(scene_tok_list, all_keyframes: bool):
        for st in scene_tok_list:
            st = st.strip()
            if not st:
                continue
            sc = nusc.get("scene", st)
            tok = sc["first_sample_token"]
            while tok:
                yield nusc.get("sample", tok)
                if not all_keyframes:
                    break
                nxt = nusc.get("sample", tok).get("next", "")
                tok = nxt if nxt else ""

    if args.sample_tokens:
        toks = [t.strip() for t in args.sample_tokens.split(",") if t.strip()]
        samples = [nusc.get("sample", t) for t in toks]
    elif args.scene_tokens:
        scene_list = [s for s in args.scene_tokens.split(",") if s.strip()]
        samples = list(_iter_samples_from_scenes(scene_list, args.scene_all_keyframes))
    else:
        samples = nusc.sample
        if args.max_samples > 0:
            samples = samples[: args.max_samples]

    grid_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for sample in samples:
        token = sample["token"]
        lidar = sample["data"]["LIDAR_TOP"]
        _, boxes, _ = nusc.get_sample_data(lidar)
        xs_l, ys_l, cs_l, xs_u, ys_u = collect_bev_points(sample, boxes, labels, args.show_no_ttc)

        if args.with_cameras:
            fig = plt.figure(figsize=(18, 7), dpi=120)
            gs = GridSpec(2, 4, figure=fig, width_ratios=[1.0, 1.0, 1.0, 1.05], wspace=0.06, hspace=0.12)

            for (cam_name, short), (gr, gc) in zip(CAM_GRID, grid_positions):
                ax_c = fig.add_subplot(gs[gr, gc])
                cam_path = nusc.get_sample_data_path(sample["data"][cam_name])
                ax_c.imshow(mpimg.imread(cam_path))
                ax_c.set_title(short, fontsize=10, fontweight="bold")
                ax_c.axis("off")

            ax_bev = fig.add_subplot(gs[:, 3])
            draw_bev_on_ax(
                ax_bev,
                token,
                args.range_m,
                xs_l,
                ys_l,
                cs_l,
                xs_u,
                ys_u,
                title=f"BEV GT TTC ({token[:8]}…)",
            )
            fig.suptitle(f"nuScenes sample {token}", fontsize=11, y=1.02)
            out_png = osp.join(args.out_dir, f"{token}_ttc_panel.png")
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)

            if args.also_bev_only:
                fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
                draw_bev_on_ax(ax, token, args.range_m, xs_l, ys_l, cs_l, xs_u, ys_u)
                fig.tight_layout()
                fig.savefig(osp.join(args.out_dir, f"{token}_ttc_bev.png"))
                plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
            draw_bev_on_ax(ax, token, args.range_m, xs_l, ys_l, cs_l, xs_u, ys_u)
            fig.tight_layout()
            fig.savefig(osp.join(args.out_dir, f"{token}_ttc_bev.png"))
            plt.close(fig)

    suffix = "panel+cams" if args.with_cameras else "bev"
    print(f"Wrote {len(samples)} image(s) ({suffix}) to {args.out_dir}")


if __name__ == "__main__":
    main()
