#!/usr/bin/env python3
"""Per–annotation TTC comparison for one nuScenes scene: GT vs physics-from-bbox vs trained MLP.

Rows use the same Hungarian assignment as ``loss_ttc`` / ``ttc_pairwise_errors``.

- **GT** — TTC from your annotation pickle (and ``gt_ttc_matched`` in the loss pipeline).
- **Baseline (#2)** — **Physics TTC** from the **baseline detector** checkpoint (``--pretrained-baseline``):
  closing-time in BEV with ego at lidar origin, using **predicted** (cx, cy) and **predicted** (vx, vy)
  from the regression head — same recipe as ``tools/ttc_utils.compute_ttc_xy_global``, ego fixed at (0,0).
- **Trained (#3)** — TTC from the **Phase-3 MLP** (``--checkpoint``) on matched query features.

Requires the same GT TTC pickle as training (``generate_ttc_labels.py``) and temporal ``--ann-file``.

**Outputs** (under ``--save-dir``):

- ``scene_ttc_compare.json`` — full rows + ``summary`` (MAE, per-class).
- ``scene_ttc_compare.csv`` — full token strings (for scripts).
- ``scene_ttc_table_clean.csv`` — rounded seconds, short token prefixes (for spreadsheets).
- ``ttc_scatter_gt_vs_pred.png`` — GT vs MLP vs physics (diagonal = perfect).
- ``ttc_dot_tracks.png`` — per-row TTC dots (GT / physics / MLP).
- ``ttc_mae_by_class.png`` — mean absolute error by class.
- ``ttc_mae_by_gt_bin.png`` — MAE stratified by **GT TTC** range (urgent vs capped horizons).
- ``ttc_error_cdf.png`` — cumulative distribution of absolute error (MLP vs physics).
- ``ttc_residual_vs_gt.png`` — residual (pred − GT) vs GT to show systematic bias.
- ``ttc_project_summary.png`` — one-page **dashboard** (scatter + bins + CDF + metrics text).
- ``ttc_cam_front_panels.png`` — CAM_FRONT per ``sample_token`` (up to ``--max-cam-panels``) + TTC dot strip. Optional **3D box wireframes** projected to the image (``--cam-bbox-ttc`` colors edges by TTC; needs nuScenes + ``ann_token``).
- ``scene_ttc_front.mp4`` (with ``--video``) — **full-scene video** in temporal order: CAM_FRONT with TTC-colored 3D wireframes. Use ``--video-panels gt,physics,mlp`` for **side-by-side columns** (same frame, one color scheme per prediction source). See ``--video-fps``, ``--video-path``, ``--video-max-width``.

Use ``--no-plots`` to skip PNGs; ``--no-csv`` to skip CSVs. Camera panels need ``--data-root`` (or ``cfg.data_root``) pointing at the nuScenes folder that contains ``samples/``.

Example:
  python tools/compare_ttc_scene.py \\
    projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py \\
    work_dirs/streampetr_ttc_frozen_20e/iter_140650.pth \\
    --pretrained-baseline ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth \\
    --ann-file data/nuscenes/nuscenes2d_temporal_infos_val.pkl \\
    --scene-token <scene-xxx> \\
    --save-dir work_dirs/phase3/scene_compare
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import sys
from functools import partial
from typing import Optional

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
if os.environ.get("STREAMPETR_KEEP_CUDA_HOME") != "1":
    os.environ.pop("CUDA_HOME", None)

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.eval_ttc_breakdown import _as_list_len_bs, _unwrap_dc  # noqa: E402
from tools.eval_ttc_mlp import (  # noqa: E402
    _configure_logging,
    _ensure_cuda,
    _patch_load_gttc_in_pipeline,
    _resolve_ttc_pkl,
)


def _r4(x):
    if x is None:
        return ""
    try:
        v = float(x)
        if not np.isfinite(v):
            return ""
        return round(v, 4)
    except (TypeError, ValueError):
        return x


def _tok_short(s: str, n: int = 8) -> str:
    if not s:
        return ""
    s = str(s)
    return s[:n] + ("…" if len(s) > n else "")


def _write_clean_csv(path: str, merged: list[dict]) -> None:
    """Readable CSV: rounded seconds, short token prefixes, clear headers."""
    cols = [
        "idx",
        "frame_idx",
        "gt_idx",
        "class",
        "gt_s",
        "physics_s",
        "mlp_s",
        "abs_err_phys",
        "abs_err_mlp",
        "sample_id",
        "ann_id",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(merged, start=1):
            w.writerow(
                {
                    "idx": i,
                    "frame_idx": r.get("frame_idx", ""),
                    "gt_idx": r.get("gt_idx", ""),
                    "class": r.get("class_name", ""),
                    "gt_s": _r4(r.get("gt_ttc_matched")),
                    "physics_s": _r4(r.get("ttc_physics_from_bbox")),
                    "mlp_s": _r4(r.get("ttc_mlp")),
                    "abs_err_phys": _r4(r.get("abs_err_physics_bbox")),
                    "abs_err_mlp": _r4(r.get("abs_err_mlp")),
                    "sample_id": _tok_short(r.get("sample_token") or "", 8),
                    "ann_id": _tok_short(r.get("ann_token") or "", 8),
                }
            )


def _summary_stats(merged: list[dict]) -> dict:
    finite = [
        r
        for r in merged
        if r.get("gt_ttc_matched") is not None and np.isfinite(r["gt_ttc_matched"])
    ]

    def mae_cls(key: str, rows: list) -> Optional[float]:
        xs = [
            abs(r[key] - r["gt_ttc_matched"])
            for r in rows
            if r.get(key) is not None
        ]
        return float(np.mean(xs)) if xs else None

    def mae(key: str) -> Optional[float]:
        xs = [
            abs(r[key] - r["gt_ttc_matched"])
            for r in finite
            if r.get(key) is not None and np.isfinite(r["gt_ttc_matched"])
        ]
        return float(np.mean(xs)) if xs else None

    def median_ae(key: str) -> Optional[float]:
        xs = [
            abs(r[key] - r["gt_ttc_matched"])
            for r in finite
            if r.get(key) is not None
        ]
        return float(np.median(xs)) if xs else None

    by_class: dict = {}
    for r in finite:
        c = r.get("class_name") or "unknown"
        by_class.setdefault(c, []).append(r)

    per_class = {}
    for c, rows in by_class.items():
        per_class[c] = {
            "n": len(rows),
            "mae_physics_s": mae_cls("ttc_physics_from_bbox", rows),
            "mae_mlp_s": mae_cls("ttc_mlp", rows),
        }

    return {
        "n_rows": len(merged),
        "n_finite_gt": len(finite),
        "mae_physics_s": mae("ttc_physics_from_bbox"),
        "mae_mlp_s": mae("ttc_mlp"),
        "median_ae_physics_s": median_ae("ttc_physics_from_bbox"),
        "median_ae_mlp_s": median_ae("ttc_mlp"),
        "n_both_preds": sum(
            1
            for r in finite
            if r.get("ttc_physics_from_bbox") is not None and r.get("ttc_mlp") is not None
        ),
        "per_class": per_class,
    }


# Colorblind-friendly palette (Okabe–Ito style) for slides / reports
_COL_MLP = "#009E73"
_COL_PHYS = "#0072B2"
_COL_GT = "#000000"


def _presentation_style():
    """Matplotlib defaults for clearer figures (slides, posters, PDF)."""
    import matplotlib as mpl

    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            mpl.style.use(name)
            break
        except OSError:
            continue
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.color": "#cccccc",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
        }
    )


def _save_plots(save_dir: str, merged: list[dict], scene_token: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _presentation_style()
    except Exception as exc:
        print(f"[compare_ttc_scene] skip plots: {exc}")
        return

    if not merged:
        return

    summary = _summary_stats(merged)

    # Align arrays for scatter (only rows with finite GT)
    rows_f = [
        r
        for r in merged
        if r.get("gt_ttc_matched") is not None and np.isfinite(r["gt_ttc_matched"])
    ]
    gtv = np.array([float(r["gt_ttc_matched"]) for r in rows_f], dtype=np.float64)
    mlpv = np.array(
        [
            float(r["ttc_mlp"])
            if r.get("ttc_mlp") is not None
            else np.nan
            for r in rows_f
        ],
        dtype=np.float64,
    )
    physv = np.array(
        [
            float(r["ttc_physics_from_bbox"])
            if r.get("ttc_physics_from_bbox") is not None
            else np.nan
            for r in rows_f
        ],
        dtype=np.float64,
    )

    if gtv.size == 0:
        return

    pred_stack = np.r_[mlpv, physv]
    mx_pred = float(np.nanmax(pred_stack)) if np.any(np.isfinite(pred_stack)) else 0.0
    lo, hi = 0.0, max(11.0, float(np.nanmax(gtv)) * 1.05, mx_pred * 1.05)

    # --- 1) GT vs pred scatter (dots) + y=x ---
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=120)
    ax.plot([lo, hi], [lo, hi], color=_COL_GT, ls="--", lw=1.1, alpha=0.55, label="Ideal (y = x)")
    m_mlp = np.isfinite(mlpv)
    m_ph = np.isfinite(physv)
    ax.scatter(
        gtv[m_mlp],
        mlpv[m_mlp],
        s=44,
        c=_COL_MLP,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.4,
        label="MLP (Phase 3)",
        zorder=3,
    )
    ax.scatter(
        gtv[m_ph],
        physv[m_ph],
        s=40,
        c=_COL_PHYS,
        alpha=0.88,
        marker="s",
        edgecolors="white",
        linewidths=0.35,
        label="Physics (bbox + vel)",
        zorder=3,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground-truth TTC (s)")
    ax.set_ylabel("Predicted TTC (s)")
    ax.set_title(f"Calibration — scene {_tok_short(scene_token, 12)}")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    p1 = os.path.join(save_dir, "ttc_scatter_gt_vs_pred.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p1}")

    # --- 2) Per-row track: horizontal position = TTC, rows = objects ---
    n = len(rows_f)
    if n > 0:
        y = np.arange(n, dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8, max(4.0, 0.18 * n + 1)), dpi=120)
        ax.scatter(gtv, y, s=46, c=_COL_GT, marker="o", label="GT", zorder=4, edgecolors="white", linewidths=0.3)
        if np.any(m_ph):
            ax.scatter(
                physv[m_ph],
                y[m_ph] + 0.12,
                s=38,
                c=_COL_PHYS,
                marker="s",
                alpha=0.92,
                label="Physics",
                zorder=3,
                edgecolors="white",
                linewidths=0.3,
            )
        if np.any(m_mlp):
            ax.scatter(
                mlpv[m_mlp],
                y[m_mlp] - 0.12,
                s=38,
                c=_COL_MLP,
                marker="^",
                alpha=0.92,
                label="MLP",
                zorder=3,
                edgecolors="white",
                linewidths=0.3,
            )
        ax.set_xlabel("TTC (s)")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{i+1}" for i in range(n)], fontsize=7)
        ax.set_ylabel("Object row (see clean CSV)")
        ax.set_title("Per-object TTC (GT vs physics vs MLP)")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(-0.5, hi)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        p2 = os.path.join(save_dir, "ttc_dot_tracks.png")
        fig.savefig(p2, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p2}")

    # --- 3) Mean |error| by class (where pred exists) ---
    classes = sorted({r.get("class_name") or "?" for r in rows_f})
    mae_p = []
    mae_m = []
    for c in classes:
        sub = [r for r in rows_f if (r.get("class_name") or "?") == c]
        xs_p = [
            abs(r["ttc_physics_from_bbox"] - r["gt_ttc_matched"])
            for r in sub
            if r.get("ttc_physics_from_bbox") is not None
        ]
        xs_m = [
            abs(r["ttc_mlp"] - r["gt_ttc_matched"])
            for r in sub
            if r.get("ttc_mlp") is not None
        ]
        mae_p.append(float(np.mean(xs_p)) if xs_p else float("nan"))
        mae_m.append(float(np.mean(xs_m)) if xs_m else float("nan"))

    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(7.5, len(classes) * 0.95), 4.4), dpi=120)
    ax.bar(x - w / 2, mae_p, width=w, label="Physics", color=_COL_PHYS, edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, mae_m, width=w, label="MLP", color=_COL_MLP, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=28, ha="right")
    ax.set_ylabel("Mean absolute error (s)")
    ax.set_title("TTC error by object class")
    ax.legend(framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    p3 = os.path.join(save_dir, "ttc_mae_by_class.png")
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p3}")

    # --- 4) MAE by GT TTC bin (urgent vs safe horizons) ---
    bin_defs = [(0.0, 3.0), (3.0, 5.0), (5.0, 7.0), (7.0, 10.01)]
    bin_labels = ["[0, 3)", "[3, 5)", "[5, 7)", "[7, 10]"]
    mae_bin_p, mae_bin_m, n_bin = [], [], []
    for bi, (lo_b, hi_b) in enumerate(bin_defs):
        if bi == len(bin_defs) - 1:
            idx = (gtv >= lo_b) & (gtv <= 10.01)
        else:
            idx = (gtv >= lo_b) & (gtv < hi_b)
        if not np.any(idx):
            mae_bin_p.append(float("nan"))
            mae_bin_m.append(float("nan"))
            n_bin.append(0)
            continue
        gp = gtv[idx]
        pp = physv[idx]
        mm = mlpv[idx]
        ae_p = np.abs(pp - gp)[np.isfinite(pp)]
        ae_m = np.abs(mm - gp)[np.isfinite(mm)]
        mae_bin_p.append(float(np.mean(ae_p)) if ae_p.size else float("nan"))
        mae_bin_m.append(float(np.mean(ae_m)) if ae_m.size else float("nan"))
        n_bin.append(int(np.sum(idx)))

    xb = np.arange(len(bin_labels))
    wb = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=120)
    ax.bar(xb - wb / 2, mae_bin_p, width=wb, label="Physics", color=_COL_PHYS, edgecolor="white", linewidth=0.5)
    ax.bar(xb + wb / 2, mae_bin_m, width=wb, label="MLP", color=_COL_MLP, edgecolor="white", linewidth=0.5)
    ax.set_xticks(xb)
    ax.set_xticklabels([f"{lbl}\nn={n_bin[i]}" for i, lbl in enumerate(bin_labels)], fontsize=9)
    ax.set_ylabel("Mean absolute error (s)")
    ax.set_xlabel("Ground-truth TTC bin")
    ax.set_title("Error by urgency (GT horizon)")
    ax.legend(framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    p4 = os.path.join(save_dir, "ttc_mae_by_gt_bin.png")
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p4}")

    # --- 5) CDF of absolute errors ---
    ae_phys_all = np.abs(physv[m_ph] - gtv[m_ph]) if np.any(m_ph) else np.array([])
    ae_mlp_all = np.abs(mlpv[m_mlp] - gtv[m_mlp]) if np.any(m_mlp) else np.array([])
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=120)
    if ae_phys_all.size:
        s_p = np.sort(ae_phys_all)
        ax.plot(s_p, np.linspace(0, 1, len(s_p), endpoint=True), color=_COL_PHYS, lw=2.2, label="Physics")
    if ae_mlp_all.size:
        s_m = np.sort(ae_mlp_all)
        ax.plot(s_m, np.linspace(0, 1, len(s_m), endpoint=True), color=_COL_MLP, lw=2.2, label="MLP")
    ax.set_xlabel("Absolute error |pred − GT| (s)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Error distribution (CDF)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    p5 = os.path.join(save_dir, "ttc_error_cdf.png")
    fig.savefig(p5, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p5}")

    # --- 6) Residual vs GT (bias check) ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=120)
    if np.any(m_mlp):
        ax.scatter(
            gtv[m_mlp],
            mlpv[m_mlp] - gtv[m_mlp],
            s=42,
            c=_COL_MLP,
            alpha=0.85,
            label="MLP residual",
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
    if np.any(m_ph):
        ax.scatter(
            gtv[m_ph],
            physv[m_ph] - gtv[m_ph],
            s=36,
            c=_COL_PHYS,
            alpha=0.78,
            marker="s",
            label="Physics residual",
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )
    ax.axhline(0.0, color=_COL_GT, ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel("Ground-truth TTC (s)")
    ax.set_ylabel("Residual = pred − GT (s)")
    ax.set_title('Systematic bias: positive → overestimate TTC (too "safe")')
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    p6 = os.path.join(save_dir, "ttc_residual_vs_gt.png")
    fig.savefig(p6, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p6}")

    # --- 7) One-page summary for reports / slides ---
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(11.5, 9.0), dpi=120)
    fig.subplots_adjust(hspace=0.36, wspace=0.32)
    ax0.plot([lo, hi], [lo, hi], color=_COL_GT, ls="--", lw=1, alpha=0.55)
    if np.any(m_mlp):
        ax0.scatter(gtv[m_mlp], mlpv[m_mlp], s=32, c=_COL_MLP, alpha=0.85, edgecolors="white", linewidths=0.3)
    if np.any(m_ph):
        ax0.scatter(gtv[m_ph], physv[m_ph], s=28, c=_COL_PHYS, alpha=0.82, marker="s", edgecolors="white", linewidths=0.25)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("GT TTC (s)")
    ax0.set_ylabel("Pred TTC (s)")
    ax0.set_title("Calibration")
    ax0.grid(True, alpha=0.35)

    ax1.bar(xb - wb / 2, mae_bin_p, width=wb, label="Physics", color=_COL_PHYS, edgecolor="white", linewidth=0.4)
    ax1.bar(xb + wb / 2, mae_bin_m, width=wb, label="MLP", color=_COL_MLP, edgecolor="white", linewidth=0.4)
    ax1.set_xticks(xb)
    ax1.set_xticklabels(bin_labels, rotation=0, fontsize=9)
    ax1.set_ylabel("MAE (s)")
    ax1.set_title("MAE by GT horizon")
    ax1.legend(fontsize=8, framealpha=0.95)
    ax1.grid(True, axis="y", alpha=0.35)

    if ae_phys_all.size:
        ax2.plot(
            np.sort(ae_phys_all),
            np.linspace(0, 1, len(ae_phys_all), endpoint=True),
            color=_COL_PHYS,
            lw=2,
            label="Physics",
        )
    if ae_mlp_all.size:
        ax2.plot(
            np.sort(ae_mlp_all),
            np.linspace(0, 1, len(ae_mlp_all), endpoint=True),
            color=_COL_MLP,
            lw=2,
            label="MLP",
        )
    ax2.set_xlabel("|err| (s)")
    ax2.set_ylabel("CDF")
    ax2.set_title("Absolute error distribution")
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.35)

    ax3.axis("off")
    mp = summary.get("mae_physics_s")
    mm = summary.get("mae_mlp_s")
    medp = summary.get("median_ae_physics_s")
    medm = summary.get("median_ae_mlp_s")
    nf = summary.get("n_finite_gt", 0)
    lines = [
        "TTC prediction — scene summary",
        "",
        f"Scene token (prefix): {_tok_short(scene_token, 16)}",
        f"Matched rows (finite GT): {nf}",
        "",
        f"Mean |error| — Physics: {mp:.4f} s" if mp is not None else "Mean |error| — Physics: n/a",
        f"Mean |error| — MLP:     {mm:.4f} s" if mm is not None else "Mean |error| — MLP:     n/a",
        "",
        f"Median |error| — Physics: {medp:.4f} s" if medp is not None else "Median |error| — Physics: n/a",
        f"Median |error| — MLP:     {medm:.4f} s" if medm is not None else "Median |error| — MLP:     n/a",
        "",
        "Lower MAE is better. Compare MAE in [0,3) s bins",
        "to see urgent-risk performance.",
    ]
    ax3.text(
        0.04,
        0.96,
        "\n".join(lines),
        transform=ax3.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#cccccc", linewidth=0.8),
    )
    fig.suptitle(f"TTC project summary — {_tok_short(scene_token, 14)}", fontsize=14, fontweight="600", y=0.98)
    p7 = os.path.join(save_dir, "ttc_project_summary.png")
    fig.savefig(p7, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p7}")


def _resolve_cam_image_path(data_root: str, rel: str) -> Optional[str]:
    """Resolve ``cams['CAM_FRONT']['data_path']`` against nuScenes ``data_root``."""
    if not rel:
        return None
    rel = str(rel).replace("\\", os.sep)
    dr = os.path.abspath(os.path.expanduser(data_root))
    candidates = []
    if os.path.isabs(rel):
        candidates.append(rel)
    candidates.append(os.path.join(dr, rel))
    if "samples" in rel:
        i = rel.replace("\\", "/").find("samples/")
        if i >= 0:
            tail = rel[i:].replace("\\", os.sep)
            candidates.append(os.path.join(dr, tail))
    candidates.append(os.path.join(dr, os.path.basename(rel)))
    candidates.append(os.path.abspath(os.path.join(os.getcwd(), rel)))
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.normpath(p)
    return None


def _load_rgb_image(path: str):
    """Return RGB float/uint8 array for ``imshow``, or None."""
    try:
        import mmcv

        im = mmcv.imread(path)
        if im is None:
            return None
        return mmcv.bgr2rgb(im)
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt

        return plt.imread(path)
    except Exception:
        return None


# nuScenes ``Box.corners()`` indices — 12 edges of the oriented 3D cuboid (camera frame).
_NUSCENES_BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _get_nuscenes_box_cam_front(nusc, sample_token: str, ann_token: str):
    """Return ``(box, camera_intrinsic_3x3)`` in CAM_FRONT coordinates, or None."""
    try:
        from pyquaternion import Quaternion
    except Exception:
        return None
    try:
        s_rec = nusc.get("sample", sample_token)
        if ann_token not in s_rec["anns"]:
            return None
        cam_token = s_rec["data"]["CAM_FRONT"]
        sd_rec = nusc.get("sample_data", cam_token)
        cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        camera_intrinsic = np.array(cs_rec["camera_intrinsic"], dtype=np.float64)
        box = nusc.get_box(ann_token)
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)
        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)
        return box, camera_intrinsic
    except Exception:
        return None


def _corners_uv_cam_front(box, camera_intrinsic: np.ndarray) -> Optional[np.ndarray]:
    """Project 8 cuboid corners to pixel ``uv`` (2×8). NaN if corner is behind the camera."""
    try:
        from nuscenes.utils.geometry_utils import view_points
    except Exception:
        return None
    corners_3d = box.corners()
    uv = np.full((2, 8), np.nan, dtype=np.float64)
    for i in range(8):
        if corners_3d[2, i] <= 1e-4:
            continue
        p = view_points(corners_3d[:, i : i + 1], camera_intrinsic, normalize=True)
        uv[0, i] = float(p[0, 0])
        uv[1, i] = float(p[1, 0])
    if not np.any(np.isfinite(uv)):
        return None
    return uv


def _nuscenes_cam_front_bbox_xyxy(
    nusc,
    sample_token: str,
    ann_token: str,
    imsize_wh: tuple[int, int],
) -> Optional[tuple[float, float, float, float]]:
    """Axis-aligned 2D bounds of the projected 3D box (legacy helper)."""
    del imsize_wh  # bounds from full corner projection
    out = _get_nuscenes_box_cam_front(nusc, sample_token, ann_token)
    if out is None:
        return None
    box, K = out
    uv = _corners_uv_cam_front(box, K)
    if uv is None:
        return None
    m = np.isfinite(uv[0]) & np.isfinite(uv[1])
    if not np.any(m):
        return None
    return (
        float(np.min(uv[0, m])),
        float(np.min(uv[1, m])),
        float(np.max(uv[0, m])),
        float(np.max(uv[1, m])),
    )


def _draw_wireframe_cv2(bgr: np.ndarray, uv: np.ndarray, color_bgr: tuple, linewidth: int = 2) -> None:
    """Draw 3D box wireframe (12 edges) on BGR image."""
    import cv2

    for ia, ib in _NUSCENES_BOX_EDGES:
        pa, pb = uv[:, ia], uv[:, ib]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        cv2.line(
            bgr,
            (int(round(pa[0])), int(round(pa[1]))),
            (int(round(pb[0])), int(round(pb[1]))),
            color_bgr,
            linewidth,
            cv2.LINE_AA,
        )


def _draw_wireframe_mpl(ax_img, uv: np.ndarray, rgba: tuple, linewidth: float = 2.2) -> None:
    """Draw 3D box wireframe on matplotlib image axes (pixel coords, origin top-left)."""
    for ia, ib in _NUSCENES_BOX_EDGES:
        pa, pb = uv[:, ia], uv[:, ib]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        ax_img.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            color=rgba,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=10,
        )


def _label_anchor_uv(uv: np.ndarray) -> tuple[float, float]:
    """Top-left-ish anchor for TTC text (min u, min v among visible corners)."""
    m = np.isfinite(uv[0]) & np.isfinite(uv[1])
    if not np.any(m):
        return 0.0, 0.0
    return float(np.min(uv[0, m])), float(np.min(uv[1, m]))


def _ttc_value_for_cam_color(r: dict, field: str) -> Optional[float]:
    if field == "gt":
        key = "gt_ttc_matched"
    elif field == "mlp":
        key = "ttc_mlp"
    elif field == "physics":
        key = "ttc_physics_from_bbox"
    else:
        return None
    v = r.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _draw_cam_front_ttc_boxes(
    ax_img,
    nusc,
    sample_token: str,
    sub: list[dict],
    bbox_ttc_field: str,
    ttc_max: float,
    imsize_wh: tuple[int, int],
    cmap,
    norm,
) -> None:
    """Draw **3D cuboid wireframes** (12 edges) on CAM_FRONT, colored by TTC (seconds)."""
    if nusc is None or bbox_ttc_field == "none":
        return
    del imsize_wh
    for r in sub:
        ann = r.get("ann_token")
        if not ann:
            continue
        out = _get_nuscenes_box_cam_front(nusc, sample_token, str(ann))
        if out is None:
            continue
        box, K = out
        uv = _corners_uv_cam_front(box, K)
        if uv is None:
            continue
        ttc = _ttc_value_for_cam_color(r, bbox_ttc_field)
        if ttc is None:
            ttc = _ttc_value_for_cam_color(r, "gt")
        if ttc is None:
            continue
        ttc_c = float(np.clip(ttc, 0.0, ttc_max))
        rgba = cmap(norm(ttc_c))
        _draw_wireframe_mpl(ax_img, uv, rgba, linewidth=2.4)
        lx, ly = _label_anchor_uv(uv)
        ty = max(2.0, ly - 4.0)
        ax_img.text(
            lx,
            ty,
            f"{ttc_c:.1f}s",
            color="white",
            fontsize=7,
            fontweight="bold",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=rgba[:3], edgecolor="white", linewidth=0.4, alpha=0.92),
        )


def _save_camera_ttc_panels(
    save_dir: str,
    merged: list[dict],
    ann_file: str,
    data_root: str,
    scene_token: str,
    max_panels: int = 6,
    *,
    bbox_ttc_field: str = "mlp",
    ttc_max_color: float = 10.0,
) -> None:
    """Multi-row figure: CAM_FRONT | per-frame TTC dots (same markers as dot tracks)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mmcv

        _presentation_style()
    except Exception as exc:
        print(f"[compare_ttc_scene] skip camera panels: {exc}")
        return

    if not merged or max_panels <= 0:
        return

    try:
        data = mmcv.load(ann_file)
    except Exception as exc:
        print(f"[compare_ttc_scene] skip camera panels (cannot load ann_file): {exc}")
        return

    infos = data.get("infos") or []
    token_to_info = {str(e.get("token")): e for e in infos if e.get("token")}

    seen: list[str] = []
    for r in merged:
        st = r.get("sample_token")
        if st and st not in seen:
            seen.append(str(st))

    panels = [st for st in seen if st in token_to_info][:max_panels]
    if not panels:
        print(
            "[compare_ttc_scene] camera panels: no sample_token from merged rows "
            "matched ann_file infos (check ann_file / sample_token fields)"
        )
        return

    from matplotlib import cm
    from matplotlib.colors import Normalize

    cmap = cm.get_cmap("turbo")
    norm = Normalize(vmin=0.0, vmax=float(ttc_max_color), clip=True)

    nusc = None
    if bbox_ttc_field != "none":
        try:
            ver = (data.get("metadata") or {}).get("version", "v1.0-mini")
            from nuscenes.nuscenes import NuScenes

            nusc = NuScenes(version=ver, dataroot=data_root, verbose=False)
        except Exception as exc:
            print(f"[compare_ttc_scene] TTC-colored boxes skipped (nuScenes): {exc}")

    n = len(panels)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(12.5, max(3.2, 3.15 * n)),
        dpi=120,
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )
    hi_global = 11.0
    for r in merged:
        g = r.get("gt_ttc_matched")
        if g is not None and np.isfinite(g):
            hi_global = max(hi_global, float(g) * 1.05)
        for k in ("ttc_mlp", "ttc_physics_from_bbox"):
            v = r.get(k)
            if v is not None and np.isfinite(v):
                hi_global = max(hi_global, float(v) * 1.05)

    for row, st in enumerate(panels):
        if n == 1:
            ax_img, ax_tt = axes[0], axes[1]
        else:
            ax_img, ax_tt = axes[row, 0], axes[row, 1]

        info = token_to_info[st]
        cams = info.get("cams") or {}
        cf = cams.get("CAM_FRONT") or {}
        rel = cf.get("data_path", "")
        path = _resolve_cam_image_path(data_root, str(rel) if rel is not None else "")

        sub = [r for r in merged if str(r.get("sample_token")) == st]
        k = len(sub)

        if path:
            rgb = _load_rgb_image(path)
            if rgb is not None:
                ax_img.imshow(rgb)
                imsize_wh = (int(rgb.shape[1]), int(rgb.shape[0]))
                _draw_cam_front_ttc_boxes(
                    ax_img,
                    nusc,
                    st,
                    sub,
                    bbox_ttc_field,
                    ttc_max_color,
                    imsize_wh,
                    cmap,
                    norm,
                )
            else:
                ax_img.text(
                    0.5,
                    0.5,
                    f"unreadable:\n{path}",
                    ha="center",
                    va="center",
                    transform=ax_img.transAxes,
                    fontsize=8,
                )
                ax_img.set_facecolor("0.85")
        else:
            ax_img.text(
                0.5,
                0.5,
                f"file not found\n(rel={rel})",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax_img.transAxes,
            )
            ax_img.set_facecolor("0.9")
        mae_mlp_line = ""
        if k:
            mae_rows = [
                abs(float(r["ttc_mlp"]) - float(r["gt_ttc_matched"]))
                for r in sub
                if r.get("ttc_mlp") is not None
                and r.get("gt_ttc_matched") is not None
                and np.isfinite(r["gt_ttc_matched"])
            ]
            if mae_rows:
                mae_mlp_line = f"\nmean |MLP−GT| = {float(np.mean(mae_rows)):.2f}s (n={len(mae_rows)})"
        ax_img.set_title(f"CAM_FRONT — {_tok_short(st, 10)}{mae_mlp_line}", fontsize=9)
        ax_img.axis("off")

        if k == 0:
            ax_tt.text(0.5, 0.5, "no rows", ha="center", va="center", transform=ax_tt.transAxes)
            ax_tt.axis("off")
            continue

        y = np.arange(k, dtype=np.float64)
        gtv = np.array(
            [
                float(r["gt_ttc_matched"])
                if r.get("gt_ttc_matched") is not None and np.isfinite(r["gt_ttc_matched"])
                else np.nan
                for r in sub
            ],
            dtype=np.float64,
        )
        physv = np.array(
            [
                float(r["ttc_physics_from_bbox"])
                if r.get("ttc_physics_from_bbox") is not None
                and np.isfinite(r["ttc_physics_from_bbox"])
                else np.nan
                for r in sub
            ],
            dtype=np.float64,
        )
        mlpv = np.array(
            [
                float(r["ttc_mlp"])
                if r.get("ttc_mlp") is not None and np.isfinite(r["ttc_mlp"])
                else np.nan
                for r in sub
            ],
            dtype=np.float64,
        )

        ax_tt.scatter(gtv, y, s=38, c=_COL_GT, marker="o", label="GT", zorder=4, edgecolors="white", linewidths=0.25)
        m_ph = np.isfinite(physv)
        m_mlp = np.isfinite(mlpv)
        if np.any(m_ph):
            ax_tt.scatter(
                physv[m_ph],
                y[m_ph] + 0.12,
                s=32,
                c=_COL_PHYS,
                marker="s",
                alpha=0.92,
                label="Physics",
                zorder=3,
                edgecolors="white",
                linewidths=0.25,
            )
        if np.any(m_mlp):
            ax_tt.scatter(
                mlpv[m_mlp],
                y[m_mlp] - 0.12,
                s=32,
                c=_COL_MLP,
                marker="^",
                alpha=0.92,
                label="MLP",
                zorder=3,
                edgecolors="white",
                linewidths=0.25,
            )

        labels = [f"{i + 1} {(r.get('class_name') or '?')[:14]}" for i, r in enumerate(sub)]
        ax_tt.set_yticks(y)
        ax_tt.set_yticklabels(labels, fontsize=7)
        ax_tt.set_xlabel("TTC (s)")
        ax_tt.set_xlim(-0.5, hi_global)
        ax_tt.grid(True, axis="x", alpha=0.3)
        if row == 0:
            ax_tt.legend(loc="upper right", fontsize=7)

    img_axes = [axes[i, 0] for i in range(n)] if n > 1 else [axes[0]]
    if nusc is not None and bbox_ttc_field != "none":
        from matplotlib.cm import ScalarMappable

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=img_axes,
            orientation="horizontal",
            fraction=0.035,
            pad=0.06,
            aspect=38,
        )
        cbar.set_label(
            f"2D box color = {bbox_ttc_field} TTC (s)   (colormap 0–{ttc_max_color:g} s)",
            fontsize=9,
        )

    fig.suptitle(f"CAM_FRONT + TTC dots — scene {_tok_short(scene_token, 12)}", y=1.01)
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.98])
    outp = os.path.join(save_dir, "ttc_cam_front_panels.png")
    fig.savefig(outp, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outp}")


def parse_args():
    p = argparse.ArgumentParser(description="Per-object TTC: GT vs physics-from-bbox vs trained MLP (one scene)")
    p.add_argument("config", help="Phase 3 config")
    p.add_argument("checkpoint", help="Trained Phase-3 checkpoint (.pth)")
    p.add_argument(
        "--pretrained-baseline",
        required=True,
        help="StreamPETR detector checkpoint used to compute **physics** TTC from predicted boxes/velocities",
    )
    p.add_argument("--ann-file", required=True, help="Temporal infos pkl")
    p.add_argument("--scene-token", required=True, help="nuScenes scene token (e.g. scene-0061)")
    p.add_argument("--ttc-pkl", default=None, help="GT TTC pickle (default: same as eval_ttc_mlp)")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument(
        "--save-dir",
        default=None,
        help="Write scene_ttc_compare.csv and scene_ttc_compare.json (default: cwd/compare_ttc_scene)",
    )
    p.add_argument("--no-csv", action="store_true")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG figures (calibration, CDF, dashboard, camera panels, etc.).",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="nuScenes data root (folder containing samples/). Sets dataset LoadGTTC + camera PNGs; "
        "default: cfg.data_root / ./data/nuscenes/",
    )
    p.add_argument(
        "--max-cam-panels",
        type=int,
        default=6,
        help="Max number of sample_token rows in ttc_cam_front_panels.png (camera + TTC dots).",
    )
    p.add_argument(
        "--cam-bbox-ttc",
        choices=("none", "gt", "mlp", "physics"),
        default="mlp",
        help="Color each CAM_FRONT 2D box edge by this TTC value (turbo colormap, 0–cam-ttc-max s). "
        "Uses nuScenes to project 3D boxes; needs non-empty ann_token on rows. "
        "Default: mlp. Use 'none' to disable overlays.",
    )
    p.add_argument(
        "--cam-ttc-max",
        type=float,
        default=10.0,
        help="Upper end of the TTC colormap for camera box colors (seconds).",
    )
    p.add_argument(
        "--video",
        action="store_true",
        help="Write scene_ttc_front.mp4: all CAM_FRONT frames in the scene (time order) with TTC-colored boxes.",
    )
    p.add_argument(
        "--video-fps",
        type=float,
        default=10.0,
        help="FPS for --video (default 10).",
    )
    p.add_argument(
        "--video-path",
        default=None,
        help="Output path for --video (default: save_dir/scene_ttc_front.mp4).",
    )
    p.add_argument(
        "--video-max-width",
        type=int,
        default=1280,
        help="Max width: single-column video uses this for the whole frame; "
        "with --video-panels, each column is scaled to about (this / N) px wide.",
    )
    p.add_argument(
        "--video-panels",
        default=None,
        help="Side-by-side compare video: comma-separated sources "
        "gt | physics | mlp | none (e.g. gt,physics,mlp). "
        "Each column uses that TTC for wireframe color. Default: one column from --cam-bbox-ttc.",
    )
    p.add_argument(
        "--samples-per-gpu",
        type=int,
        default=None,
        help="Override cfg.data.samples_per_gpu for this scene dataloader. "
        "Use **1** on 8–12GB GPUs to avoid OOM (default: from config, often 2).",
    )
    return p.parse_args()


def _patch_meta_keys_sample_idx(node) -> int:
    """Ensure ``sample_idx`` is collected into ``img_metas`` (needed for annotation tokens)."""
    n = 0
    if isinstance(node, (list, tuple)):
        for item in node:
            n += _patch_meta_keys_sample_idx(item)
        return n
    if isinstance(node, dict):
        if node.get("type") == "Collect3D":
            mk = node.get("meta_keys")
            if mk and "sample_idx" not in mk:
                node["meta_keys"] = tuple(list(mk) + ["sample_idx"])
                return 1
        for k in ("pipeline", "transforms", "img_transform"):
            if k in node:
                n += _patch_meta_keys_sample_idx(node[k])
        return n
    return 0


def _patch_data_train_ann(cfg, ann_file: str, ttc_pkl: str, data_root: str | None = None):
    ann_file = os.path.abspath(ann_file)
    ttc_pkl = os.path.abspath(ttc_pkl)
    dr = os.path.abspath(os.path.expanduser(data_root)) if data_root else None
    train_cfg = copy.deepcopy(cfg.data.train)
    if hasattr(train_cfg, "to_dict"):
        train_cfg = train_cfg.to_dict()
    elif not isinstance(train_cfg, dict):
        train_cfg = dict(train_cfg)
    train_cfg["ann_file"] = ann_file
    if dr is not None:
        train_cfg["data_root"] = dr
    pl = train_cfg.get("pipeline")
    if not pl:
        raise RuntimeError("cfg.data.train has no pipeline")
    # eval_ttc_mlp older builds: 3-arg only (no data_root on LoadGTTC).
    try:
        n = _patch_load_gttc_in_pipeline(pl, ann_file, ttc_pkl, dr)
    except TypeError:
        n = _patch_load_gttc_in_pipeline(pl, ann_file, ttc_pkl)
    if n == 0:
        raise RuntimeError("No LoadGTTC in pipeline; use a Phase-3 TTC config.")
    _patch_meta_keys_sample_idx(pl)
    return train_cfg


def _build_load_gttc_mapper(cfg, ann_file: str, ttc_pkl: str, data_root: str | None = None):
    from projects.mmdet3d_plugin.datasets.pipelines.ttc_pipeline import LoadGTTC

    train_cfg = copy.deepcopy(cfg.data.train)
    if hasattr(train_cfg, "to_dict"):
        train_cfg = train_cfg.to_dict()
    pl = train_cfg["pipeline"]
    dr = os.path.abspath(os.path.expanduser(data_root)) if data_root else None
    for step in pl:
        if isinstance(step, dict) and step.get("type") == "LoadGTTC":
            d = copy.deepcopy(step)
            d.pop("type", None)
            d["ann_file"] = os.path.abspath(ann_file)
            d["ttc_pkl"] = os.path.abspath(ttc_pkl)
            if dr is not None:
                d["data_root"] = dr
            return LoadGTTC(**d)
    raise RuntimeError("LoadGTTC not found in cfg.data.train.pipeline")


def _scene_indices(dataset, scene_token: str) -> list[int]:
    out = []
    for i in range(len(dataset)):
        if dataset.data_infos[i].get("scene_token") == scene_token:
            out.append(i)
    return out


def _ordered_sample_tokens(dataset, idxs: list[int]) -> list[str]:
    """Scene sample tokens sorted by nuScenes ``timestamp`` (temporal playback order)."""
    pairs: list[tuple[int, str]] = []
    for i in idxs:
        info = dataset.data_infos[i]
        ts = info.get("timestamp")
        if ts is None:
            ts = 0
        pairs.append((int(ts), str(info["token"])))
    pairs.sort(key=lambda x: x[0])
    return [p[1] for p in pairs]


def _draw_ttc_boxes_cv2_bgr(
    bgr: np.ndarray,
    nusc,
    sample_token: str,
    sub: list[dict],
    bbox_ttc_field: str,
    ttc_max: float,
    proj_wh: tuple[int, int],
    cmap,
    norm,
) -> None:
    """Draw TTC-colored **3D cuboid wireframes** on a BGR image (in-place).

    ``proj_wh`` is ``(W, H)`` of the **native** CAM_FRONT JPEG that nuScenes intrinsics ``K``
    refer to. If ``bgr`` was resized (e.g. ``--video-max-width``), corner ``uv`` are scaled
    to the display size. Calling ``Box.render_cv2`` on a resized image is invalid and is
    skipped in that case (wireframe only).
    """
    if nusc is None or bbox_ttc_field == "none":
        return
    import cv2

    pw, ph = int(proj_wh[0]), int(proj_wh[1])
    if pw <= 0 or ph <= 0:
        return
    disp_w, disp_h = int(bgr.shape[1]), int(bgr.shape[0])
    use_native_render = abs(disp_w - pw) <= 1 and abs(disp_h - ph) <= 1

    for r in sub:
        ann = r.get("ann_token")
        if not ann:
            continue
        out = _get_nuscenes_box_cam_front(nusc, sample_token, str(ann))
        if out is None:
            continue
        box, K = out
        uv = _corners_uv_cam_front(box, K)
        if uv is None:
            continue
        ttc = _ttc_value_for_cam_color(r, bbox_ttc_field)
        if ttc is None:
            ttc = _ttc_value_for_cam_color(r, "gt")
        if ttc is None:
            continue
        ttc_c = float(np.clip(ttc, 0.0, ttc_max))
        rgba = cmap(norm(ttc_c))
        color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        if use_native_render:
            try:
                box.render_cv2(bgr, K, normalize=True, colors=color, linewidth=2)
                uv_draw = uv
            except Exception:
                _draw_wireframe_cv2(bgr, uv, color, linewidth=2)
                uv_draw = uv
        else:
            su = disp_w / float(pw)
            sv = disp_h / float(ph)
            uv_draw = uv.copy()
            uv_draw[0, :] *= su
            uv_draw[1, :] *= sv
            _draw_wireframe_cv2(bgr, uv_draw, color, linewidth=2)
        lx, ly = _label_anchor_uv(uv_draw)
        label = f"{ttc_c:.1f}s"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, int(ly) - th - 4)
        cv2.rectangle(bgr, (int(lx), ty), (int(lx) + tw + 3, int(ly)), color, -1)
        cv2.putText(
            bgr,
            label,
            (int(lx) + 1, int(ly) - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _video_panel_label(field: str) -> str:
    return {
        "gt": "GT TTC",
        "physics": "Physics TTC (bbox + vel)",
        "mlp": "MLP TTC (Phase 3)",
        "none": "Camera (no boxes)",
    }.get(field, field)


def _add_video_panel_title_bar(bgr: np.ndarray, title: str, *, bar_h: int = 44) -> np.ndarray:
    """Dark bar + title above a BGR panel (for side-by-side compare)."""
    import cv2

    _w = bgr.shape[1]
    bar = np.zeros((bar_h, _w, 3), dtype=np.uint8)
    bar[:] = (26, 26, 26)
    cv2.putText(
        bar,
        title,
        (10, bar_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, bgr])


def _save_scene_ttc_video(
    merged: list[dict],
    full_ds,
    scene_idxs: list[int],
    ann_file: str,
    data_root: str,
    scene_token: str,
    *,
    video_path: str,
    fps: float,
    bbox_ttc_field: str,
    ttc_max_color: float,
    max_width: int,
    panel_fields: list[str] | None = None,
) -> None:
    """Encode temporal CAM_FRONT sequence + TTC boxes to MP4 (OpenCV).

    If ``panel_fields`` has multiple entries, each frame is ``cv2.hconcat`` of
    one view per source (gt / physics / mlp / none), same colormap for TTC.
    """
    try:
        import mmcv
        import cv2
        from matplotlib import cm
        from matplotlib.colors import Normalize
    except ImportError as exc:
        print(f"[compare_ttc_scene] --video skipped (import): {exc}")
        return

    try:
        data = mmcv.load(ann_file)
    except Exception as exc:
        print(f"[compare_ttc_scene] --video skipped (ann_file): {exc}")
        return

    infos = data.get("infos") or []
    token_to_info = {str(e.get("token")): e for e in infos if e.get("token")}
    order = _ordered_sample_tokens(full_ds, scene_idxs)
    cmap = cm.get_cmap("turbo")
    norm = Normalize(vmin=0.0, vmax=float(ttc_max_color), clip=True)

    fields: list[str]
    if panel_fields:
        fields = list(panel_fields)
    else:
        fields = [bbox_ttc_field]
    n_p = len(fields)
    if n_p == 0:
        fields = ["none"]

    need_nusc = any(f != "none" for f in fields)
    nusc = None
    if need_nusc:
        try:
            ver = (data.get("metadata") or {}).get("version", "v1.0-mini")
            from nuscenes.nuscenes import NuScenes

            nusc = NuScenes(version=ver, dataroot=data_root, verbose=False)
        except Exception as exc:
            print(f"[compare_ttc_scene] video: nuScenes load failed, boxes may be missing: {exc}")

    frames: list[np.ndarray] = []
    for st in order:
        if st not in token_to_info:
            continue
        info = token_to_info[st]
        cf = (info.get("cams") or {}).get("CAM_FRONT") or {}
        rel = cf.get("data_path", "")
        path = _resolve_cam_image_path(data_root, str(rel) if rel is not None else "")
        if not path:
            continue
        bgr0 = mmcv.imread(path)
        if bgr0 is None:
            continue
        h0, w0 = bgr0.shape[:2]

        if n_p <= 1:
            bgr = bgr0
            if max_width > 0 and w0 > max_width:
                scale = max_width / float(w0)
                nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
                bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            sub = [r for r in merged if str(r.get("sample_token")) == st]
            fld = fields[0]
            if fld != "none":
                _draw_ttc_boxes_cv2_bgr(
                    bgr, nusc, st, sub, fld, ttc_max_color, (w0, h0), cmap, norm
                )
            frames.append(bgr)
            continue

        per_w = max(280, max_width // n_p) if max_width > 0 else w0
        if max_width > 0 and w0 > per_w:
            scale = per_w / float(w0)
            nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
            bgr0 = cv2.resize(bgr0, (nw, nh), interpolation=cv2.INTER_AREA)

        sub = [r for r in merged if str(r.get("sample_token")) == st]
        cols: list[np.ndarray] = []
        for fld in fields:
            pane = bgr0.copy()
            if fld != "none":
                _draw_ttc_boxes_cv2_bgr(
                    pane, nusc, st, sub, fld, ttc_max_color, (w0, h0), cmap, norm
                )
            pane = _add_video_panel_title_bar(pane, _video_panel_label(fld))
            cols.append(pane)

        mh = min(c.shape[0] for c in cols)
        mw = min(c.shape[1] for c in cols)
        cols_resized = [cv2.resize(c, (mw, mh), interpolation=cv2.INTER_AREA) for c in cols]
        row = cv2.hconcat(cols_resized)
        frames.append(row)

    if not frames:
        print("[compare_ttc_scene] --video: no frames; check CAM_FRONT paths and --data-root")
        return

    h, w = frames[0].shape[:2]
    fps_i = max(1, int(round(float(fps))))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, float(fps_i), (w, h))
    if not out.isOpened():
        print(f"[compare_ttc_scene] --video: could not open VideoWriter for {video_path!r}")
        return
    for fr in frames:
        if fr.shape[0] != h or fr.shape[1] != w:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        out.write(fr)
    out.release()
    mode = f"{n_p}-panel compare" if n_p > 1 else "single"
    print(f"Wrote video ({mode}, {len(frames)} frames @ {fps_i} fps): {video_path}")


def _collect_scene_rows(
    model,
    batches: list,
    device_id: int,
    mapper,
    class_names: list[str],
    labels_pkl: dict,
    *,
    physics: bool,
) -> list[dict]:
    """One forward over cached batches. ``physics=True``: bbox-based closing TTC; else TTC MLP."""
    head = model.module.pts_bbox_head
    det = model.module
    num_frame_losses = int(det.num_frame_losses)
    rows: list[dict] = []

    model.train()
    with torch.no_grad():
        for data in batches:
            data = scatter(data, [device_id])[0]
            # StreamPETR keeps temporal state on the head; a prior batch can leave memory at a
            # different batch size (e.g. last batch size 1 vs next size 2) → temporal_alignment crash.
            if hasattr(head, "reset_memory"):
                head.reset_memory()
            outs_frames: list = []

            def _hook(_m, _inp, out):
                outs_frames.append(out)

            handle = det.pts_bbox_head.register_forward_hook(_hook)
            try:
                model(return_loss=True, **data)
            finally:
                handle.remove()

            T = int(data["img"].size(1))
            start_f = max(0, T - num_frame_losses)
            for i in range(start_f, T):
                if i >= len(outs_frames):
                    break
                outs = outs_frames[i]
                if outs is None:
                    continue
                if physics:
                    if outs.get("all_bbox_preds") is None:
                        continue
                elif outs.get("ttc_query_feats") is None:
                    continue
                gt_ttc_i = data["gt_ttc"][i] if "gt_ttc" in data else None
                if gt_ttc_i is None:
                    continue
                bs = int(outs["all_cls_scores"][-1].size(0))
                gt_ttc_list = _as_list_len_bs(gt_ttc_i, bs)
                gtb = _unwrap_dc(data["gt_bboxes_3d"][i])
                gtl = _unwrap_dc(data["gt_labels_3d"][i])
                gt_bl = _as_list_len_bs(gtb, bs)
                gt_ll = _as_list_len_bs(gtl, bs)
                imetas = _unwrap_dc(data["img_metas"][i])
                if physics:
                    if not hasattr(head, "ttc_pairwise_physics_from_preds"):
                        raise RuntimeError("StreamPETRHead needs ttc_pairwise_physics_from_preds")
                    pw = head.ttc_pairwise_physics_from_preds(
                        outs,
                        gt_bl,
                        gt_ll,
                        gt_ttc_list,
                        return_gt_indices=True,
                    )
                else:
                    pw = head.ttc_pairwise_errors(
                        outs,
                        gt_bl,
                        gt_ll,
                        gt_ttc_list,
                        return_gt_indices=True,
                    )
                if pw is None:
                    continue
                pred = pw["pred"].numpy()
                tgt = pw["tgt"].numpy()
                cls_ids = pw["cls"].numpy()
                b_idx = pw["batch_idx"].numpy()
                g_idx = pw["gt_idx"].numpy()
                for k in range(pred.shape[0]):
                    b = int(b_idx[k])
                    gix = int(g_idx[k])
                    meta_b = _unwrap_dc(imetas[b])
                    if isinstance(meta_b, (list, tuple)) and len(meta_b) > 0:
                        meta_b = meta_b[0]
                    if not isinstance(meta_b, dict):
                        raise TypeError(f"img_metas[{i}][{b}] expected dict-like meta, got {type(meta_b)}")
                    sample_token = meta_b.get("sample_idx")
                    if sample_token is None:
                        raise KeyError(
                            "img_metas missing sample_idx; ensure Collect3D meta_keys includes sample_idx "
                            "(compare_ttc_scene patches this)."
                        )
                    gtb_b = gt_bl[b]
                    gtl_b = gt_ll[b]
                    ann_tokens = mapper.map_gt_to_ann_tokens(sample_token, gtb_b, gtl_b)
                    ann_tok = ann_tokens[gix] if gix < len(ann_tokens) else None
                    gt_from_pkl = None
                    if ann_tok and ann_tok in labels_pkl:
                        gt_from_pkl = float(labels_pkl[ann_tok]["ttc"])
                    row = {
                        "frame_idx": i,
                        "sample_token": sample_token,
                        "gt_idx": int(gix),
                        "ann_token": ann_tok or "",
                        "class_name": class_names[int(cls_ids[k])] if cls_ids[k] < len(class_names) else str(cls_ids[k]),
                        "gt_ttc_matched": float(tgt[k]),
                        "gt_ttc_pkl": gt_from_pkl,
                    }
                    if physics:
                        row["ttc_physics_from_bbox"] = float(pred[k])
                    else:
                        row["ttc_mlp"] = float(pred[k])
                    rows.append(row)
    return rows


def _load_labels_flat(ttc_pkl: str) -> dict:
    import pickle

    with open(ttc_pkl, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "labels" in payload:
        return payload["labels"]
    return payload


def _resolved_nuscenes_root(cfg, args_data_root: str | None) -> str:
    """Absolute nuScenes root for camera paths and fallbacks (cfg + defaults)."""
    dr = args_data_root
    if not dr:
        dr = getattr(cfg, "data_root", None)
    if not dr and hasattr(cfg, "data") and hasattr(cfg.data, "train"):
        dr = getattr(cfg.data.train, "data_root", None)
    if not dr:
        dr = "./data/nuscenes/"
    return os.path.abspath(os.path.expanduser(str(dr)))


def main():
    args = parse_args()
    _configure_logging()
    _ensure_cuda(args.gpu_id)

    cfg = Config.fromfile(args.config)
    cfg.gpu_ids = [args.gpu_id]

    ttc_pkl = _resolve_ttc_pkl(args.ann_file, args.ttc_pkl)
    print(f"GT TTC pickle: {ttc_pkl}")
    labels_flat = _load_labels_flat(ttc_pkl)

    data_root_override = (
        os.path.abspath(os.path.expanduser(args.data_root)) if args.data_root else None
    )
    data_cfg = _patch_data_train_ann(cfg, args.ann_file, ttc_pkl, data_root=data_root_override)
    # Plugin datasets (CustomNuScenesDataset) live in mmdet's DATASETS, not mmdet3d's builder.
    import projects.mmdet3d_plugin.datasets  # noqa: F401 — register CustomNuScenesDataset
    from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset
    from mmdet3d.models import build_model
    from torch.utils.data import DataLoader

    full_ds = custom_build_dataset(data_cfg)
    idxs = _scene_indices(full_ds, args.scene_token)
    if not idxs:
        raise SystemExit(f"No samples found for scene_token={args.scene_token!r} in {args.ann_file}")

    print(f"Scene {args.scene_token!r}: {len(idxs)} dataset indices (temporal samples)")

    subset = torch.utils.data.Subset(full_ds, idxs)
    # Subset lacks ``dataset.flag``; StreamPETR's InfiniteGroupEachSampleInBatchSampler also assumes
    # enough sequence groups per batch. Use plain DataLoader + mmcv collate for this filtered eval.
    spg = int(args.samples_per_gpu) if args.samples_per_gpu is not None else int(cfg.data.samples_per_gpu)
    if spg < 1:
        raise SystemExit("--samples-per-gpu must be >= 1")
    if args.samples_per_gpu is not None:
        print(f"Using samples_per_gpu={spg} (overrides cfg {cfg.data.samples_per_gpu}) for this scene run.")
    # Subset can be any length: last batch can be < spg; collate handles it.
    data_loader = DataLoader(
        subset,
        batch_size=spg,
        shuffle=False,
        num_workers=int(cfg.data.workers_per_gpu),
        collate_fn=partial(collate, samples_per_gpu=spg),
        pin_memory=False,
    )

    class_names = list(cfg.class_names) if hasattr(cfg, "class_names") and cfg.class_names else []
    if not class_names:
        class_names = [
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

    mapper = _build_load_gttc_mapper(cfg, args.ann_file, ttc_pkl, data_root=data_root_override)

    def fresh_model():
        m = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
        m.init_weights()
        return MMDataParallel(m.cuda(args.gpu_id), device_ids=[args.gpu_id])

    print("Caching batches (DataLoader is single-pass per model run)...")
    batches_cpu: list = []
    for data in data_loader:
        batches_cpu.append(data)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=== Baseline: physics TTC from predicted bbox (lidar BEV, ego at origin) ===")
    model_b = fresh_model()
    load_checkpoint(model_b.module, args.pretrained_baseline, map_location="cpu", strict=False)
    rows_b = _collect_scene_rows(
        model_b, batches_cpu, args.gpu_id, mapper, class_names, labels_flat, physics=True
    )
    del model_b
    torch.cuda.empty_cache()

    print("=== Trained: TTC MLP (Phase-3) ===")
    model_t = fresh_model()
    load_checkpoint(model_t.module, args.checkpoint, map_location="cpu", strict=False)
    rows_t = _collect_scene_rows(
        model_t, batches_cpu, args.gpu_id, mapper, class_names, labels_flat, physics=False
    )

    # Join on (frame_idx, sample_token, gt_idx). Do not use ann_token alone: multiple GTs can
    # share empty ann_token + same class after Hungarian mapping, which used to collapse dict rows.
    keyfn = lambda r: (r["frame_idx"], r["sample_token"], r["gt_idx"])
    by_b = {keyfn(r): r for r in rows_b}
    by_t = {keyfn(r): r for r in rows_t}
    keys = sorted(set(by_b.keys()) | set(by_t.keys()))

    merged: list[dict] = []
    for k in keys:
        rb = by_b.get(k)
        rt = by_t.get(k)
        base = (rb or rt).copy()
        base["ttc_physics_from_bbox"] = float(rb["ttc_physics_from_bbox"]) if rb and "ttc_physics_from_bbox" in rb else None
        base["ttc_mlp"] = float(rt["ttc_mlp"]) if rt and "ttc_mlp" in rt else None
        if "gt_ttc_matched" in base:
            gtm = base["gt_ttc_matched"]
            pb = base.get("ttc_physics_from_bbox")
            pm = base.get("ttc_mlp")
            base["abs_err_physics_bbox"] = abs(pb - gtm) if pb is not None and np.isfinite(gtm) else None
            base["abs_err_mlp"] = abs(pm - gtm) if pm is not None and np.isfinite(gtm) else None
        merged.append(base)

    save_dir = args.save_dir or os.path.join(os.getcwd(), "compare_ttc_scene")
    os.makedirs(save_dir, exist_ok=True)
    summary = _summary_stats(merged)
    json_path = os.path.join(save_dir, "scene_ttc_compare.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene_token": args.scene_token,
                "ann_file": os.path.abspath(args.ann_file),
                "ttc_pkl": os.path.abspath(ttc_pkl),
                "pretrained_baseline": os.path.abspath(args.pretrained_baseline),
                "checkpoint": os.path.abspath(args.checkpoint),
                "n_rows_physics_from_bbox": len(rows_b),
                "n_rows_mlp": len(rows_t),
                "n_rows_merged": len(merged),
                "summary": summary,
                "rows": merged,
            },
            f,
            indent=2,
        )
    print(f"Wrote {json_path}")

    if not args.no_csv and merged:
        csv_path = os.path.join(save_dir, "scene_ttc_compare.csv")
        cols = [
            "frame_idx",
            "sample_token",
            "gt_idx",
            "ann_token",
            "class_name",
            "gt_ttc_matched",
            "gt_ttc_pkl",
            "ttc_physics_from_bbox",
            "ttc_mlp",
            "abs_err_physics_bbox",
            "abs_err_mlp",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in merged:
                w.writerow({c: r.get(c) for c in cols})
        print(f"Wrote {csv_path} (full tokens)")

        clean_path = os.path.join(save_dir, "scene_ttc_table_clean.csv")
        _write_clean_csv(clean_path, merged)
        print(f"Wrote {clean_path} (rounded, short ids — use for spreadsheets)")

    if merged and not args.no_plots:
        print("Writing figures…")
        _save_plots(save_dir, merged, args.scene_token)
        data_root = _resolved_nuscenes_root(cfg, args.data_root)
        _save_camera_ttc_panels(
            save_dir,
            merged,
            args.ann_file,
            data_root,
            args.scene_token,
            max_panels=args.max_cam_panels,
            bbox_ttc_field=args.cam_bbox_ttc,
            ttc_max_color=args.cam_ttc_max,
        )

    if merged and args.video:
        data_root_v = _resolved_nuscenes_root(cfg, args.data_root)
        vpanels: list[str] | None = None
        if args.video_panels:
            vpanels = [x.strip().lower() for x in args.video_panels.split(",") if x.strip()]
            allowed = {"gt", "physics", "mlp", "none"}
            bad = [x for x in vpanels if x not in allowed]
            if bad:
                raise SystemExit(
                    f"--video-panels: invalid {bad!r}; allowed: gt, physics, mlp, none (comma-separated)"
                )
        video_out = args.video_path
        if not video_out:
            video_out = os.path.join(
                save_dir,
                "scene_ttc_front_compare.mp4" if vpanels and len(vpanels) > 1 else "scene_ttc_front.mp4",
            )
        _save_scene_ttc_video(
            merged,
            full_ds,
            idxs,
            args.ann_file,
            data_root_v,
            args.scene_token,
            video_path=os.path.abspath(video_out),
            fps=args.video_fps,
            bbox_ttc_field=args.cam_bbox_ttc,
            ttc_max_color=args.cam_ttc_max,
            max_width=int(args.video_max_width),
            panel_fields=vpanels,
        )

    finite = [r for r in merged if r.get("gt_ttc_matched") is not None and np.isfinite(r["gt_ttc_matched"])]
    if finite:
        def mae(key_pred):
            xs = [
                abs(r[key_pred] - r["gt_ttc_matched"])
                for r in finite
                if r.get(key_pred) is not None
            ]
            return float(np.mean(xs)) if xs else float("nan")

        print(
            f"Scene MAE |pred-GT|: physics-from-bbox={mae('ttc_physics_from_bbox'):.4f}s  "
            f"mlp={mae('ttc_mlp'):.4f}s  (n={len(finite)} merged rows with finite GT)"
        )


if __name__ == "__main__":
    main()
