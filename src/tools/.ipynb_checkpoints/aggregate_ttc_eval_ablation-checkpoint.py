#!/usr/bin/env python3
"""Merge TTC eval artifacts under a checkpoint work dir into one ablation table (JSON + Markdown).

Reads when present:
  <work_dir>/eval_ttc_mlp/ttc_loss_metrics.json
  <work_dir>/eval_ttc_breakdown/ttc_breakdown.json
  <work_dir>/compare_ttc_scene_*/scene_ttc_compare.json  (one row per scene compare)

Writes:
  <work_dir>/ttc_eval_ablation.json
  <work_dir>/ttc_eval_ablation.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ablation] skip {path}: {e}", file=sys.stderr)
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work-dir",
        required=True,
        help="Directory containing eval_ttc_mlp/, eval_ttc_breakdown/, compare_ttc_scene_*/",
    )
    args = p.parse_args()
    wd = os.path.abspath(os.path.expanduser(args.work_dir))
    os.makedirs(wd, exist_ok=True)

    rows: list[dict] = []
    loss_p = os.path.join(wd, "eval_ttc_mlp", "ttc_loss_metrics.json")
    bd_p = os.path.join(wd, "eval_ttc_breakdown", "ttc_breakdown.json")
    loss = _load_json(loss_p)
    bd = _load_json(bd_p)

    if loss:
        rows.append(
            {
                "source": "eval_ttc_mlp (val subset)",
                "trained_mean_sum_loss_ttc_per_batch": loss.get("trained_mean_sum_loss_ttc_per_batch"),
                "baseline_mean_sum_loss_ttc_per_batch": loss.get("baseline_mean_sum_loss_ttc_per_batch"),
                "n_batches": loss.get("n_batches"),
                "max_batches_arg": loss.get("max_batches"),
                "ann_file": loss.get("ann_file"),
            }
        )

    if bd and bd.get("error"):
        rows.append(
            {
                "source": "eval_ttc_breakdown",
                "error": bd.get("error"),
                "n_batches": bd.get("n_batches"),
            }
        )
    elif bd and "overall" in bd:
        o = bd["overall"]
        rows.append(
            {
                "source": "eval_ttc_breakdown (matched GT, val subset)",
                "n_pairs": bd.get("n_pairs"),
                "n_batches": bd.get("n_batches"),
                "mae_s_mlp_vs_gt": o.get("mae_s"),
                "rmse_s": o.get("rmse_s"),
            }
        )

    scene_paths = sorted(glob.glob(os.path.join(wd, "compare_ttc_scene_*", "scene_ttc_compare.json")))
    scene_summaries: list[dict] = []
    for sp in scene_paths:
        data = _load_json(sp)
        if not data:
            continue
        st = data.get("scene_token") or os.path.basename(os.path.dirname(sp)).replace("compare_ttc_scene_", "")
        summ = data.get("summary") or {}
        scene_summaries.append(
            {
                "scene_token": st,
                "n_finite_gt": summ.get("n_finite_gt"),
                "mae_physics_s": summ.get("mae_physics_s"),
                "mae_mlp_s": summ.get("mae_mlp_s"),
                "median_ae_physics_s": summ.get("median_ae_physics_s"),
                "median_ae_mlp_s": summ.get("median_ae_mlp_s"),
                "path": sp,
            }
        )
        rows.append(
            {
                "source": f"compare_ttc_scene ({st[:8]}…)",
                "scene_token": st,
                "n_finite_gt": summ.get("n_finite_gt"),
                "mae_physics_s": summ.get("mae_physics_s"),
                "mae_mlp_s": summ.get("mae_mlp_s"),
            }
        )

    # Macro average over scene compares (MAE MLP / physics)
    macro = None
    if scene_summaries:
        mp = [s["mae_mlp_s"] for s in scene_summaries if s.get("mae_mlp_s") is not None]
        pp = [s["mae_physics_s"] for s in scene_summaries if s.get("mae_physics_s") is not None]
        macro = {
            "n_scenes": len(scene_summaries),
            "mean_mae_mlp_s_over_scenes": sum(mp) / len(mp) if mp else None,
            "mean_mae_physics_s_over_scenes": sum(pp) / len(pp) if pp else None,
        }
        rows.append({"source": "macro mean (scene compares only)", **macro})

    out = {
        "work_dir": wd,
        "loss_metrics_path": loss_p if loss else None,
        "breakdown_path": bd_p if bd else None,
        "scene_compare_json_glob": os.path.join(wd, "compare_ttc_scene_*", "scene_ttc_compare.json"),
        "n_scene_jsons": len(scene_paths),
        "macro_scene_average": macro,
        "table_rows": rows,
    }

    jp = os.path.join(wd, "ttc_eval_ablation.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[ablation] wrote {jp}")

    # Markdown table (compact)
    lines = [
        "# TTC evaluation ablation / summary",
        "",
        f"**Work dir:** `{wd}`",
        "",
        "| Source | Key metrics |",
        "|--------|-------------|",
    ]
    for r in rows:
        src = str(r.get("source", "")).replace("|", "\\|")
        parts = []
        for k, v in r.items():
            if k == "source":
                continue
            if v is None:
                continue
            parts.append(f"{k}={v}")
        lines.append(f"| {src} | {'; '.join(parts)} |")

    lines.extend(
        [
            "",
            "## Scene-level compares",
            "",
            "| scene_token | n_finite_gt | MAE physics (s) | MAE MLP (s) |",
            "|-------------|-------------|-----------------|---------------|",
        ]
    )
    for s in scene_summaries:
        lines.append(
            f"| `{s['scene_token']}` | {s.get('n_finite_gt')} | "
            f"{s.get('mae_physics_s')} | {s.get('mae_mlp_s')} |"
        )
    if not scene_summaries:
        lines.append("| *(none)* | | | |")

    mp = os.path.join(wd, "ttc_eval_ablation.md")
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ablation] wrote {mp}")


if __name__ == "__main__":
    main()
