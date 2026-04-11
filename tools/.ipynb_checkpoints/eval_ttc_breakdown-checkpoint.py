#!/usr/bin/env python3
"""Per–GT-object TTC accuracy (MAE / RMSE / counts) by nuScenes class and by GT TTC bin.

Uses the same matcher and TTC head as training (``StreamPETRHead.loss_ttc``). Run on a temporal
``--ann-file`` with GT TTC pickle, same as ``eval_ttc_mlp.py``.

Example:
  python tools/eval_ttc_breakdown.py \\
    projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py \\
    work_dirs/streampetr_ttc_frozen_20e/latest.pth \\
    --ann-file data/nuscenes/nuscenes2d_temporal_infos_val.pkl \\
    --max-batches 50

Writes ``eval_ttc_breakdown/`` under the checkpoint directory: ``ttc_breakdown.json`` and
``ttc_breakdown_by_class.png`` (if matplotlib is available).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
if os.environ.get("STREAMPETR_KEEP_CUDA_HOME") != "1":
    os.environ.pop("CUDA_HOME", None)

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer, MMDataParallel, scatter
from mmcv.runner import load_checkpoint

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.eval_ttc_mlp import (  # noqa: E402
    _configure_logging,
    _ensure_cuda,
    _patch_data_train_ann,
    _resolve_ttc_pkl,
)


def parse_args():
    p = argparse.ArgumentParser(description="TTC MAE/RMSE by class and GT-TTC bin (Phase 3)")
    p.add_argument("config", help="Phase 3 config")
    p.add_argument("checkpoint", help="Checkpoint .pth")
    p.add_argument("--ann-file", required=True, help="Temporal infos pkl")
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--ttc-pkl", default=None)
    p.add_argument(
        "--save-dir",
        default=None,
        help="Default: <checkpoint_dir>/eval_ttc_breakdown",
    )
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def _accumulate_class_names(cfg) -> list[str]:
    if hasattr(cfg, "class_names") and cfg.class_names:
        return list(cfg.class_names)
    return [
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


def _stats(pred: np.ndarray, tgt: np.ndarray) -> dict:
    err = pred - tgt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"n": int(pred.shape[0]), "mae_s": mae, "rmse_s": rmse}


def _unwrap_dc(x):
    """Unwrap mmcv DataContainer nesting (scatter + forward leave DC on some keys)."""
    while isinstance(x, DataContainer):
        x = x.data
    return x


def _as_list_len_bs(x, bs: int):
    """Match GT batch structure to ``bs`` (``all_cls_scores`` batch dim). Pad with None; truncate if longer."""
    x = _unwrap_dc(x)
    if x is None:
        return [None] * bs
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:
            return [x] * bs
        n = x.size(0)
        if n == bs:
            return [x[i] for i in range(bs)]
        if n < bs:
            return [x[i] for i in range(n)] + [None] * (bs - n)
        return [x[i] for i in range(bs)]
    if isinstance(x, (list, tuple)):
        lst = list(x)
    else:
        lst = [x]
    if len(lst) == bs:
        return lst
    if len(lst) < bs:
        return lst + [None] * (bs - len(lst))
    return lst[:bs]


def run_breakdown(model, data_loader, device_id: int, class_names: list[str], max_batches: int):
    """Collect pred/tgt/class across loss frames; return dict of aggregates."""
    head = model.module.pts_bbox_head
    if not hasattr(head, "ttc_pairwise_errors"):
        raise RuntimeError("pts_bbox_head has no ttc_pairwise_errors (need StreamPETR head).")
    det = model.module
    num_frame_losses = int(det.num_frame_losses)

    pred_all: list[np.ndarray] = []
    tgt_all: list[np.ndarray] = []
    cls_all: list[np.ndarray] = []

    model.train()
    n_batch = 0
    with torch.no_grad():
        for bi, data in enumerate(data_loader):
            if bi >= max_batches:
                break
            data = scatter(data, [device_id])[0]
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
                if outs is None or outs.get("ttc_query_feats") is None:
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
                pw = head.ttc_pairwise_errors(outs, gt_bl, gt_ll, gt_ttc_list)
                if pw is None:
                    continue
                pred_all.append(pw["pred"].numpy())
                tgt_all.append(pw["tgt"].numpy())
                cls_all.append(pw["cls"].numpy())

            n_batch += 1

    if not pred_all:
        return {"error": "no TTC pairs collected", "n_batches": n_batch}

    pred = np.concatenate(pred_all)
    tgt = np.concatenate(tgt_all)
    cls = np.concatenate(cls_all)
    overall = _stats(pred, tgt)

    by_class: dict = {}
    n_cls = len(class_names)
    for c in range(n_cls):
        m = cls == c
        if not np.any(m):
            by_class[class_names[c]] = {"n": 0, "mae_s": None, "rmse_s": None}
        else:
            by_class[class_names[c]] = _stats(pred[m], tgt[m])

    # GT TTC bins (seconds): [0,1), [1,3), [3,10), [10, inf)
    bins = [(0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 1e9)]
    by_gt_bin: dict = {}
    for lo, hi in bins:
        m = (tgt >= lo) & (tgt < hi)
        key = f"[{lo:g},{hi:g})" if hi < 100 else f"[{lo:g},inf)"
        if not np.any(m):
            by_gt_bin[key] = {"n": 0, "mae_s": None, "rmse_s": None}
        else:
            by_gt_bin[key] = _stats(pred[m], tgt[m])

    return {
        "n_batches": n_batch,
        "n_pairs": int(pred.shape[0]),
        "overall": overall,
        "by_class": by_class,
        "by_gt_ttc_bin_s": by_gt_bin,
    }


def _plot_by_class(save_path: str, by_class: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    names = sorted(by_class.keys(), key=lambda k: (by_class[k].get("n") or 0), reverse=True)
    maes = [by_class[k]["mae_s"] for k in names]
    ns = [by_class[k].get("n") or 0 for k in names]
    labels = [f"{k}\n(n={ns[i]})" for i, k in enumerate(names)]
    vals = [0.0 if m is None else m for m in maes]
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    x = np.arange(len(names))
    ax.bar(x, vals, color="#2d6a4f", edgecolor="0.2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("MAE |pred − GT| (s)")
    ax.set_title("TTC error by nuScenes class (matched GT)")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    _configure_logging()
    _ensure_cuda(args.gpu_id)

    cfg = Config.fromfile(args.config)
    cfg.gpu_ids = [args.gpu_id]

    import projects.mmdet3d_plugin.datasets  # noqa: F401 — register CustomNuScenesDataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader, custom_build_dataset
    from mmdet3d.models import build_model

    ttc_pkl = _resolve_ttc_pkl(args.ann_file, args.ttc_pkl)
    print(f"Using GT TTC pickle: {ttc_pkl}")
    data_cfg = _patch_data_train_ann(cfg, args.ann_file, ttc_pkl)
    dataset = custom_build_dataset(data_cfg)
    seed = getattr(cfg, "seed", 0)
    data_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=seed,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        runner_type=cfg.runner,
    )

    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    model = MMDataParallel(model.cuda(args.gpu_id), device_ids=[args.gpu_id])
    load_checkpoint(model.module, args.checkpoint, map_location="cpu", strict=False)

    class_names = _accumulate_class_names(cfg)
    print("Collecting matched pred/GT TTC (same assignment as training loss_ttc)...")
    out = run_breakdown(model, data_loader, args.gpu_id, class_names, args.max_batches)
    print(json.dumps(out, indent=2))

    if not args.no_save:
        save_dir = args.save_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.checkpoint)) or ".",
            "eval_ttc_breakdown",
        )
        os.makedirs(save_dir, exist_ok=True)
        js = os.path.join(save_dir, "ttc_breakdown.json")
        with open(js, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {js}")
        png = os.path.join(save_dir, "ttc_breakdown_by_class.png")
        if "by_class" in out:
            _plot_by_class(png, out["by_class"])
            if os.path.isfile(png):
                print(f"Wrote {png}")


if __name__ == "__main__":
    main()
