#!/usr/bin/env python3
"""Mean TTC loss on a val (or train) split — compares trained Phase-3 checkpoint vs pretrained baseline.

Standard ``tools/test.py`` does not run the TTC MLP. This script rebuilds the training forward
(with GT TTC) on a chosen ``ann_file`` and reports the average (per batch) sum of all
``frame_*_loss_ttc`` terms.

Usage:
  python tools/eval_ttc_mlp.py \\
    projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py \\
    work_dirs/streampetr_ttc_frozen_20e/iter_140650.pth \\
    --ann-file data/nuscenes/nuscenes2d_temporal_infos_val.pkl \\
    --max-batches 100

Optional baseline (pretrained backbone; TTC head random if not in ckpt):
  python tools/eval_ttc_mlp.py config.py trained.pth --ann-file ... \\
    --pretrained-baseline ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth

Writes by default (under ``<checkpoint_dir>/eval_ttc_mlp/``): ``ttc_loss_compare.png``,
``ttc_loss_metrics.json``. Use ``--no-save`` to skip; ``--save-dir DIR`` to choose a folder.

``LoadGTTC`` needs **one** GT TTC pickle (from ``generate_ttc_labels.py``) whose keys cover the
ann tokens in your temporal ``--ann-file``. You do **not** need separate train/val TTC files: a
single ``v1.0-trainval`` (or mini) labels pickle works with both train and val **infos** pkls.
Pass ``--ttc-pkl`` or set ``STREAMPETR_TTC_PKL``; otherwise the first matching
``ttc_gt_labels*.pkl`` next to the infos file is used.

Verbose mmcv lines about init_weights (backbone unchanged) are normal if the backbone is already
pretrained. Default eval uses CUDA (``--gpu-id 0`` for typical 1×GPU Slurm).

Use ``--device cpu`` when GPUs are broken or unavailable (login node, bad nodes, no ``nvidia-smi``).
CPU is much slower — use a small ``--max-batches`` (e.g. 10–20) for a quick sanity check.

Intermittent ``CUDA unknown error`` on shared nodes: optional retries via env
``CUDA_INIT_RETRIES`` (default 5) and ``CUDA_INIT_RETRY_DELAY_SEC`` (default 3).
"""
from __future__ import annotations

import argparse
import copy
import glob
import logging
import math
import os
import sys
import time

# Set before importing torch — can avoid flaky "CUDA unknown error" on some Slurm/driver stacks.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# Login-shell CUDA_HOME=/usr/local/cuda often breaks conda PyTorch (nvidia-smi OK, torch.cuda False).
if os.environ.get("STREAMPETR_KEEP_CUDA_HOME") != "1":
    os.environ.pop("CUDA_HOME", None)

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, scatter
from mmcv.runner import load_checkpoint

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _configure_logging() -> None:
    # mmcv logs every parameter where init_weights does not change values (normal for pretrained backbone).
    logging.getLogger("mmcv").setLevel(logging.WARNING)


def _ensure_cuda(gpu_id: int) -> None:
    """Check GPU; retry a few times — some clusters return transient CUDA unknown errors."""
    retries = int(os.environ.get("CUDA_INIT_RETRIES", "5"))
    delay = float(os.environ.get("CUDA_INIT_RETRY_DELAY_SEC", "3"))
    last: BaseException | None = None

    for attempt in range(retries):
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is False")
            n = torch.cuda.device_count()
            if gpu_id >= n:
                raise RuntimeError(
                    f"--gpu-id {gpu_id} but only {n} device(s) visible "
                    f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r})"
                )
            torch.cuda.set_device(gpu_id)
            torch.zeros(1, device=f"cuda:{gpu_id}")
            if attempt:
                print(f"[eval_ttc_mlp] CUDA OK after {attempt + 1} attempt(s)", flush=True)
            return
        except RuntimeError as e:
            last = e
            if attempt + 1 < retries:
                print(
                    f"[eval_ttc_mlp] CUDA init failed ({attempt + 1}/{retries}): {e!s}; "
                    f"sleep {delay}s (set CUDA_INIT_RETRIES=1 to disable retries)",
                    flush=True,
                )
                time.sleep(delay)

    raise RuntimeError(
        "CUDA is not available to PyTorch after retries.\n"
        "  • If `nvidia-smi` failed in your Slurm log: use --gres=gpu:1 on a GPU partition (not a login node).\n"
        "  • If failures are random: bad GPU nodes exist — resubmit or exclude the node; try another partition.\n"
        "  • If `nvidia-smi` always works but PyTorch never does: GPU build of PyTorch + matching `module load cuda`.\n"
        "  • Use --gpu-id 0 for a single-GPU job; do not change CUDA_VISIBLE_DEVICES after Python starts."
    ) from last


def parse_args():
    p = argparse.ArgumentParser(description="Mean TTC loss on a split (Phase 3 MLP eval)")
    p.add_argument("config", help="Phase 3 config (same as training)")
    p.add_argument("checkpoint", help="Trained checkpoint .pth (Phase 3)")
    p.add_argument(
        "--ann-file",
        required=True,
        help="Temporal infos pkl (e.g. nuscenes2d_temporal_infos_val.pkl)",
    )
    p.add_argument(
        "--pretrained-baseline",
        default=None,
        help="If set, also report mean loss after loading only this checkpoint (untrained TTC if keys missing).",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Batches to average (default 100). Use 0 for the **entire** val dataloader (full split).",
    )
    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="cuda: need working GPU + CUDA PyTorch. cpu: no GPU (slow; shrink --max-batches).",
    )
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument(
        "--save-dir",
        default=None,
        help="Write PNG bar chart + JSON metrics here. Default: <checkpoint_dir>/eval_ttc_mlp",
    )
    p.add_argument("--no-save", action="store_true", help="Skip writing PNG/JSON.")
    p.add_argument(
        "--ttc-pkl",
        default=None,
        help="GT TTC pickle from generate_ttc_labels.py (ann token -> TTC). "
        "Default: try STREAMPETR_TTC_PKL, then common names next to --ann-file "
        "(e.g. ttc_gt_labels_v1_0_trainval.pkl for val split).",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="nuScenes root (folder with samples/). Overrides cfg.data.train.data_root when set.",
    )
    return p.parse_args()


def _resolve_ttc_pkl(ann_file: str, ttc_pkl_arg: str | None) -> str:
    """LoadGTTC needs a pickle that contains TTC for all ann tokens in ann_file (often trainval)."""
    ann_file = os.path.abspath(os.path.expanduser(ann_file))
    data_dir = os.path.dirname(ann_file)
    candidates: list[str] = []
    if ttc_pkl_arg:
        candidates.append(os.path.abspath(os.path.expanduser(ttc_pkl_arg)))
    env_pkl = os.environ.get("STREAMPETR_TTC_PKL")
    if env_pkl:
        candidates.append(os.path.abspath(os.path.expanduser(env_pkl)))
    base = os.path.basename(ann_file).lower()
    if "val" in base:
        candidates.extend(
            [
                os.path.join(data_dir, "ttc_gt_labels_v1_0_trainval.pkl"),
                os.path.join(data_dir, "ttc_gt_labels_val.pkl"),
                os.path.join(data_dir, "ttc_gt_labels_train.pkl"),
            ]
        )
    elif "train" in base:
        candidates.extend(
            [
                os.path.join(data_dir, "ttc_gt_labels_train.pkl"),
                os.path.join(data_dir, "ttc_gt_labels_v1_0_trainval.pkl"),
            ]
        )
    candidates.extend(
        [
            os.path.join(data_dir, "ttc_gt_labels_v1_0_mini.pkl"),
            os.path.join(data_dir, "ttc_gt_labels_v1_0_trainval.pkl"),
            os.path.join(data_dir, "ttc_gt_labels.pkl"),
        ]
    )
    for p in sorted(glob.glob(os.path.join(data_dir, "ttc_gt_labels*.pkl"))):
        candidates.append(os.path.abspath(p))
    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    for c in uniq:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "Could not find GT TTC pickle. Tried:\n  "
        + "\n  ".join(uniq)
        + "\n\nGenerate one with tools/generate_ttc_labels.py, then pass --ttc-pkl PATH "
        "or set STREAMPETR_TTC_PKL. One trainval-scale pickle covers train and val ann tokens."
    )


def _patch_load_gttc_in_pipeline(
    node, ann_file: str, ttc_pkl: str, data_root: str | None = None
) -> int:
    """Recursively set ann_file + ttc_pkl on every LoadGTTC step (handles nested transforms)."""
    n = 0
    if isinstance(node, (list, tuple)):
        for item in node:
            n += _patch_load_gttc_in_pipeline(item, ann_file, ttc_pkl, data_root)
        return n
    if isinstance(node, dict):
        if node.get("type") == "LoadGTTC":
            node["ann_file"] = ann_file
            node["ttc_pkl"] = ttc_pkl
            if data_root is not None:
                node["data_root"] = data_root
            return 1
        for k in ("pipeline", "transforms", "img_transform"):
            if k in node:
                n += _patch_load_gttc_in_pipeline(node[k], ann_file, ttc_pkl, data_root)
        return n
    return 0


def _patch_data_train_ann(
    cfg, ann_file: str, ttc_pkl: str, data_root: str | None = None
):
    ann_file = os.path.abspath(ann_file)
    ttc_pkl = os.path.abspath(ttc_pkl)
    dr = os.path.abspath(os.path.expanduser(data_root)) if data_root else None
    # Full deepcopy — shallow copies can leave LoadGTTC.ttc_pkl pointing at config defaults.
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
        raise RuntimeError("cfg.data.train has no 'pipeline'")
    n = _patch_load_gttc_in_pipeline(pl, ann_file, ttc_pkl, dr)
    if n == 0:
        raise RuntimeError(
            "eval_ttc_mlp: no LoadGTTC in data.train.pipeline. "
            "Use a Phase-3 TTC config (e.g. stream_petr_vov_ttc_frozen_20e.py)."
        )
    return train_cfg


def _sum_ttc_terms(losses) -> float:
    s = 0.0
    for k, v in losses.items():
        if "loss_ttc" in k and hasattr(v, "item"):
            s += float(v.item())
    return s


def _run_eval(
    model_parallel,
    data_loader,
    device_id: int,
    max_batches: int,
    *,
    use_cpu: bool,
) -> tuple[float, int]:
    # Match training forward (temporal / no_grad branches); eval() can change behavior.
    model_parallel.train()
    total = 0.0
    n_batch = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if max_batches > 0 and i >= max_batches:
                break
            # GPU: pre-scatter to cuda. Do not pass pre-scattered batches through MMDataParallel.forward —
            # it runs scatter again and can drop keys (e.g. ``img``), causing KeyError in ``forward_train``.
            # Call the wrapped module directly after manual scatter.
            if not use_cpu:
                data = scatter(data, [device_id])[0]
            head = model_parallel.module.pts_bbox_head
            if hasattr(head, "reset_memory"):
                head.reset_memory()
            if use_cpu:
                losses = model_parallel(return_loss=True, **data)
            else:
                losses = model_parallel.module(return_loss=True, **data)
            if losses is None:
                continue
            total += _sum_ttc_terms(losses)
            n_batch += 1
    if n_batch == 0:
        return float("nan"), 0
    return total / n_batch, n_batch


def _default_save_dir(checkpoint: str) -> str:
    d = os.path.dirname(os.path.abspath(checkpoint))
    return os.path.join(d if d else ".", "eval_ttc_mlp")


def _save_eval_visuals(
    save_dir: str,
    *,
    baseline_mean: float | None,
    baseline_nb: int,
    trained_mean: float,
    trained_nb: int,
    ttc_pkl: str,
    args: argparse.Namespace,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    import json

    metrics = {
        "trained_mean_sum_loss_ttc_per_batch": trained_mean,
        "n_batches": trained_nb,
        "ann_file": os.path.abspath(args.ann_file),
        "ttc_pkl": ttc_pkl,
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": os.path.abspath(args.config),
        "max_batches": args.max_batches,
        "device": getattr(args, "device", "cuda"),
    }
    if baseline_mean is not None:
        metrics["baseline_mean_sum_loss_ttc_per_batch"] = baseline_mean
        metrics["baseline_n_batches"] = baseline_nb

    json_path = os.path.join(save_dir, "ttc_loss_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  wrote {json_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  (skip PNG: matplotlib unavailable: {exc})")
        return

    labels: list[str] = []
    vals: list[float] = []
    colors: list[str] = []
    if baseline_mean is not None:
        labels.append("Pretrained baseline")
        vals.append(baseline_mean)
        colors.append("#6b8cae")
    labels.append("Trained Phase-3")
    vals.append(trained_mean)
    colors.append("#2d6a4f")

    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=120)
    ann_short = os.path.basename(args.ann_file)
    mb = int(getattr(args, "max_batches", 0))
    # Never label as "batches≤0" — max_batches<=0 means entire val loader.
    _bt = "full val (all batches)" if mb <= 0 else f"first {mb} batches only"

    def _finite_plot_height(v: float) -> float:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return 0.0
        return float(v)

    # No batches → NaN means; bar chart would be blank — show a clear message instead.
    if trained_nb <= 0 or (isinstance(trained_mean, float) and math.isnan(trained_mean)):
        ax.axis("off")
        ax.set_title(f"TTC MLP eval  ({ann_short}, {_bt})")
        ax.text(
            0.5,
            0.55,
            "No batches evaluated (n_batches=0 or loss is NaN).\n"
            "Check: ann_file / --data-root, dataloader length, and job logs for errors.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
    else:
        x = range(len(labels))
        plot_heights = [_finite_plot_height(v) for v in vals]
        bars = ax.bar(x, plot_heights, color=colors, edgecolor="0.2", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Mean Σ frame_*_loss_ttc  (per batch)")
        ax.set_title(f"TTC MLP eval  ({ann_short}, {_bt})")
        finite = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
        ymax = max(finite) if finite else 1.0
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
        for b, v in zip(bars, vals):
            lbl = "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
            ytop = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                ytop,
                lbl,
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.tight_layout()
    png_path = os.path.join(save_dir, "ttc_loss_compare.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")


def main():
    args = parse_args()
    _configure_logging()
    use_cpu = args.device == "cpu"
    if use_cpu:
        print(
            "[eval_ttc_mlp] --device cpu: no CUDA required; expect long runtimes. "
            "Consider --max-batches 10–20 for a quick check.",
            flush=True,
        )
    else:
        _ensure_cuda(args.gpu_id)

    cfg = Config.fromfile(args.config)
    cfg.gpu_ids = [] if use_cpu else [args.gpu_id]

    import projects.mmdet3d_plugin.datasets  # noqa: F401 — register CustomNuScenesDataset
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader, custom_build_dataset
    from mmdet3d.models import build_model

    ttc_pkl = _resolve_ttc_pkl(args.ann_file, args.ttc_pkl)
    print(f"Using GT TTC pickle: {ttc_pkl}")
    data_cfg = _patch_data_train_ann(cfg, args.ann_file, ttc_pkl, data_root=args.data_root)
    dataset = custom_build_dataset(data_cfg)
    seed = getattr(cfg, "seed", 0)
    workers = 0 if use_cpu else cfg.data.workers_per_gpu
    data_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        workers,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=seed,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        runner_type=cfg.runner,
    )
    print(
        f"[eval_ttc_mlp] val batches in loader: {len(data_loader)}  "
        f"(using {'ALL batches (full split)' if args.max_batches <= 0 else f'first {args.max_batches} batches only'})",
        flush=True,
    )

    def fresh_model():
        m = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
        m.init_weights()
        if use_cpu:
            return MMDataParallel(m.cpu(), device_ids=[])
        return MMDataParallel(m.cuda(args.gpu_id), device_ids=[args.gpu_id])

    baseline_mean = None
    baseline_nb = 0
    if args.pretrained_baseline:
        print("=== Pretrained baseline (TTC head untrained if weights missing in ckpt) ===")
        model = fresh_model()
        load_checkpoint(model.module, args.pretrained_baseline, map_location="cpu", strict=False)
        baseline_mean, baseline_nb = _run_eval(
            model, data_loader, args.gpu_id, args.max_batches, use_cpu=use_cpu
        )
        print(f"  mean sum(loss_ttc terms) per batch: {baseline_mean:.6f}  (batches={baseline_nb})\n")

    print("=== Trained Phase-3 checkpoint ===")
    model = fresh_model()
    load_checkpoint(model.module, args.checkpoint, map_location="cpu", strict=False)
    trained_mean, trained_nb = _run_eval(
        model, data_loader, args.gpu_id, args.max_batches, use_cpu=use_cpu
    )
    print(f"  mean sum(loss_ttc terms) per batch: {trained_mean:.6f}  (batches={trained_nb})")

    print(
        "\nLower is better (same split, same batches). "
        "Compare to baseline row above if you passed --pretrained-baseline."
    )

    if not args.no_save:
        save_dir = args.save_dir or _default_save_dir(args.checkpoint)
        print(f"\nSaving eval artifacts under {save_dir}")
        _save_eval_visuals(
            save_dir,
            baseline_mean=baseline_mean,
            baseline_nb=baseline_nb,
            trained_mean=trained_mean,
            trained_nb=trained_nb,
            ttc_pkl=ttc_pkl,
            args=args,
        )


if __name__ == "__main__":
    main()
