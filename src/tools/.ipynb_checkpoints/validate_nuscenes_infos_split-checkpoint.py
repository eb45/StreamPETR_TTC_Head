#!/usr/bin/env python3
"""Check nuscenes2d_temporal_infos_train/val.pkl match an expected nuScenes split.

Each file is dict(infos=..., metadata=dict(version=...)) from tools/data_converter.
"""
from __future__ import annotations

import argparse
import os
import sys

import mmcv


def _load_version_and_len(path: str) -> tuple[str, int]:
    data = mmcv.load(path)
    meta = data.get("metadata") or {}
    ver = meta.get("version")
    if not ver:
        print(f"ERROR: {path} has no metadata.version (wrong or corrupt infos file).", file=sys.stderr)
        sys.exit(1)
    return str(ver), len(data.get("infos", []))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--expect",
        choices=("v1.0-trainval", "v1.0-mini"),
        required=True,
        help="Expected metadata.version in both pkls",
    )
    p.add_argument(
        "--train-pkl",
        default=None,
        help="Path to nuscenes2d_temporal_infos_train.pkl (default: <data-root>/...)",
    )
    p.add_argument(
        "--val-pkl",
        default=None,
        help="Path to nuscenes2d_temporal_infos_val.pkl (default: <data-root>/...)",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="If set, default train/val paths are under this directory (trailing / ok)",
    )
    args = p.parse_args()

    root = args.data_root
    if root:
        root = os.path.abspath(os.path.expanduser(root.rstrip("/") + "/"))
        train = args.train_pkl or os.path.join(root, "nuscenes2d_temporal_infos_train.pkl")
        val = args.val_pkl or os.path.join(root, "nuscenes2d_temporal_infos_val.pkl")
    else:
        train = args.train_pkl
        val = args.val_pkl
    if not train or not val:
        p.error("Provide --data-root or both --train-pkl and --val-pkl")

    n_train = n_val = 0
    for label, path in (("train", train), ("val", val)):
        if not os.path.isfile(path):
            print(f"ERROR: missing {label} infos file: {path}", file=sys.stderr)
            sys.exit(1)
        ver, n = _load_version_and_len(path)
        if label == "train":
            n_train = n
        else:
            n_val = n
        if ver != args.expect:
            print(
                f"ERROR: {label} infos {path}\n"
                f"       metadata.version = {ver!r}, expected {args.expect!r}\n"
                f"       (Regenerate with create_data_nusc.py for the matching split, or fix paths.)",
                file=sys.stderr,
            )
            sys.exit(1)
    print(
        f"OK: infos match split {args.expect} "
        f"(train samples={n_train}, val samples={n_val})"
    )


if __name__ == "__main__":
    main()
