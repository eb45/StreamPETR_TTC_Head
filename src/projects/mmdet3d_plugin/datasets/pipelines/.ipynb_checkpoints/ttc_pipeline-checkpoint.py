# Copyright (c) OpenMMLab. All rights reserved.
"""Attach per-GT TTC targets (aligned to final gt_bboxes_3d after filters)."""
import os.path as osp
import pickle

import mmcv
import numpy as np
from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class LoadGTTC(object):
    """Load GT TTC from ``generate_ttc_labels.py`` pickle, matched to each GT box.

    Place this **after** ``ObjectRangeFilter`` and ``ObjectNameFilter`` in the
    training pipeline so ``gt_bboxes_3d`` matches filtering used by detection.
    """

    def __init__(
        self,
        ttc_pkl,
        class_names,
        point_cloud_range,
        ann_file,
        data_root,
        nuscenes_version="v1.0-mini",
        use_valid_flag=True,
    ):
        ttc_pkl = osp.expanduser(ttc_pkl)
        with open(ttc_pkl, "rb") as f:
            payload = pickle.load(f)
        self.labels = payload["labels"] if isinstance(payload, dict) and "labels" in payload else payload
        self.class_names = list(class_names)
        self.pc_range = np.asarray(point_cloud_range, dtype=np.float32)
        data_root = osp.expanduser(data_root)
        self._token2info = {e["token"]: e for e in mmcv.load(ann_file)["infos"]}
        self.data_root = data_root
        self.nuscenes_version = nuscenes_version
        self.use_valid_flag = use_valid_flag
        self._nusc = None

    def _nuscenes(self):
        if self._nusc is None:
            from nuscenes.nuscenes import NuScenes

            self._nusc = NuScenes(version=self.nuscenes_version, dataroot=self.data_root, verbose=False)
        return self._nusc

    def _in_bev(self, xy):
        x, y = float(xy[0]), float(xy[1])
        r = self.pc_range
        return bool(r[0] <= x <= r[3] and r[1] <= y <= r[4])

    def _candidates(self, sample_token):
        info = self._token2info[sample_token]
        nusc = self._nuscenes()
        sample = nusc.get("sample", sample_token)
        anns = sample["anns"]
        if self.use_valid_flag:
            mask = info["valid_flag"].astype(bool)
        else:
            mask = info["num_lidar_pts"] > 0
        names = info["gt_names"]
        boxes = info["gt_boxes"]
        assert len(anns) == len(mask)
        cands = []
        for i, tok in enumerate(anns):
            if not mask[i]:
                continue
            nm = names[i]
            if nm not in self.class_names:
                continue
            ctr = boxes[i, :3]
            if not self._in_bev(ctr[:2]):
                continue
            lab = self.class_names.index(nm)
            cands.append((tok, ctr.astype(np.float64), int(lab)))
        return cands

    def map_gt_to_ann_tokens(self, sample_token, gtb, labs):
        """Map each GT box index to a nuScenes annotation token (same LSAP as :meth:`__call__`)."""
        from scipy.optimize import linear_sum_assignment

        if isinstance(labs, np.ndarray):
            labels_np = labs
        elif hasattr(labs, "detach"):
            labels_np = labs.detach().cpu().numpy()
        else:
            labels_np = np.asarray(labs)
        gc = gtb.gravity_center
        if hasattr(gc, "detach"):
            centers = gc.detach().cpu().numpy()
        else:
            centers = np.asarray(gc)
        n = centers.shape[0]
        cands = self._candidates(sample_token)
        m = len(cands)
        out = [None] * n
        if n == 0 or m == 0:
            return out
        cost = np.full((n, m), 1e6, dtype=np.float64)
        for i in range(n):
            for j, c in enumerate(cands):
                if c[2] != int(labels_np[i]):
                    continue
                cost[i, j] = np.linalg.norm(centers[i, :2] - c[1][:2])
        rind, cind = linear_sum_assignment(cost)
        for r, c in zip(rind, cind):
            if cost[r, c] >= 1e5:
                continue
            out[r] = cands[c][0]
        return out

    def __call__(self, results):
        from scipy.optimize import linear_sum_assignment

        sample_token = results["sample_idx"]
        gtb = results["gt_bboxes_3d"]
        labs = results["gt_labels_3d"]
        if isinstance(labs, np.ndarray):
            labels_np = labs
        else:
            labels_np = labs.numpy()
        centers = gtb.gravity_center.numpy()
        n = centers.shape[0]
        cands = self._candidates(sample_token)
        m = len(cands)
        cost = np.full((n, m), 1e6, dtype=np.float64)
        for i in range(n):
            for j, c in enumerate(cands):
                if c[2] != int(labels_np[i]):
                    continue
                cost[i, j] = np.linalg.norm(centers[i, :2] - c[1][:2])
        if n == 0:
            results["gt_ttc"] = np.zeros((0,), dtype=np.float32)
            return results
        if m == 0:
            results["gt_ttc"] = np.full((n,), np.nan, dtype=np.float32)
            return results
        rind, cind = linear_sum_assignment(cost)
        ttc = np.full((n,), np.nan, dtype=np.float32)
        for r, c in zip(rind, cind):
            if cost[r, c] >= 1e5:
                continue
            tok = cands[c][0]
            if tok in self.labels:
                ttc[r] = float(self.labels[tok]["ttc"])
        results["gt_ttc"] = ttc
        return results
