"""Helpers for denoising (DN) queries: unwrap gt from img_metas after DataParallel/scatter."""
from __future__ import annotations

import torch


def unwrap_dc(x):
    """DataContainer or value already unwrapped."""
    return x._data if hasattr(x, "_data") else x


def _tensor_to_boxes_2d(g: torch.Tensor) -> torch.Tensor:
    """LiDARInstance3DBoxes expects (N, box_dim); some pipelines pass a single box as 1D (7,) or (9,)."""
    if g.dim() != 1:
        return g
    if g.numel() == 0:
        return g.new_zeros((0, 7))
    if g.numel() in (7, 9):
        return g.unsqueeze(0)
    if g.numel() % 7 == 0:
        return g.view(-1, 7)
    if g.numel() % 9 == 0:
        return g.view(-1, 9)
    return g.unsqueeze(0)


def gt_boxes_for_dn(img_meta):
    """img_metas['gt_bboxes_3d'] may be DC, LiDARInstance3DBoxes, or Tensor."""
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    g = unwrap_dc(img_meta["gt_bboxes_3d"])
    if isinstance(g, torch.Tensor):
        g = _tensor_to_boxes_2d(g)
        g = LiDARInstance3DBoxes(g, box_dim=g.size(-1))
    return g


def gt_labels_for_dn(img_meta, num_boxes=None):
    """Unwrap gt_labels_3d; optional ``num_boxes`` trims/pads to match ``gt_bboxes_3d`` (fixes DN IndexError)."""
    lab = unwrap_dc(img_meta["gt_labels_3d"])
    if not isinstance(lab, torch.Tensor):
        return lab
    lab = lab.reshape(-1)
    if num_boxes is None:
        return lab
    if lab.numel() > num_boxes:
        return lab[:num_boxes]
    if lab.numel() < num_boxes:
        pad = lab.new_zeros(num_boxes - lab.numel(), dtype=lab.dtype, device=lab.device)
        return torch.cat([lab, pad], dim=0)
    return lab
