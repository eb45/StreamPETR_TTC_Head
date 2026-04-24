# StreamPETR Phase 3 (v3 ablation): TTC head = **query embedding + predicted vx, vy**.
#
# Compare:
#   Baseline (embed only): projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py
#   v2 (wider/deeper MLP, embed only): projects/configs/StreamPETR_ttc_v2/stream_petr_vov_ttc_frozen_20e.py
#   This v3: same tiered loss as v2, but ``TTCRiskHeadV3`` fuses LayerNorm(query) with BEV velocity
#             from the frozen reg head (``all_bbox_preds[..., 8:10]``, lidar frame).
#
# Train (example):
#   bash tools/dist_train.sh projects/configs/StreamPETR_ttc_v3/stream_petr_vov_ttc_frozen_20e.py 1 \
#     --work-dir ./work_dirs/streampetr_ttc_v3_frozen_20e
#
import os

_base_ = ['../StreamPETR_ttc_v2/stream_petr_vov_ttc_frozen_20e.py']

model = dict(
    pts_bbox_head=dict(
        ttc_head=dict(
            type='TTCRiskHeadV3',
            embed_dims=256,
            hidden_dim=384,
            num_layers=2,
            dropout=0.1,
            ttc_max=10.0,
            vel_scale=30.0,
        ),
    ),
)
