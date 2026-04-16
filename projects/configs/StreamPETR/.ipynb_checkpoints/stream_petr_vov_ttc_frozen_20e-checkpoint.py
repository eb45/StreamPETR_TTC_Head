# StreamPETR Phase 3: train TTC MLP on frozen backbone.
#
# Before training:
#   1) Build TTC labels (all samples for the chosen nuScenes version; lookup is by ann token):
#        python tools/generate_ttc_labels.py --data-root ./data/nuscenes \
#          --version v1.0-mini --out ./data/nuscenes/ttc_gt_labels_train.pkl
#   2) Point STREAMPETR_TTC_PKL at that file (or edit the default below). Use a full StreamPETR
#      checkpoint in load_from (override with env STREAMPETR_LOAD_FROM if you prefer).
#
# Weights & Biases (optional): pip install wandb && wandb login
#   USE_WANDB=1 WANDB_PROJECT=myproj [WANDB_ENTITY=team] [WANDB_NAME=run1]
#
import os

_base_ = ['./stream_petr_vov_flash_800_bs2_seq_24e.py']

# nuScenes root for images + DB tables. Match ttc_mlp_head.sh (export NUSCENES_ROOT=...).
_dr = os.environ.get('NUSCENES_ROOT', '').strip()
data_root = _dr if _dr else './data/nuscenes/'
if not data_root.endswith('/'):
    data_root = data_root + '/'
# NuScenes devkit expects v1.0-mini | v1.0-trainval | v1.0-test (not the shorthand v1.0).
_nv = os.environ.get('NUSCENES_VER', 'v1.0-trainval').strip()
if _nv == 'v1.0':
    _nv = 'v1.0-trainval'
nuscenes_version = _nv

# Must be set here — mmcv can execute this file before base names exist.
# Default is **1 GPU** (typical workstation / single Slurm GPU). For 8× GPU training, use e.g.
#   --cfg-options num_gpus=8 num_iters_per_epoch=1758 runner.max_iters=35160 ...
num_gpus = 1
batch_size = 2

# From stream_petr_vov_flash_800_bs2_seq_24e.py — mmcv may not inject base names before this file runs.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
ida_aug_conf = {
    'resize_lim': (0.47, 0.625),
    'final_dim': (320, 800),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img'] + collect_keys,
                meta_keys=(
                    'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token')),
        ]),
]

_log_hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook'),
]
if os.environ.get('USE_WANDB', '0') == '1':
    _wb = dict(project=os.environ.get('WANDB_PROJECT', 'streampetr-ttc'))
    _name = os.environ.get('WANDB_NAME') or os.environ.get('WANDB_RUN_NAME')
    if _name:
        _wb['name'] = _name
    if os.environ.get('WANDB_ENTITY'):
        _wb['entity'] = os.environ['WANDB_ENTITY']
    if os.environ.get('WANDB_TAGS'):
        _wb['tags'] = [t.strip() for t in os.environ['WANDB_TAGS'].split(',') if t.strip()]
    if os.environ.get('WANDB_NOTES'):
        _wb['notes'] = os.environ['WANDB_NOTES']
    _log_hooks.append(dict(type='WandbLoggerHook', init_kwargs=_wb))

log_config = dict(interval=50, hooks=_log_hooks)

# Pickle from tools/generate_ttc_labels.py (ann token -> TTC). Overridable via env for Slurm/scripts.
ttc_pkl = os.environ.get(
    'STREAMPETR_TTC_PKL', './data/nuscenes/ttc_gt_labels_train.pkl')

num_epochs = 20
num_iters_per_epoch = 28130 // (num_gpus * batch_size)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
         with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='LoadGTTC',
        ttc_pkl=ttc_pkl,
        class_names=class_names,
        point_cloud_range=point_cloud_range,
        ann_file=data_root + 'nuscenes2d_temporal_infos_train.pkl',
        data_root=data_root,
        nuscenes_version=nuscenes_version,
    ),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels',
                                 'centers2d', 'depths', 'prev_exists', 'gt_ttc'] + collect_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token',
                    'gt_bboxes_3d', 'gt_labels_3d')),
]

# TTC loss: original single threshold (GT TTC < ttc_low_thresh_s → ttc_low_weight). Tiered
# critical/urgent weights are disabled (ttc_crit_thresh_s=0 in StreamPETRHead).
model = dict(
    pts_bbox_head=dict(
        ttc_head=dict(
            embed_dims=256,
            hidden_dim=256,
            num_layers=0,
            dropout=0.1,
            ttc_max=10.0,
        ),
        loss_ttc_weight=1.0,
        ttc_crit_thresh_s=0.0,
        ttc_crit_weight=1.0,
        ttc_low_thresh_s=3.0,
        ttc_low_weight=3.0,
        ttc_smooth_beta=1.0,
        dn_weight=0.0,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
    ),
    img_roi_head=dict(
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=0.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=0.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=0.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=0.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=0.0)),
)

optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=1e-4,
    paramwise_cfg=dict(custom_keys={}),
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=1.0, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

evaluation = dict(interval=num_iters_per_epoch, pipeline=test_pipeline)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

find_unused_parameters = True

custom_hooks = [
    dict(type='FreezeDetectorExceptTTCHook'),
]

load_from = os.environ.get(
    'STREAMPETR_LOAD_FROM', 'ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth')

# Merge with base: keep ann_file/data_root consistent with LoadGTTC (env NUSCENES_ROOT at parse time).
# Each worker loads nuScenes + TTC pickle state — high RAM. Default 2 workers; set WORKERS_PER_GPU=0
# (main process only) or 1 if Slurm still OOM-kills workers.
_workers = int(os.environ.get("WORKERS_PER_GPU") or "2")
data = dict(
    workers_per_gpu=_workers,
    train=dict(
        pipeline=train_pipeline,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_temporal_infos_train.pkl',
    ),
)
