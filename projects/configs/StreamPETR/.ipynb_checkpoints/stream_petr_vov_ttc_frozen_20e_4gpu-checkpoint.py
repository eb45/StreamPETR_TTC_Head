# Same as stream_petr_vov_ttc_frozen_20e.py but for 4× GPU distributed training.
# Effective batch = num_gpus * batch_size = 4 * 2 = 8 (per-step global batch).
#
# Launch:
#   bash tools/dist_train.sh projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e_4gpu.py 4 \
#     --work-dir ./work_dirs/streampetr_ttc_frozen_20e_tiered_4gpu
#
_base_ = ['./stream_petr_vov_ttc_frozen_20e.py']

num_gpus = 4
batch_size = 2

num_iters_per_epoch = 28130 // (num_gpus * batch_size)

# Must match num_epochs in stream_petr_vov_ttc_frozen_20e.py (child configs don't see base names
# like test_pipeline / num_epochs during mmcv Config.fromfile exec — see NameError).
num_epochs = 20

# Deep-merge with base: only override keys here; keep evaluation.pipeline from base.
evaluation = dict(interval=num_iters_per_epoch)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
