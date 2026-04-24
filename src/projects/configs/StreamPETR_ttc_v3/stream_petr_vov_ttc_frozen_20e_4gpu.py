#
# Launch:
#   bash tools/dist_train.sh projects/configs/StreamPETR_ttc_v3/stream_petr_vov_ttc_frozen_20e_6gpu.py 6 \
#     --work-dir ./work_dirs/streampetr_ttc_v3_frozen_20e_6gpu
#
_base_ = ['./stream_petr_vov_ttc_frozen_20e.py']

num_gpus = 4
batch_size = 1

num_iters_per_epoch = 28130 // (num_gpus * batch_size)

# Must match num_epochs in base (child merge does not re-exec base for name lookup).
num_epochs = 10

evaluation = dict(interval=num_iters_per_epoch)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
