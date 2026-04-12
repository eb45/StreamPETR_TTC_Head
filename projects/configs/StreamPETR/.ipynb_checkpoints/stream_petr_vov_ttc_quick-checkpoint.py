# StreamPETR Phase 3 — faster multi-GPU schedule with solid TTC progress.
#
# Effective global batch = num_gpus * batch_size (per GPU). With the defaults below:
#   num_iters_per_epoch = 28130 // (4 * 2) = 3516
#   max_iters = 6 * 3516 = 21096  (~roughly 2–4 h wall-clock depending on GPU; faster than 1×GPU 10ep)
#
# Launch (must match num_gpus):
#   bash tools/dist_train.sh projects/configs/StreamPETR/stream_petr_vov_ttc_quick.py 4 \
#     --work-dir ./work_dirs/streampetr_ttc_quick --autoscale_lr
#
# Only 2 GPUs? Set num_gpus = 2 here (or --cfg-options) and either keep num_epochs=6 for longer
# training or drop num_epochs to ~4 to cap time (28130//(2*2) iters per epoch).

_base_ = ['./stream_petr_vov_ttc_frozen_20e.py']

# Match Slurm / node: dist_train.sh last arg must equal num_gpus.
num_gpus = 4
batch_size = 2
# 6 epoch-equivalents: stronger TTC fit than 2ep; still far shorter than full 1×GPU long run.
num_epochs = 6

num_iters_per_epoch = 28130 // (num_gpus * batch_size)

# Deep-merge with base for evaluation (keep pipeline from base; mmcv child exec has no test_pipeline).
evaluation = dict(interval=num_iters_per_epoch)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
