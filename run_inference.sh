#!/bin/bash
#SBATCH --job-name=streampetr_eval_viz
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd ~/eb408/CS372/StreamPETR

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env

export PYTHONPATH=$(pwd):$PYTHONPATH

# Step 1 — run evaluation and get stats
python tools/test.py \
  projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py \
  ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth \
  --eval bbox

# Step 2 — generate visualizations
python3 tools/visualize.py