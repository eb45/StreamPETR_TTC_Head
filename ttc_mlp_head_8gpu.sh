#!/bin/bash
#SBATCH --job-name=streampetr_ttc_phase3_8gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# 8× GPU Phase 3 (single node): same logic as ttc_mlp_head_4gpu.sh with NUM_GPUS=8 defaults.
#SBATCH --partition=common
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

set -euo pipefail

# Must match GPU count above unless you also change #SBATCH --gres.
export NUM_GPUS="${NUM_GPUS:-8}"
# Per-process DataLoader workers; keep low on TTC+nuScenes to avoid host OOM.
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"

_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "${_REPO}/ttc_mlp_head_4gpu.sh" "$@"
