#!/bin/bash
# Slurm (CPU, high RAM): generate ttc_gt_labels_*.pkl via tools/generate_ttc_labels.py
# Full trainval needs memory — do not run on low-RAM login nodes.
#
# Submit from repo root:
#   sbatch run_generate_ttc_labels_slurm.sh
#
# Overrides:
#   NUSCENES_ROOT=/path/to/nuscenes VERSION=v1.0-trainval MEM=128G sbatch run_generate_ttc_labels_slurm.sh

#SBATCH --job-name=ttc_labels
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=common
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs

_REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${_REPO}"

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env

export PYTHONPATH="$(pwd):$(pwd)/tools:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Parent of v1.0-trainval/ (must contain maps/, samples/, sweeps/, v1.0-trainval/).
# Example: NUSCENES_ROOT=/hpc/home/eb408/eb408/CS372/StreamPETR/data_full/nuscenes
NUSCENES_ROOT="${NUSCENES_ROOT:-${_REPO}/data_full/nuscenes}"
VERSION="${VERSION:-v1.0-trainval}"
OUT="${OUT:-${NUSCENES_ROOT%/}/ttc_gt_labels_v1_0_trainval.pkl}"

echo "[ttc_labels] REPO=${_REPO}"
echo "[ttc_labels] NUSCENES_ROOT=${NUSCENES_ROOT}"
echo "[ttc_labels] VERSION=${VERSION}"
echo "[ttc_labels] OUT=${OUT}"

python -u tools/generate_ttc_labels.py \
  --data-root "${NUSCENES_ROOT}" \
  --version "${VERSION}" \
  --out "${OUT}" \
  --progress-every 2000

echo "[ttc_labels] Done. Check: ls -lh ${OUT}"
