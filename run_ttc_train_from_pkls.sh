#!/bin/bash
# Slurm: TTC training only — assumes nuscenes2d_temporal_infos_{train,val}.pkl already exist under NUSCENES_ROOT.
# Does NOT run tools/create_data_nusc.py (skips the long 28k+ sample prep).
#
# You still need ttc_gt_labels_*.pkl for the TTC head. If it is missing:
#   - Run once:  python tools/generate_ttc_labels.py --data-root ... --version v1.0-trainval --out ...
#   - Or submit ttc_mlp_head.sh with default AUTO_PREP_DATA=1 (infos skipped if OK; only TTC is generated).
#
# Submit from the **repo root** (where ttc_mlp_head.sh lives):
#   cd /path/to/StreamPETR && NUSCENES_ROOT=/path/to/nuscenes sbatch run_ttc_train_from_pkls.sh
#
# Slurm copies this script to /var/spool/slurmd/... — we use SLURM_SUBMIT_DIR to find ttc_mlp_head.sh.
#
# Same #SBATCH / env as ttc_mlp_head.sh; override NUSCENES_ROOT, WORK_DIR, STREAMPETR_* as usual.

#SBATCH --job-name=streampetr_ttc_from_pkls
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO="${SLURM_SUBMIT_DIR:-${_SCRIPT_DIR}}"
if [[ ! -f "${_REPO}/ttc_mlp_head.sh" ]]; then
  echo "FATAL: ttc_mlp_head.sh not found in ${_REPO}. Run: cd /path/to/StreamPETR && sbatch run_ttc_train_from_pkls.sh" >&2
  exit 1
fi

export AUTO_PREP_DATA=0
exec bash "${_REPO}/ttc_mlp_head.sh" "$@"
