#!/bin/bash
# Full trainval val eval. Sets NUSCENES_ROOT/ANN_FILE/CHECKPOINT defaults, then runs run_eval_ttc_mlp.sh
# Run from StreamPETR repo root:  cd /path/to/StreamPETR && sbatch slurm_scripts/run_eval_ttc_mlp_full.sh
# Resolves the inner script: ./run_eval_ttc_mlp.sh or ./slurm_scripts/run_eval_ttc_mlp.sh
# Override: STREAMPETR_REPO=...  or  EVAL_SCRIPT=/path/to/run_eval_ttc_mlp.sh

#SBATCH --job-name=eval_ttc_mlp_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

set -euo pipefail
mkdir -p logs

_root="${STREAMPETR_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "$(cd "${_root}" && pwd)" || { echo "FATAL: cannot cd to ${_root}" >&2; exit 1; }
[[ -f src/tools/eval_ttc_mlp.py ]] || {
  echo "FATAL: not at StreamPETR root (missing src/tools/eval_ttc_mlp.py). Run: cd /path/to/StreamPETR && sbatch run_eval_ttc_mlp_full.sh" >&2
  echo "  Or set STREAMPETR_REPO to the repo root." >&2
  exit 1
}
_REPO="$(pwd)"

if [[ -n "${NUSCENES_ROOT:-}" ]]; then
  :
elif [[ -n "${DATA_FULL:-}" ]]; then
  export NUSCENES_ROOT="${DATA_FULL}"
elif [[ -d "${_REPO}/data_full" ]]; then
  export NUSCENES_ROOT="${_REPO}/data_full"
else
  export NUSCENES_ROOT="/work/eb408/nuscenes/train"
fi
export NUSCENES_ROOT="${NUSCENES_ROOT%/}/"
if [[ ! -f "${NUSCENES_ROOT}nuscenes2d_temporal_infos_train.pkl" ]] && [[ -f "${NUSCENES_ROOT}nuscenes/nuscenes2d_temporal_infos_train.pkl" ]]; then
  export NUSCENES_ROOT="${NUSCENES_ROOT}nuscenes/"
  echo "[eval full] Detected nested nuscenes/ — using NUSCENES_ROOT=${NUSCENES_ROOT}"
fi

export ANN_FILE="${ANN_FILE:-${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl}"
export STREAMPETR_TTC_PKL="${STREAMPETR_TTC_PKL:-${NUSCENES_ROOT}ttc_gt_labels_v1_0_trainval.pkl}"

export SKIP_COMPARE_SCENE="${SKIP_COMPARE_SCENE:-0}"
export COMPARE_VIDEO="${COMPARE_VIDEO:-1}"
export COMPARE_NUM_SCENES="${COMPARE_NUM_SCENES:-5}"
export SKIP_ABLATION_TABLE="${SKIP_ABLATION_TABLE:-0}"

export VALIDATE_NUSCENES_INFOS_EXPECT="${VALIDATE_NUSCENES_INFOS_EXPECT:-v1.0-trainval}"

export CHECKPOINT="${CHECKPOINT:-${CKPT:-work_dirs/streampetr_ttc_v3_frozen_20e_4gpu/latest.pth}}"

export MAX_BATCHES="${MAX_BATCHES:-1000}"
export BREAKDOWN_MAX_BATCHES="${BREAKDOWN_MAX_BATCHES:-1000}"

echo "[eval full] NUSCENES_ROOT=${NUSCENES_ROOT}  (set DATA_FULL or NUSCENES_ROOT to override auto-detect)"
echo "[eval full] ANN_FILE=${ANN_FILE}"
echo "[eval full] STREAMPETR_TTC_PKL=${STREAMPETR_TTC_PKL}"
echo "[eval full] CHECKPOINT=${CHECKPOINT}"
echo "[eval full] MAX_BATCHES=${MAX_BATCHES}  BREAKDOWN_MAX_BATCHES=${BREAKDOWN_MAX_BATCHES}  (set to 0 for full val)"
echo "[eval full] SKIP_COMPARE_SCENE=${SKIP_COMPARE_SCENE}  COMPARE_NUM_SCENES=${COMPARE_NUM_SCENES}  COMPARE_VIDEO=${COMPARE_VIDEO:-1}"

_eval_inner=""
if [[ -n "${EVAL_SCRIPT:-}" ]]; then
  [[ -f "${EVAL_SCRIPT}" ]] || { echo "FATAL: EVAL_SCRIPT=${EVAL_SCRIPT} not found." >&2; exit 1; }
  _eval_inner="${EVAL_SCRIPT}"
else
  for _c in "./run_eval_ttc_mlp.sh" "./slurm_scripts/run_eval_ttc_mlp.sh"; do
    if [[ -f "${_c}" ]]; then
      _eval_inner="${_c}"
      break
    fi
  done
fi
[[ -n "${_eval_inner}" ]] || {
  echo "FATAL: run_eval_ttc_mlp.sh not found in ${_REPO} (tried ./run_eval_ttc_mlp.sh and ./slurm_scripts/run_eval_ttc_mlp.sh)." >&2
  echo "  Copy it next to this repo layout, or: EVAL_SCRIPT=/path/to/run_eval_ttc_mlp.sh sbatch ..." >&2
  exit 1
}
echo "[eval full] using ${_eval_inner}"
exec bash "${_eval_inner}" "$@"
