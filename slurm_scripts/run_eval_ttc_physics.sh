#!/bin/bash
# Val-wide **physics TTC** (BEV closure on predicted 3D boxes+vel) via eval_ttc_breakdown.py --physics.
# Uses the **frozen StreamPETR detector** checkpoint, not a TTC-finetuned head.
#
# From repo root:
#   sbatch slurm_scripts/run_eval_ttc_physics.sh
#
# Set paths before sbatch (use `export` so Slurm keeps them, or `sbatch --export=ALL`):
#   export NUSCENES_ROOT=/path/to/nuscenes/
#   export PRETRAINED_DETECTOR=ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth
#   export TTC_MAX_BATCHES=0    # 0 = full val; else cap batches
#
# Outputs (default): JSON + PNG next to the detector checkpoint:
#   <dirname(PRETRAINED_DETECTOR)>/eval_ttc_physics_breakdown/ttc_physics_breakdown.json
#   <dirname(PRETRAINED_DETECTOR)>/eval_ttc_physics_breakdown/ttc_breakdown_by_class.png
# Override: SAVE_DIR=/path/to/dir  (json/png written there instead)
#
# Slurm logs:  logs/eval_ttc_physics_<jobid>.out  and  .err

#SBATCH --job-name=eval_ttc_physics
#SBATCH --output=logs/eval_ttc_physics_%j.out
#SBATCH --error=logs/eval_ttc_physics_%j.err
#SBATCH --partition=common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

set -euo pipefail
mkdir -p logs

_root="${STREAMPETR_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$(cd "${_root}" && pwd)" || { echo "FATAL: cannot cd to ${_root}" >&2; exit 1; }
[[ -f src/tools/eval_ttc_breakdown.py ]] || {
  echo "FATAL: not at StreamPETR root (missing src/tools/eval_ttc_breakdown.py). cd repo root and sbatch, or set STREAMPETR_REPO=" >&2
  exit 1
}

# --- conda / GPU env (edit if your cluster differs) ---
for _modinit in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /hpc/group/naderilab/navid/miniconda3/etc/profile.d/modules.sh; do
  [[ -f "${_modinit}" ]] && source "${_modinit}" && break
done
if type module &>/dev/null; then
  module load cuda/11.8 2>/dev/null || echo "[physics-eval] module load cuda/11.8 skipped or failed; continuing."
fi
if [[ -f /hpc/group/naderilab/navid/miniconda3/bin/activate ]]; then
  # shellcheck source=/dev/null
  source /hpc/group/naderilab/navid/miniconda3/bin/activate
  conda activate "${STREAMPETR_CONDA_ENV:-$HOME/eb408/CS372/streampetr_env}"
else
  echo "WARN: miniconda activate not found; ensure Python has torch + mmdet3d" >&2
fi
if [[ "${STREAMPETR_KEEP_CUDA_HOME:-0}" != "1" ]]; then
  if [[ -n "${CUDA_HOME:-}" ]]; then
    echo "[physics TTC] unsetting CUDA_HOME for conda PyTorch (set STREAMPETR_KEEP_CUDA_HOME=1 to keep)"
  fi
  unset CUDA_HOME
else
  [[ -n "${CUDA_HOME:-}" ]] && [[ -d "${CUDA_HOME}/lib64" ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ "${EVAL_LD_NO_USR64:-0}" != "1" ]]; then
  export LD_LIBRARY_PATH="/usr/lib64:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
fi
_PYP="$(pwd):$(pwd)/src:$(pwd)/src/tools"
[[ -d "$(pwd)/mmdetection3d/mmdet3d" ]] && _PYP="$(pwd)/mmdetection3d:${_PYP}"
export PYTHONPATH="${_PYP}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

echo "============================================================"
echo "[physics TTC] job_id=${SLURM_JOB_ID:-?}  host=$(hostname)  pwd=$(pwd)"
echo "[physics TTC] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"
nvidia-smi -L || { echo "FATAL: nvidia-smi failed" >&2; exit 1; }

# --- data / model (env overrides) ---
# Phase-3 config: head must define ttc_pairwise_physics_from_preds
CONFIG="${CONFIG:-src/projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py}"
# Frozen detector (same as TTC pretrain) — **not** work_dirs/.../latest TTC checkpoint
export PRETRAINED_DETECTOR="${PRETRAINED_DETECTOR:-${STREAMPETR_BASELINE:-ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth}}"

if [[ -n "${NUSCENES_ROOT:-}" ]]; then
  export NUSCENES_ROOT="${NUSCENES_ROOT%/}/"
else
  export NUSCENES_ROOT="$(pwd)/data/nuscenes/"
fi
if [[ ! -f "${NUSCENES_ROOT}nuscenes2d_temporal_infos_train.pkl" ]] && [[ -f "${NUSCENES_ROOT}nuscenes/nuscenes2d_temporal_infos_train.pkl" ]]; then
  export NUSCENES_ROOT="${NUSCENES_ROOT}nuscenes/"
  echo "[physics TTC] nested nuscenes/ — NUSCENES_ROOT=${NUSCENES_ROOT}"
fi

export ANN_FILE="${ANN_FILE:-${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl}"
export STREAMPETR_TTC_PKL="${STREAMPETR_TTC_PKL:-${NUSCENES_ROOT}ttc_gt_labels_v1_0_trainval.pkl}"
# Default 1000 to match other ablation/eval runs; set 0 for full val.
TTC_MAX_BATCHES="${TTC_MAX_BATCHES:-1000}"
export GPU_ID="${GPU_ID:-0}"
# Optional: set SAVE_DIR to a fixed dir (e.g. work_dirs/ttc_ablation_eval_cache/physics)
SAVE_DIR="${SAVE_DIR:-}"
if [[ -n "${SAVE_DIR}" ]]; then
  mkdir -p "${SAVE_DIR}"
fi

echo "[physics TTC] CONFIG=${CONFIG}"
echo "[physics TTC] PRETRAINED_DETECTOR=${PRETRAINED_DETECTOR}"
echo "[physics TTC] ANN_FILE=${ANN_FILE}"
echo "[physics TTC] NUSCENES_ROOT (data-root)=${NUSCENES_ROOT}"
echo "[physics TTC] STREAMPETR_TTC_PKL=${STREAMPETR_TTC_PKL}"
echo "[physics TTC] TTC_MAX_BATCHES=${TTC_MAX_BATCHES}  (0 = full val)"
echo "[physics TTC] SAVE_DIR=${SAVE_DIR:-<default: next to checkpoint>/eval_ttc_physics_breakdown/}"

if [[ ! -f "${PRETRAINED_DETECTOR}" ]]; then
  echo "FATAL: detector checkpoint not found: ${PRETRAINED_DETECTOR}" >&2
  exit 1
fi
if [[ ! -f "${ANN_FILE}" ]]; then
  echo "FATAL: ANN_FILE missing: ${ANN_FILE}" >&2
  exit 1
fi
if [[ ! -f "${STREAMPETR_TTC_PKL}" ]]; then
  echo "FATAL: TTC pickle missing: ${STREAMPETR_TTC_PKL}" >&2
  exit 1
fi

python -u -c "import os; import torch; print('[physics TTC] torch cuda:', torch.cuda.is_available())" || exit 1
python -u -c "import mmdet3d" || { echo "FATAL: mmdet3d import failed" >&2; exit 1; }

set -x
if [[ -n "${SAVE_DIR}" ]]; then
  python -u src/tools/eval_ttc_breakdown.py \
    "${CONFIG}" \
    "${PRETRAINED_DETECTOR}" \
    --ann-file "${ANN_FILE}" \
    --ttc-pkl "${STREAMPETR_TTC_PKL}" \
    --data-root "${NUSCENES_ROOT}" \
    --physics \
    --max-batches "${TTC_MAX_BATCHES}" \
    --gpu-id "${GPU_ID}" \
    --save-dir "${SAVE_DIR}"
else
  python -u src/tools/eval_ttc_breakdown.py \
    "${CONFIG}" \
    "${PRETRAINED_DETECTOR}" \
    --ann-file "${ANN_FILE}" \
    --ttc-pkl "${STREAMPETR_TTC_PKL}" \
    --data-root "${NUSCENES_ROOT}" \
    --physics \
    --max-batches "${TTC_MAX_BATCHES}" \
    --gpu-id "${GPU_ID}"
fi
set +x

if [[ -n "${SAVE_DIR}" ]]; then
  _OUT_JSON="${SAVE_DIR}/ttc_physics_breakdown.json"
else
  _CKD="$(cd "$(dirname "${PRETRAINED_DETECTOR}")" && pwd)"
  _OUT_JSON="${_CKD}/eval_ttc_physics_breakdown/ttc_physics_breakdown.json"
fi
echo "============================================================"
echo "[physics TTC] done. Main metric file (if run succeeded):"
echo "  ${_OUT_JSON}"
echo "Slurm stdout log:    logs/eval_ttc_physics_${SLURM_JOB_ID}.out"
echo "Slurm stderr log:    logs/eval_ttc_physics_${SLURM_JOB_ID}.err"
echo "============================================================"
