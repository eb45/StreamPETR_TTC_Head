#!/bin/bash
# Generate one scene MP4 via compare_ttc_scene.py (GT / physics / MLP panels).
#
# From repo root:
#   SCENE_TOKEN=<token> sbatch --export=ALL slurm_scripts/run_scene_mp4.sh
#
# Logs:
#   logs/scene_mp4_<jobid>.out
#   logs/scene_mp4_<jobid>.err

#SBATCH --job-name=scene_mp4
#SBATCH --output=logs/scene_mp4_%j.out
#SBATCH --error=logs/scene_mp4_%j.err
#SBATCH --partition=common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

set -euo pipefail
mkdir -p logs

# ===================== EDIT DEFAULTS HERE =====================
: "${SCENE_TOKEN:=}"  # required
: "${CONFIG:=src/projects/configs/StreamPETR_ttc_v3/stream_petr_vov_ttc_frozen_20e.py}"
: "${CHECKPOINT:=work_dirs/streampetr_ttc_v3_frozen_20e_4gpu/latest.pth}"
: "${PRETRAINED_BASELINE:=ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth}"
: "${COMPARE_SAMPLES_PER_GPU:=1}"   # recommended on 8-12GB GPUs
: "${VIDEO_PANELS:=gt,physics,mlp}" # or "mlp" for single-column
: "${VIDEO_FPS:=10}"
: "${VIDEO_MAX_WIDTH:=1280}"
: "${GPU_ID:=0}"
# =============================================================

if [[ -z "${SCENE_TOKEN}" ]]; then
  echo "FATAL: SCENE_TOKEN is required. Example:"
  echo "  SCENE_TOKEN=e60ef590e3614187b7800db3e5284e1a sbatch --export=ALL slurm_scripts/run_scene_mp4.sh"
  exit 1
fi

_root="${STREAMPETR_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$(cd "${_root}" && pwd)" || { echo "FATAL: cannot cd to ${_root}" >&2; exit 1; }
[[ -f src/tools/compare_ttc_scene.py ]] || {
  echo "FATAL: not at StreamPETR root (missing src/tools/compare_ttc_scene.py)." >&2
  exit 1
}

for _modinit in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /hpc/group/naderilab/navid/miniconda3/etc/profile.d/modules.sh; do
  [[ -f "${_modinit}" ]] && source "${_modinit}" && break
done
if type module &>/dev/null; then
  module load cuda/11.8 2>/dev/null || true
fi
if [[ -f /hpc/group/naderilab/navid/miniconda3/bin/activate ]]; then
  # shellcheck source=/dev/null
  source /hpc/group/naderilab/navid/miniconda3/bin/activate
  conda activate "${STREAMPETR_CONDA_ENV:-$HOME/eb408/CS372/streampetr_env}" || true
fi

if [[ "${STREAMPETR_KEEP_CUDA_HOME:-0}" != "1" ]]; then
  unset CUDA_HOME
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
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=0

NUSCENES_ROOT_RESOLVED="${NUSCENES_ROOT:-$(pwd)/data/nuscenes}"
NUSCENES_ROOT_RESOLVED="${NUSCENES_ROOT_RESOLVED%/}/"
if [[ ! -f "${NUSCENES_ROOT_RESOLVED}nuscenes2d_temporal_infos_train.pkl" ]] && [[ -f "${NUSCENES_ROOT_RESOLVED}nuscenes/nuscenes2d_temporal_infos_train.pkl" ]]; then
  NUSCENES_ROOT_RESOLVED="${NUSCENES_ROOT_RESOLVED}nuscenes/"
  echo "[scene_mp4] nested nuscenes/ detected: ${NUSCENES_ROOT_RESOLVED}"
fi
ANN_FILE="${ANN_FILE:-${NUSCENES_ROOT_RESOLVED}nuscenes2d_temporal_infos_val.pkl}"
STREAMPETR_TTC_PKL="${STREAMPETR_TTC_PKL:-${NUSCENES_ROOT_RESOLVED}ttc_gt_labels_v1_0_trainval.pkl}"

[[ -f "${CONFIG}" ]] || { echo "FATAL: CONFIG not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "FATAL: CHECKPOINT not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${PRETRAINED_BASELINE}" ]] || { echo "FATAL: PRETRAINED_BASELINE not found: ${PRETRAINED_BASELINE}" >&2; exit 1; }
[[ -f "${ANN_FILE}" ]] || { echo "FATAL: ANN_FILE not found: ${ANN_FILE}" >&2; exit 1; }
[[ -f "${STREAMPETR_TTC_PKL}" ]] || { echo "FATAL: STREAMPETR_TTC_PKL not found: ${STREAMPETR_TTC_PKL}" >&2; exit 1; }

_ckpt_dir="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
SAVE_DIR="${SAVE_DIR:-${_ckpt_dir}/compare_ttc_scene_${SCENE_TOKEN}}"
mkdir -p "${SAVE_DIR}"
VIDEO_PATH="${VIDEO_PATH:-${SAVE_DIR}/scene_ttc_front_compare.mp4}"

echo "============================================================"
echo "[scene_mp4] job=${SLURM_JOB_ID:-?} host=$(hostname)"
echo "[scene_mp4] SCENE_TOKEN=${SCENE_TOKEN}"
echo "[scene_mp4] CONFIG=${CONFIG}"
echo "[scene_mp4] CHECKPOINT=${CHECKPOINT}"
echo "[scene_mp4] PRETRAINED_BASELINE=${PRETRAINED_BASELINE}"
echo "[scene_mp4] ANN_FILE=${ANN_FILE}"
echo "[scene_mp4] TTC_PKL=${STREAMPETR_TTC_PKL}"
echo "[scene_mp4] DATA_ROOT=${NUSCENES_ROOT_RESOLVED}"
echo "[scene_mp4] VIDEO_PANELS=${VIDEO_PANELS} FPS=${VIDEO_FPS}"
echo "[scene_mp4] SAVE_DIR=${SAVE_DIR}"
echo "[scene_mp4] VIDEO_PATH=${VIDEO_PATH}"
echo "============================================================"

nvidia-smi -L || { echo "FATAL: nvidia-smi failed." >&2; exit 1; }
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'n=', torch.cuda.device_count())" || exit 1
python -u -c "import mmdet3d; print('mmdet3d OK')" || { echo "FATAL: mmdet3d import failed" >&2; exit 1; }

set -x
python -u src/tools/compare_ttc_scene.py "${CONFIG}" "${CHECKPOINT}" \
  --pretrained-baseline "${PRETRAINED_BASELINE}" \
  --ann-file "${ANN_FILE}" \
  --scene-token "${SCENE_TOKEN}" \
  --ttc-pkl "${STREAMPETR_TTC_PKL}" \
  --data-root "${NUSCENES_ROOT_RESOLVED}" \
  --save-dir "${SAVE_DIR}" \
  --gpu-id "${GPU_ID}" \
  --samples-per-gpu "${COMPARE_SAMPLES_PER_GPU}" \
  --video \
  --video-panels "${VIDEO_PANELS}" \
  --video-fps "${VIDEO_FPS}" \
  --video-max-width "${VIDEO_MAX_WIDTH}" \
  --video-path "${VIDEO_PATH}"
set +x

echo "[scene_mp4] done -> ${VIDEO_PATH}"
