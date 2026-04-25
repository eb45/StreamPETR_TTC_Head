#!/bin/bash
# Run one-or-many scene TTC qualitative compare on Slurm (GT vs physics vs MLP).
#
# Usage:
#   1) Edit the "EDIT BELOW" section in this file, then:
#        sbatch slurm_scripts/run_compare_ttc_scene.sh
#   2) Or override at submit time:
#        SCENE_TOKENS=tok1,tok2,tok3 CHECKPOINT=... COMPARE_SAMPLES_PER_GPU=1 sbatch --export=ALL slurm_scripts/run_compare_ttc_scene.sh
#
# Logs:
#   logs/compare_ttc_scene_<jobid>.out
#   logs/compare_ttc_scene_<jobid>.err

#SBATCH --job-name=compare_ttc_scene
#SBATCH --output=logs/compare_ttc_scene_%j.out
#SBATCH --error=logs/compare_ttc_scene_%j.err
#SBATCH --partition=common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

set -euo pipefail
mkdir -p logs

# ===================== EDIT BELOW =====================
# Phase-3 TTC config.
: "${CONFIG:=src/projects/configs/StreamPETR_ttc_v3/stream_petr_vov_ttc_frozen_20e.py}"
# Trained TTC checkpoint (MLP).
: "${CHECKPOINT:=work_dirs/streampetr_ttc_v3_frozen_20e_4gpu/latest.pth}"
# Frozen detector checkpoint for physics baseline.
: "${PRETRAINED_BASELINE:=ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth}"
# Scene token to render/compare (used when SCENE_TOKENS is unset).
: "${SCENE_TOKEN:=e60ef590e3614187b7800db3e5284e1a}"
# Optional comma-separated scene list. If set, script loops over all tokens.
# Default below matches your qualitative table:
#   intersection: c525507ee2ef4c6d8bb64b0e0cf0dd32
#   straight-away: 40209c4e465d4b4e8341ebd52be0d842
#   parking-lot: e60ef590e3614187b7800db3e5284e1a
#   near-miss: 68e79a88244f447f993a72da444b29ba
: "${SCENE_TOKENS:=c525507ee2ef4c6d8bb64b0e0cf0dd32,40209c4e465d4b4e8341ebd52be0d842,e60ef590e3614187b7800db3e5284e1a,68e79a88244f447f993a72da444b29ba}"
# 1 is strongly recommended on 8–12GB GPUs to avoid OOM.
: "${COMPARE_SAMPLES_PER_GPU:=1}"
# Camera panels in PNG summary.
: "${MAX_CAM_PANELS:=6}"
# Box color source for camera overlays: none|gt|physics|mlp
: "${CAM_BBOX_TTC:=mlp}"
# Optional video
: "${COMPARE_VIDEO:=0}"
: "${VIDEO_PANELS:=gt,physics,mlp}"
: "${VIDEO_FPS:=10}"
# ======================================================

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
  echo "[compare] nested nuscenes/ detected: ${NUSCENES_ROOT_RESOLVED}"
fi
ANN_FILE="${ANN_FILE:-${NUSCENES_ROOT_RESOLVED}nuscenes2d_temporal_infos_val.pkl}"
STREAMPETR_TTC_PKL="${STREAMPETR_TTC_PKL:-${NUSCENES_ROOT_RESOLVED}ttc_gt_labels_v1_0_trainval.pkl}"
GPU_ID="${GPU_ID:-0}"

[[ -f "${CONFIG}" ]] || { echo "FATAL: CONFIG not found: ${CONFIG}" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "FATAL: CHECKPOINT not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -f "${PRETRAINED_BASELINE}" ]] || { echo "FATAL: PRETRAINED_BASELINE not found: ${PRETRAINED_BASELINE}" >&2; exit 1; }
[[ -f "${ANN_FILE}" ]] || { echo "FATAL: ANN_FILE not found: ${ANN_FILE}" >&2; exit 1; }
[[ -f "${STREAMPETR_TTC_PKL}" ]] || { echo "FATAL: STREAMPETR_TTC_PKL not found: ${STREAMPETR_TTC_PKL}" >&2; exit 1; }

_ckpt_dir="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
SAVE_DIR_BASE="${SAVE_DIR:-${_ckpt_dir}}"
mkdir -p "${SAVE_DIR_BASE}"

echo "============================================================"
echo "[compare] job=${SLURM_JOB_ID:-?} host=$(hostname)"
echo "[compare] CONFIG=${CONFIG}"
echo "[compare] CHECKPOINT=${CHECKPOINT}"
echo "[compare] PRETRAINED_BASELINE=${PRETRAINED_BASELINE}"
echo "[compare] SCENE_TOKEN(default)=${SCENE_TOKEN}"
echo "[compare] SCENE_TOKENS=${SCENE_TOKENS:-<unset>}"
echo "[compare] ANN_FILE=${ANN_FILE}"
echo "[compare] STREAMPETR_TTC_PKL=${STREAMPETR_TTC_PKL}"
echo "[compare] DATA_ROOT=${NUSCENES_ROOT_RESOLVED}"
echo "[compare] COMPARE_SAMPLES_PER_GPU=${COMPARE_SAMPLES_PER_GPU}"
echo "[compare] SAVE_DIR_BASE=${SAVE_DIR_BASE}"
echo "============================================================"

nvidia-smi -L || { echo "FATAL: nvidia-smi failed." >&2; exit 1; }
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), 'n=', torch.cuda.device_count())" || exit 1
python -u -c "import mmdet3d; print('mmdet3d OK')" || { echo "FATAL: mmdet3d import failed" >&2; exit 1; }

declare -a _scene_list=()
if [[ -n "${SCENE_TOKENS}" ]]; then
  IFS=',' read -r -a _scene_list <<< "${SCENE_TOKENS}"
else
  _scene_list=("${SCENE_TOKEN}")
fi

_n=0
for _st in "${_scene_list[@]}"; do
  _st="${_st// /}"
  [[ -z "${_st}" ]] && continue
  _n=$((_n + 1))
  _save="${SAVE_DIR_BASE}/compare_ttc_scene_${_st}"
  mkdir -p "${_save}"
  echo "[compare] scene ${_n}/${#_scene_list[@]} token=${_st} save_dir=${_save}"
  cmd=(
    python -u src/tools/compare_ttc_scene.py "${CONFIG}" "${CHECKPOINT}"
    --pretrained-baseline "${PRETRAINED_BASELINE}"
    --ann-file "${ANN_FILE}"
    --scene-token "${_st}"
    --ttc-pkl "${STREAMPETR_TTC_PKL}"
    --data-root "${NUSCENES_ROOT_RESOLVED}"
    --save-dir "${_save}"
    --gpu-id "${GPU_ID}"
    --max-cam-panels "${MAX_CAM_PANELS}"
    --cam-bbox-ttc "${CAM_BBOX_TTC}"
    --samples-per-gpu "${COMPARE_SAMPLES_PER_GPU}"
  )
  if [[ "${COMPARE_VIDEO}" == "1" ]]; then
    cmd+=(--video --video-panels "${VIDEO_PANELS}" --video-fps "${VIDEO_FPS}")
  fi

  set -x
  "${cmd[@]}"
  set +x
done

if [[ "${_n}" -eq 0 ]]; then
  echo "FATAL: no valid scene tokens. Set SCENE_TOKEN or SCENE_TOKENS." >&2
  exit 1
fi
echo "[compare] done (${_n} scene(s)) -> ${SAVE_DIR_BASE}"
