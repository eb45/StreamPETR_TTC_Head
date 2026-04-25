#!/bin/bash
# Run 3 TTC eval_ttc_breakdown jobs in one Slurm allocation:
#   1) Physics baseline (detector ckpt + --physics)
#   2) MLP v1 (query embeddings)
#   3) MLP v3 (query + velocity)
#
# From repo root:
#   sbatch slurm_scripts/run_ttc_ablation_3way.sh
#
# Logs:
#   logs/ttc_ablation_3way_<jobid>.out
#   logs/ttc_ablation_3way_<jobid>.err

#SBATCH --job-name=ttc_ablation_3way
#SBATCH --output=logs/ttc_ablation_3way_%j.out
#SBATCH --error=logs/ttc_ablation_3way_%j.err
#SBATCH --partition=common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs

# ===================== EDIT BELOW =====================
: "${TTC_MAX_BATCHES:=1000}"   # keep 1000 for apples-to-apples; use 0 for full val
: "${GPU_ID:=0}"

# Data / labels
: "${NUSCENES_ROOT:=}"
: "${ANN_FILE:=}"
: "${STREAMPETR_TTC_PKL:=}"

# Configs / checkpoints
: "${CONFIG_V1:=src/projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py}"
: "${CONFIG_V3:=src/projects/configs/StreamPETR_ttc_v3/stream_petr_vov_ttc_frozen_20e.py}"
: "${PRETRAINED_DETECTOR:=ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth}"
: "${CKPT_V1:=work_dirs/streampetr_ttc_frozen_20e/latest.pth}"
: "${CKPT_V3:=work_dirs/streampetr_ttc_v3_frozen_20e_4gpu/latest.pth}"

# Output root (three subdirs created: physics, mlp_v1, mlp_v3)
: "${OUT_CACHE:=work_dirs/ttc_ablation_eval_cache}"
# ======================================================

_root="${STREAMPETR_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$(cd "${_root}" && pwd)" || { echo "FATAL: cannot cd to ${_root}" >&2; exit 1; }
[[ -f src/tools/eval_ttc_breakdown.py ]] || {
  echo "FATAL: not at StreamPETR root (missing src/tools/eval_ttc_breakdown.py)" >&2
  exit 1
}

for _modinit in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /hpc/group/naderilab/navid/miniconda3/etc/profile.d/modules.sh; do
  [[ -f "${_modinit}" ]] && source "${_modinit}" && break
done
if type module &>/dev/null; then
  module load cuda/11.8 2>/dev/null || echo "[ablation] note: module load cuda/11.8 failed/unavailable; continuing."
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

if [[ -n "${NUSCENES_ROOT}" ]]; then
  NUSCENES_ROOT="${NUSCENES_ROOT%/}/"
else
  NUSCENES_ROOT="$(pwd)/data/nuscenes/"
fi
if [[ ! -f "${NUSCENES_ROOT}nuscenes2d_temporal_infos_train.pkl" ]] && [[ -f "${NUSCENES_ROOT}nuscenes/nuscenes2d_temporal_infos_train.pkl" ]]; then
  NUSCENES_ROOT="${NUSCENES_ROOT}nuscenes/"
  echo "[ablation] nested nuscenes/ detected -> ${NUSCENES_ROOT}"
fi
if [[ -z "${ANN_FILE}" ]]; then
  ANN_FILE="${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl"
fi
if [[ -z "${STREAMPETR_TTC_PKL}" ]]; then
  STREAMPETR_TTC_PKL="${NUSCENES_ROOT}ttc_gt_labels_v1_0_trainval.pkl"
fi

mkdir -p "${OUT_CACHE}/physics" "${OUT_CACHE}/mlp_v1" "${OUT_CACHE}/mlp_v3"

echo "============================================================"
echo "[ablation] job=${SLURM_JOB_ID:-?} host=$(hostname)"
echo "[ablation] TTC_MAX_BATCHES=${TTC_MAX_BATCHES} GPU_ID=${GPU_ID}"
echo "[ablation] ANN_FILE=${ANN_FILE}"
echo "[ablation] STREAMPETR_TTC_PKL=${STREAMPETR_TTC_PKL}"
echo "[ablation] NUSCENES_ROOT=${NUSCENES_ROOT}"
echo "[ablation] PRETRAINED_DETECTOR=${PRETRAINED_DETECTOR}"
echo "[ablation] CKPT_V1=${CKPT_V1}"
echo "[ablation] CKPT_V3=${CKPT_V3}"
echo "[ablation] OUT_CACHE=${OUT_CACHE}"
echo "============================================================"

nvidia-smi -L || { echo "FATAL: nvidia-smi failed" >&2; exit 1; }
python -u -c "import torch; print('[ablation] torch cuda:', torch.cuda.is_available(), torch.cuda.device_count())" || exit 1
python -u -c "import mmdet3d; print('[ablation] mmdet3d OK')" || { echo "FATAL: mmdet3d import failed" >&2; exit 1; }

for _f in "${CONFIG_V1}" "${CONFIG_V3}" "${PRETRAINED_DETECTOR}" "${CKPT_V1}" "${CKPT_V3}" "${ANN_FILE}" "${STREAMPETR_TTC_PKL}"; do
  [[ -f "${_f}" ]] || { echo "FATAL: missing file: ${_f}" >&2; exit 1; }
done

set -x
# 1) Physics baseline
python -u src/tools/eval_ttc_breakdown.py \
  "${CONFIG_V1}" "${PRETRAINED_DETECTOR}" \
  --ann-file "${ANN_FILE}" \
  --ttc-pkl "${STREAMPETR_TTC_PKL}" \
  --data-root "${NUSCENES_ROOT}" \
  --max-batches "${TTC_MAX_BATCHES}" \
  --gpu-id "${GPU_ID}" \
  --physics \
  --save-dir "${OUT_CACHE}/physics"

# 2) MLP v1
python -u src/tools/eval_ttc_breakdown.py \
  "${CONFIG_V1}" "${CKPT_V1}" \
  --ann-file "${ANN_FILE}" \
  --ttc-pkl "${STREAMPETR_TTC_PKL}" \
  --data-root "${NUSCENES_ROOT}" \
  --max-batches "${TTC_MAX_BATCHES}" \
  --gpu-id "${GPU_ID}" \
  --save-dir "${OUT_CACHE}/mlp_v1"

# 3) MLP v3
python -u src/tools/eval_ttc_breakdown.py \
  "${CONFIG_V3}" "${CKPT_V3}" \
  --ann-file "${ANN_FILE}" \
  --ttc-pkl "${STREAMPETR_TTC_PKL}" \
  --data-root "${NUSCENES_ROOT}" \
  --max-batches "${TTC_MAX_BATCHES}" \
  --gpu-id "${GPU_ID}" \
  --save-dir "${OUT_CACHE}/mlp_v3"
set +x

echo "[ablation] done."
echo "  physics -> ${OUT_CACHE}/physics/ttc_physics_breakdown.json"
echo "  mlp_v1  -> ${OUT_CACHE}/mlp_v1/ttc_breakdown.json"
echo "  mlp_v3  -> ${OUT_CACHE}/mlp_v3/ttc_breakdown.json"
