#!/bin/bash
#SBATCH --job-name=streampetr_ttc_phase3_dist
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Distributed Phase 3: frozen StreamPETR + TTC MLP (single node).
# Default request is 4 GPUs; for 8 GPUs submit with sbatch overrides:
#   sbatch --gres=gpu:8 --cpus-per-task=32 --mem=128G ttc_mlp_head_4gpu.sh
#SBATCH --partition=common
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs work_dirs

cd ~/eb408/CS372/StreamPETR

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env

# Fail fast if this node cannot use CUDA.
echo "[gpu] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || {
    echo "[gpu] FATAL: nvidia-smi failed. Request a GPU job (--gres=gpu:<N>) on a GPU partition." >&2
    exit 1
  }
else
  echo "[gpu] WARN: nvidia-smi not in PATH (modules may need: module load cuda ...)"
fi
if ! python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available (torch.cuda.is_available() is False)")
_ = torch.zeros(1, device="cuda")
print("[gpu] torch OK:", torch.version.cuda, "|", torch.cuda.device_count(), "GPU(s)")
PY
then
  echo "[gpu] FATAL: PyTorch could not initialize CUDA." >&2
  exit 1
fi

export PYTHONPATH="$(pwd):$(pwd)/tools:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Data / model paths.
NUSCENES_ROOT="${NUSCENES_ROOT:-$(pwd)/data/nuscenes}"
NUSCENES_ROOT="${NUSCENES_ROOT%/}/"
NUSCENES_VER="${NUSCENES_VER:-v1.0-trainval}"
export NUSCENES_ROOT
export NUSCENES_VER

# Dist knobs (single node).
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-28130}"
# workers are per GPU process; 1 is safer for host RAM in TTC pipeline.
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
AUTO_PREP_DATA="${AUTO_PREP_DATA:-1}"

if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "FATAL: NUM_GPUS must be >= 1 (got ${NUM_GPUS})" >&2
  exit 1
fi
if [[ "${BATCH_SIZE}" -lt 1 ]]; then
  echo "FATAL: BATCH_SIZE must be >= 1 (got ${BATCH_SIZE})" >&2
  exit 1
fi

_default_ttc_pkl() {
  case "${NUSCENES_VER}" in
    v1.0-mini) echo "${NUSCENES_ROOT}ttc_gt_labels_v1_0_mini.pkl" ;;
    v1.0|v1.0-trainval) echo "${NUSCENES_ROOT}ttc_gt_labels_v1_0_trainval.pkl" ;;
    *) echo "${NUSCENES_ROOT}ttc_gt_labels_v1_0_trainval.pkl" ;;
  esac
}
export STREAMPETR_TTC_PKL="${STREAMPETR_TTC_PKL:-$(_default_ttc_pkl)}"
export STREAMPETR_LOAD_FROM="${STREAMPETR_LOAD_FROM:-$(pwd)/ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth}"
export USE_WANDB="${USE_WANDB:-1}"

INFOS_TRAIN="${NUSCENES_ROOT}/nuscenes2d_temporal_infos_train.pkl"

_create_data_version() {
  case "${NUSCENES_VER}" in
    v1.0-mini) echo v1.0-mini ;;
    v1.0|v1.0-trainval) echo v1.0 ;;
    *)
      echo "ttc_mlp_head_4gpu: NUSCENES_VER=${NUSCENES_VER} not supported for auto-prep" >&2
      return 1
      ;;
  esac
}

_ttc_gen_version() {
  case "${NUSCENES_VER}" in
    v1.0-mini) echo v1.0-mini ;;
    v1.0|v1.0-trainval) echo v1.0-trainval ;;
    v1.0-test) echo v1.0-test ;;
    *)
      echo "ttc_mlp_head_4gpu: NUSCENES_VER=${NUSCENES_VER} not supported for TTC generation" >&2
      return 1
      ;;
  esac
}

_infos_ok() {
  python -u -c "import mmcv, sys; mmcv.load(sys.argv[1])" "$1"
}

_ttc_pkl_ok() {
  python -u -c "import pickle, sys; p=sys.argv[1]; f=open(p,'rb'); pickle.load(f); f.close()" "$1"
}

run_create_nuscenes_infos() {
  local ver
  ver="$(_create_data_version)" || exit 1
  echo "[auto-prep] Running tools/create_data_nusc.py (version=${ver}, root=${NUSCENES_ROOT}) ..."
  _cd_args=(--root-path "${NUSCENES_ROOT}" --out-dir "${NUSCENES_ROOT}" --extra-tag nuscenes2d --version "${ver}")
  if [[ "${ver}" == v1.0 ]] && [[ "${CREATE_DATA_GEN_TEST:-0}" != "1" ]]; then
    _cd_args+=(--skip-test)
  fi
  python tools/create_data_nusc.py "${_cd_args[@]}"
}

run_generate_ttc_labels() {
  local gver
  gver="$(_ttc_gen_version)" || exit 1
  echo "[auto-prep] Running tools/generate_ttc_labels.py (version=${gver}) -> ${STREAMPETR_TTC_PKL}"
  python tools/generate_ttc_labels.py \
    --data-root "${NUSCENES_ROOT}" \
    --version "${gver}" \
    --out "${STREAMPETR_TTC_PKL}"
}

if [[ "${AUTO_PREP_DATA}" == "1" ]]; then
  if _infos_ok "${INFOS_TRAIN}" 2>/dev/null; then
    echo "[auto-prep] OK  nuscenes2d temporal infos: ${INFOS_TRAIN}"
  else
    echo "[auto-prep] Missing or invalid ${INFOS_TRAIN}; regenerating nuScenes infos."
    run_create_nuscenes_infos
    _infos_ok "${INFOS_TRAIN}" || { echo "[auto-prep] Still cannot load ${INFOS_TRAIN} after create_data_nusc."; exit 1; }
    echo "[auto-prep] OK  nuscenes2d temporal infos: ${INFOS_TRAIN}"
  fi

  if _ttc_pkl_ok "${STREAMPETR_TTC_PKL}" 2>/dev/null; then
    echo "[auto-prep] OK  TTC labels pickle: ${STREAMPETR_TTC_PKL}"
  else
    echo "[auto-prep] Missing or invalid ${STREAMPETR_TTC_PKL}; generating TTC labels."
    run_generate_ttc_labels
    _ttc_pkl_ok "${STREAMPETR_TTC_PKL}" || { echo "[auto-prep] TTC pickle still invalid after generate_ttc_labels."; exit 1; }
    echo "[auto-prep] OK  TTC labels pickle: ${STREAMPETR_TTC_PKL}"
  fi
else
  [[ -f "${STREAMPETR_TTC_PKL}" ]] || { echo "Missing TTC pickle: ${STREAMPETR_TTC_PKL}"; exit 1; }
  _infos_ok "${INFOS_TRAIN}" || { echo "Cannot load ${INFOS_TRAIN}; fix the file or set AUTO_PREP_DATA=1."; exit 1; }
  echo "[auto-prep] OK  nuscenes2d temporal infos: ${INFOS_TRAIN}"
  _ttc_pkl_ok "${STREAMPETR_TTC_PKL}" || { echo "Cannot load TTC pickle: ${STREAMPETR_TTC_PKL}"; exit 1; }
  echo "[auto-prep] OK  TTC labels pickle: ${STREAMPETR_TTC_PKL}"
fi

# Keep schedule consistent across different GPU counts.
NUM_ITERS_PER_EPOCH=$(( TRAIN_SAMPLES / (NUM_GPUS * BATCH_SIZE) ))
if [[ "${NUM_ITERS_PER_EPOCH}" -lt 1 ]]; then
  echo "FATAL: computed NUM_ITERS_PER_EPOCH=${NUM_ITERS_PER_EPOCH}. Adjust NUM_GPUS/BATCH_SIZE/TRAIN_SAMPLES." >&2
  exit 1
fi
MAX_ITERS=$(( NUM_ITERS_PER_EPOCH * NUM_EPOCHS ))

WORK_DIR="${WORK_DIR:-./work_dirs/streampetr_ttc_frozen_20e_${NUM_GPUS}gpu}"
RUN_TRAIN_CURVE_PLOT="${RUN_TRAIN_CURVE_PLOT:-1}"
RUN_TTC_BEV="${RUN_TTC_BEV:-0}"
TTC_BEV_OUT="${TTC_BEV_OUT:-${WORK_DIR}/ttc_bev_gt}"
TTC_BEV_WITH_CAMERAS="${TTC_BEV_WITH_CAMERAS:-1}"
TTC_BEV_MAX_SAMPLES="${TTC_BEV_MAX_SAMPLES:-0}"

CONFIG="projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py"
echo "[train] dist config=${CONFIG} gpus=${NUM_GPUS} batch_size=${BATCH_SIZE} iters/epoch=${NUM_ITERS_PER_EPOCH} max_iters=${MAX_ITERS}"
bash tools/dist_train.sh "${CONFIG}" "${NUM_GPUS}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options \
    num_gpus="${NUM_GPUS}" \
    batch_size="${BATCH_SIZE}" \
    num_iters_per_epoch="${NUM_ITERS_PER_EPOCH}" \
    runner.max_iters="${MAX_ITERS}" \
    checkpoint_config.interval="${NUM_ITERS_PER_EPOCH}" \
    evaluation.interval="${NUM_ITERS_PER_EPOCH}" \
    data.train.data_root="${NUSCENES_ROOT}" \
    data.train.ann_file="${NUSCENES_ROOT}nuscenes2d_temporal_infos_train.pkl" \
    data.val.data_root="${NUSCENES_ROOT}" \
    data.val.ann_file="${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl" \
    data.test.data_root="${NUSCENES_ROOT}" \
    data.test.ann_file="${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl" \
  "$@"

if [[ "${RUN_TRAIN_CURVE_PLOT}" == "1" ]]; then
  python tools/plot_workdir_train_curves.py --work-dir "${WORK_DIR}" --keys loss_ttc \
    || echo "Note: train_curves.png skipped (matplotlib or loss_ttc missing in log)."
fi

if [[ "${RUN_TTC_BEV}" == "1" ]]; then
  case "${NUSCENES_VER}" in
    v1.0) _bev_ver=v1.0-trainval ;;
    *) _bev_ver="${NUSCENES_VER}" ;;
  esac
  _b=(python tools/visualize_ttc_bev.py --data-root "${NUSCENES_ROOT}" --ttc-labels "${STREAMPETR_TTC_PKL}" \
    --out-dir "${TTC_BEV_OUT}" --version "${_bev_ver}" --max-samples "${TTC_BEV_MAX_SAMPLES}")
  [[ "${TTC_BEV_WITH_CAMERAS}" == "1" ]] && _b+=(--with-cameras)
  "${_b[@]}" || echo "Note: BEV visualization failed."
fi

echo "Done. work_dir=${WORK_DIR}  USE_WANDB=${USE_WANDB}"
