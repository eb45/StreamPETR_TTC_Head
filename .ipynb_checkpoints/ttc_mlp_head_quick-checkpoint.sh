#!/bin/bash
#SBATCH --job-name=streampetr_ttc_quick
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# One node, multi-GPU — must match projects/configs/StreamPETR/stream_petr_vov_ttc_quick.py (num_gpus=4).
# For 2 GPUs: change #SBATCH --gres, GPUS below, and set num_gpus + runner.max_iters in quick.py or --cfg-options.
#SBATCH --partition=common
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

# Phase 3 (quick): frozen StreamPETR + TTC head via stream_petr_vov_ttc_quick.py + distributed train.
# Same env pattern as slurm_ttc_phase3.sh; uses tools/dist_train.sh (PyTorch launcher), not single-GPU train.py.
#
# Optional env: AUTO_PREP_DATA, USE_WANDB, WORK_DIR, NUSCENES_*, STREAMPETR_*, RUN_TRAIN_CURVE_PLOT, RUN_TTC_BEV, etc.

set -euo pipefail
mkdir -p logs work_dirs

cd ~/eb408/CS372/StreamPETR

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env

export PYTHONPATH="$(pwd):$(pwd)/tools:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

NUSCENES_ROOT="${NUSCENES_ROOT:-$(pwd)/data/nuscenes}"
NUSCENES_VER="${NUSCENES_VER:-v1.0-mini}"
AUTO_PREP_DATA="${AUTO_PREP_DATA:-1}"

_default_ttc_pkl() {
  case "${NUSCENES_VER}" in
    v1.0-mini) echo "$(pwd)/data/nuscenes/ttc_gt_labels_v1_0_mini.pkl" ;;
    v1.0|v1.0-trainval) echo "$(pwd)/data/nuscenes/ttc_gt_labels_v1_0_trainval.pkl" ;;
    *) echo "$(pwd)/data/nuscenes/ttc_gt_labels_v1_0_mini.pkl" ;;
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
      echo "slurm_ttc_phase3_quick: NUSCENES_VER=${NUSCENES_VER} not supported for auto-prep" >&2
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
      echo "slurm_ttc_phase3_quick: NUSCENES_VER=${NUSCENES_VER} not supported for TTC generation" >&2
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
  python tools/create_data_nusc.py \
    --root-path "${NUSCENES_ROOT}" \
    --out-dir "${NUSCENES_ROOT}" \
    --extra-tag nuscenes2d \
    --version "${ver}"
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

WORK_DIR="${WORK_DIR:-./work_dirs/streampetr_ttc_quick}"
RUN_TRAIN_CURVE_PLOT="${RUN_TRAIN_CURVE_PLOT:-1}"
RUN_TTC_BEV="${RUN_TTC_BEV:-0}"
TTC_BEV_OUT="${TTC_BEV_OUT:-${WORK_DIR}/ttc_bev_gt}"
TTC_BEV_WITH_CAMERAS="${TTC_BEV_WITH_CAMERAS:-1}"
TTC_BEV_MAX_SAMPLES="${TTC_BEV_MAX_SAMPLES:-0}"

# Must match num_gpus in stream_petr_vov_ttc_quick.py (default 4).
GPUS="${GPUS:-4}"

echo "[train] dist_train.sh quick config  GPUS=${GPUS}  work-dir=${WORK_DIR}"
bash tools/dist_train.sh \
  projects/configs/StreamPETR/stream_petr_vov_ttc_quick.py \
  "${GPUS}" \
  --work-dir "${WORK_DIR}" \
  --autoscale_lr \
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
