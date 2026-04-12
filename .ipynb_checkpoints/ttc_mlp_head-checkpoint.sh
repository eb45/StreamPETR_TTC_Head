#!/bin/bash
#SBATCH --job-name=streampetr_ttc_phase3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# GPU: pick a partition that matches your PyTorch build (same idea as slurm_ttc_phase2.sh).
# Default `common` avoids Hopper (H200) unless your env has a recent CUDA/torch stack.
# For scavenger-h200, set e.g.  #SBATCH --partition=scavenger-h200  and  #SBATCH --account=scavenger-h200
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# Phase 3: frozen StreamPETR + TTC MLP train (+ optional W&B, matplotlib curve, BEV viz).
# Same env as slurm_ttc_phase2.sh (cd + conda). Edit paths below if your clone differs.
#
# Optional env:
#   AUTO_PREP_DATA=1 (default) — if nuScenes temporal infos or TTC pickle are missing or
#     unreadable, regenerate (tools/create_data_nusc.py, tools/generate_ttc_labels.py).
#     Set AUTO_PREP_DATA=0 to skip and fail fast if files are bad.
#   USE_WANDB, RUN_TRAIN_CURVE_PLOT, RUN_TTC_BEV, TTC_BEV_*, NUSCENES_ROOT, NUSCENES_VER,
#   STREAMPETR_TTC_PKL, STREAMPETR_LOAD_FROM, WORK_DIR, STREAMPETR_*.

set -euo pipefail
mkdir -p logs work_dirs

cd ~/eb408/CS372/StreamPETR

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env

# Repo root + tools/ so tools/create_data_nusc.py can import data_converter.
export PYTHONPATH="$(pwd):$(pwd)/tools:${PYTHONPATH:-}"
# Flush Python print/log lines to Slurm *.out immediately (otherwise long silent gaps).
export PYTHONUNBUFFERED=1

NUSCENES_ROOT="${NUSCENES_ROOT:-$(pwd)/data/nuscenes}"
NUSCENES_VER="${NUSCENES_VER:-v1.0-mini}"
AUTO_PREP_DATA="${AUTO_PREP_DATA:-1}"

# Default TTC path follows nuScenes split (override with STREAMPETR_TTC_PKL).
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

# create_data_nusc.py --version: v1.0 (full) or v1.0-mini only.
_create_data_version() {
  case "${NUSCENES_VER}" in
    v1.0-mini) echo v1.0-mini ;;
    v1.0|v1.0-trainval) echo v1.0 ;;
    *)
      echo "slurm_ttc_phase3: NUSCENES_VER=${NUSCENES_VER} not supported for auto-prep; use v1.0-mini, v1.0, or v1.0-trainval" >&2
      return 1
      ;;
  esac
}

# generate_ttc_labels.py --version: exact NuScenes version strings.
_ttc_gen_version() {
  case "${NUSCENES_VER}" in
    v1.0-mini) echo v1.0-mini ;;
    v1.0|v1.0-trainval) echo v1.0-trainval ;;
    v1.0-test) echo v1.0-test ;;
    *)
      echo "slurm_ttc_phase3: NUSCENES_VER=${NUSCENES_VER} not supported for TTC generation" >&2
      return 1
      ;;
  esac
}

# Load-only checks (no Python print — Slurm often line-buffers or drops tiny prints; we echo from bash).
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
  echo "[auto-prep] Note: full v1.0 can take a long time and needs the full nuScenes tree under data_root."
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

# Default output dir for tiered TTC loss + wider MLP (distinct from older streampetr_ttc_frozen_20e runs).
WORK_DIR="${WORK_DIR:-./work_dirs/streampetr_ttc_frozen_20e_tiered}"
RUN_TRAIN_CURVE_PLOT="${RUN_TRAIN_CURVE_PLOT:-1}"
RUN_TTC_BEV="${RUN_TTC_BEV:-0}"
TTC_BEV_OUT="${TTC_BEV_OUT:-${WORK_DIR}/ttc_bev_gt}"
TTC_BEV_WITH_CAMERAS="${TTC_BEV_WITH_CAMERAS:-1}"
TTC_BEV_MAX_SAMPLES="${TTC_BEV_MAX_SAMPLES:-0}"

# Config defaults to 1× GPU (see stream_petr_vov_ttc_frozen_20e.py). For multi-GPU, pass --cfg-options.
# Training logs every 50 iterations (log_config.interval). First log line appears after iter 50.
# Before that: loading checkpoint + building DataLoader + first batches can take many minutes with little output.
echo "[train] Starting tools/train.py  work-dir=${WORK_DIR}  (tail -f ${WORK_DIR}/*.log.json for JSON logs)"
python tools/train.py projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py \
  --work-dir "${WORK_DIR}" \
  --launcher none \
  "$@"

if [[ "${RUN_TRAIN_CURVE_PLOT}" == "1" ]]; then
  python tools/plot_workdir_train_curves.py --work-dir "${WORK_DIR}" --keys loss_ttc \
    || echo "Note: train_curves.png skipped (matplotlib or loss_ttc missing in log)."
fi

if [[ "${RUN_TTC_BEV}" == "1" ]]; then
  # NuScenes Python API uses v1.0-trainval / v1.0-mini, not the shorthand v1.0.
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
