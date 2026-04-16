#!/bin/bash
#SBATCH --job-name=streampetr_ttc_phase3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# GPU: pick a partition that matches your PyTorch build (same idea as slurm_ttc_phase2.sh).
# Default `common` avoids Hopper (H200) unless your env has a recent CUDA/torch stack.
# For scavenger-h200, set e.g.  #SBATCH --partition=scavenger-h200  and  #SBATCH --account=scavenger-h200
#SBATCH --partition=common
#SBATCH --gres=gpu:1
# TTC + LoadGTTC + nuScenes per worker is RAM-heavy; raise if workers OOM (see WORKERS_PER_GPU).
#SBATCH --mem=64G
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
#   WORKERS_PER_GPU — dataloader workers (default 2 in stream_petr_vov_ttc_frozen_20e.py). Use 0 or 1 if OOM.
#   SKIP_VALIDATE_NUSCENES_INFOS=1 — skip tools/validate_nuscenes_infos_split.py (e.g. pkls without metadata).
#   CREATE_DATA_GEN_TEST=1 — run create_data_nusc.py with v1.0-test (requires nuScenes test split on disk).

set -euo pipefail
mkdir -p logs work_dirs

cd ~/eb408/CS372/StreamPETR

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env

# Fail fast if this node cannot use CUDA (login nodes, missing --gres, bad driver, wrong partition).
# "Failed to get device handle for GPU 0" / RuntimeError: CUDA unknown error usually means no usable GPU here.
echo "[gpu] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || {
    echo "[gpu] FATAL: nvidia-smi failed. Request a GPU job: #SBATCH --gres=gpu:1 on a GPU partition; do not run train on a login node." >&2
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
print("[gpu] torch OK:", torch.version.cuda, "|", torch.cuda.get_device_name(0))
PY
then
  echo "[gpu] FATAL: PyTorch could not initialize CUDA. Match PyTorch+CUDA to the cluster driver/GPU type, or try another partition (e.g. not Hopper without a matching torch build)." >&2
  exit 1
fi

# Repo root + tools/ so tools/create_data_nusc.py can import data_converter.
export PYTHONPATH="$(pwd):$(pwd)/tools:${PYTHONPATH:-}"
# Flush Python print/log lines to Slurm *.out immediately (otherwise long silent gaps).
export PYTHONUNBUFFERED=1

# Full trainval: point at extracted nuScenes root (samples/, sweeps/, v1.0-trainval/). Edit if needed.
NUSCENES_ROOT="${NUSCENES_ROOT:-/work/eb408/nuscenes/train}"
NUSCENES_ROOT="${NUSCENES_ROOT%/}/"
NUSCENES_VER="${NUSCENES_VER:-v1.0-trainval}"
# Must be exported: stream_petr_vov_ttc_frozen_20e.py reads NUSCENES_* from os.environ at config load.
export NUSCENES_ROOT
export NUSCENES_VER
# Optional: override dataloader workers (integer). Lower = less host RAM per job.
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
AUTO_PREP_DATA="${AUTO_PREP_DATA:-1}"

# Default TTC path follows nuScenes split (override with STREAMPETR_TTC_PKL); lives next to infos under NUSCENES_ROOT.
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
  _cd_args=(--root-path "${NUSCENES_ROOT}" --out-dir "${NUSCENES_ROOT}" --extra-tag nuscenes2d --version "${ver}")
  # v1.0 also tries v1.0-test unless skipped — most users only have trainval (no v1.0-test/ on disk).
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

if [[ "${SKIP_VALIDATE_NUSCENES_INFOS:-0}" != "1" ]]; then
  _v_expect=""
  case "${NUSCENES_VER}" in
    v1.0-mini) _v_expect=v1.0-mini ;;
    v1.0|v1.0-trainval) _v_expect=v1.0-trainval ;;
  esac
  if [[ -n "${_v_expect}" ]]; then
    echo "[validate] train+val infos metadata.version should be ${_v_expect}"
    python -u tools/validate_nuscenes_infos_split.py --expect "${_v_expect}" --data-root "${NUSCENES_ROOT}"
  fi
fi

WORK_DIR="${WORK_DIR:-./work_dirs/streampetr_ttc_frozen_20e}"
RUN_TRAIN_CURVE_PLOT="${RUN_TRAIN_CURVE_PLOT:-1}"
RUN_TTC_BEV="${RUN_TTC_BEV:-0}"
TTC_BEV_OUT="${TTC_BEV_OUT:-${WORK_DIR}/ttc_bev_gt}"
TTC_BEV_WITH_CAMERAS="${TTC_BEV_WITH_CAMERAS:-1}"
TTC_BEV_MAX_SAMPLES="${TTC_BEV_MAX_SAMPLES:-0}"

# Config defaults to 1× GPU (see stream_petr_vov_ttc_frozen_20e.py). For multi-GPU, pass --cfg-options.
# Training logs every 50 iterations (log_config.interval). First log line appears after iter 50.
# Before that: loading checkpoint + building DataLoader + first batches can take many minutes with little output.
echo "[train] Starting tools/train.py  work-dir=${WORK_DIR}  data_root=${NUSCENES_ROOT}  (tail -f ${WORK_DIR}/*.log.json for JSON logs)"
# Config defaults to ./data/nuscenes/; override so training reads the same tree as NUSCENES_ROOT.
python tools/train.py projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py \
  --work-dir "${WORK_DIR}" \
  --launcher none \
  --cfg-options \
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
