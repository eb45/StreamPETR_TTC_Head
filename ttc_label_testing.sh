#!/bin/bash
#SBATCH --job-name=streampetr_ttc_phase2
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase 2 pipeline:
#   1) GT TTC pickle (CPU)
#   2) StreamPETR eval + save preds (GPU)
#   3) Heuristic / linear baselines (CPU)
#   4) TTC visuals: 6 cameras + BEV panel by default (*_ttc_panel.png) (CPU)
#
# Requires: nuScenes under ${NUSCENES_ROOT},
#           temporal infos pkl matching config val split,
#           ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth
#
# Optional env overrides:
#   RUN_TTC_BEV=0           — skip step 4
#   TTC_BEV_OUT=./work_dirs/ttc_bev_gt
#   TTC_BEV_MAX_SAMPLES=0  — 0 = all samples in DB version (mini is small)
#   TTC_BEV_WITH_CAMERAS=0 — BEV only (*_ttc_bev.png); default is 1 (panel + cams)

set -euo pipefail
mkdir -p logs work_dirs

cd ~/eb408/CS372/StreamPETR

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env

# With `set -u`, never expand unset PYTHONPATH (common in batch jobs).
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# --- paths (edit if your tree differs) ---
NUSCENES_ROOT="${NUSCENES_ROOT:-./data/nuscenes}"
INFO_VAL="${INFO_VAL:-${NUSCENES_ROOT}/nuscenes2d_temporal_infos_val.pkl}"
TTC_PKL="${TTC_PKL:-${NUSCENES_ROOT}/ttc_gt_labels_v1_0_mini.pkl}"
PREDS_PKL="${PREDS_PKL:-./work_dirs/mini_val_preds_ttc.pkl}"
NUSCENES_VER="${NUSCENES_VER:-v1.0-mini}"
TTC_BEV_OUT="${TTC_BEV_OUT:-./work_dirs/ttc_bev_gt}"
TTC_BEV_MAX_SAMPLES="${TTC_BEV_MAX_SAMPLES:-0}"
TTC_BEV_WITH_CAMERAS="${TTC_BEV_WITH_CAMERAS:-1}"
RUN_TTC_BEV="${RUN_TTC_BEV:-1}"

# Step 1 — ground-truth TTC labels (CPU only; skip if file exists)
if [[ ! -f "${TTC_PKL}" ]]; then
  echo "[1/4] Generating TTC labels -> ${TTC_PKL}"
  python tools/generate_ttc_labels.py \
    --data-root "${NUSCENES_ROOT}" \
    --version "${NUSCENES_VER}" \
    --out "${TTC_PKL}"
else
  echo "[1/4] Skip TTC labels (found ${TTC_PKL})"
fi

# Step 2 — detection eval + raw bbox pickle for baseline matching
echo "[2/4] Running detection eval and writing predictions -> ${PREDS_PKL}"
python tools/test.py \
  projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py \
  ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth \
  --eval bbox \
  --out "${PREDS_PKL}"

# Step 3 — heuristic + linear baselines vs GT TTC
echo "[3/4] TTC baselines"
python tools/ttc_heuristic_baseline.py \
  --data-root "${NUSCENES_ROOT}" \
  --info "${INFO_VAL}" \
  --ttc-labels "${TTC_PKL}" \
  --pred-results "${PREDS_PKL}" \
  --version "${NUSCENES_VER}"

# Step 4 — BEV images (ego center, objects colored by GT TTC)
if [[ "${RUN_TTC_BEV}" == "1" ]]; then
  echo "[4/4] BEV TTC visuals -> ${TTC_BEV_OUT}"
  _vb=(
    python tools/visualize_ttc_bev.py
    --data-root "${NUSCENES_ROOT}"
    --ttc-labels "${TTC_PKL}"
    --out-dir "${TTC_BEV_OUT}"
    --version "${NUSCENES_VER}"
    --max-samples "${TTC_BEV_MAX_SAMPLES}"
  )
  if [[ "${TTC_BEV_WITH_CAMERAS}" == "1" ]]; then
    _vb+=(--with-cameras)
  fi
  "${_vb[@]}"
else
  echo "[4/4] Skipped (RUN_TTC_BEV != 1)"
fi

echo "Done. Log: logs/${SLURM_JOB_NAME:-streampetr_ttc_phase2}_${SLURM_JOB_ID:-manual}.out"
echo "BEV PNGs: ${TTC_BEV_OUT}/"
