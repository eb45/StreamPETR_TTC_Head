#!/bin/bash
# TTC MLP eval — nuScenes FULL trainval val split (v1.0-trainval).
# Wraps run_eval_ttc_mlp.sh. Data root resolution (first match wins):
#   1) NUSCENES_ROOT if already set
#   2) DATA_FULL — e.g. absolute path to your full nuScenes + pkls tree
#   3) <repo>/data_full if that directory exists
#   4) legacy default /work/eb408/nuscenes/train (HPC)
# Required: nuScenes root directory that contains samples/ and the *_infos_*.pkl files.
# Many setups use an extra wrapper folder (e.g. data_full/) with the real tree in data_full/nuscenes/;
# this script auto-switches to .../nuscenes/ when train pkl is only there.
#
# Slurm:
#   sbatch run_eval_ttc_mlp_full.sh
#   DATA_FULL=/path/to/nuscenes_full CHECKPOINT=work_dirs/streampetr_ttc_frozen_20e_4gpu/latest.pth sbatch run_eval_ttc_mlp_full.sh
#
# Optional env: same as run_eval_ttc_mlp.sh, plus:
#   DATA_FULL — shortcut; sets NUSCENES_ROOT for full trainval eval (overridden by NUSCENES_ROOT)
#   NUSCENES_ROOT — explicit nuScenes root (must contain nuscenes2d_temporal_infos_val.pkl + ttc_gt_labels_v1_0_trainval.pkl)
#   CHECKPOINT / CKPT — default: latest 4-GPU Phase-3 run (override for other work_dirs)
#   MAX_BATCHES, BREAKDOWN_MAX_BATCHES — default 1000 each (first 1000 val minibatches).
#     Use MAX_BATCHES=0 BREAKDOWN_MAX_BATCHES=0 for **entire** val split (all batches).
#   SKIP_COMPARE_SCENE — default 0: run compare_ttc_scene. Set to 1 to skip.
#   COMPARE_NUM_SCENES — default 5: first N unique val scenes from ANN_FILE (override SCENE_TOKEN / SCENE_TOKENS).
#   SCENE_TOKEN — single scene (if COMPARE_NUM_SCENES=1 and unset, first scene in ANN_FILE).
#   SCENE_TOKENS — comma-separated explicit list (overrides COMPARE_NUM_SCENES).
#   COMPARE_VIDEO_ALL_SCENES — default 0: MP4 only for first scene; 1 = video for every scene.
#   SKIP_ABLATION_TABLE — default 0: write ttc_eval_ablation.md/json after compare.
#   SKIP_VALIDATE_NUSCENES_INFOS=1 — skip train/val infos version check (e.g. old pkls without metadata)
#
# Does not conda install/remove; delegates to run_eval_ttc_mlp.sh

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
# Slurm copies the batch script to /var/spool/slurmd/... — BASH_SOURCE is not the repo. Use submit dir.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _REPO="${SLURM_SUBMIT_DIR}"
else
  _REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${_REPO}"

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
# If pkls are under <root>/nuscenes/ (not directly under <root>/), use the inner folder.
if [[ ! -f "${NUSCENES_ROOT}nuscenes2d_temporal_infos_train.pkl" ]] && [[ -f "${NUSCENES_ROOT}nuscenes/nuscenes2d_temporal_infos_train.pkl" ]]; then
  export NUSCENES_ROOT="${NUSCENES_ROOT}nuscenes/"
  echo "[eval full] Detected nested nuscenes/ — using NUSCENES_ROOT=${NUSCENES_ROOT}"
fi

export ANN_FILE="${ANN_FILE:-${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl}"
export STREAMPETR_TTC_PKL="${STREAMPETR_TTC_PKL:-${NUSCENES_ROOT}ttc_gt_labels_v1_0_trainval.pkl}"

# Per-scene compare + optional CAM_FRONT MP4 (delegates to run_eval_ttc_mlp.sh → compare_ttc_scene.py).
export SKIP_COMPARE_SCENE="${SKIP_COMPARE_SCENE:-0}"
export COMPARE_VIDEO="${COMPARE_VIDEO:-1}"
export COMPARE_NUM_SCENES="${COMPARE_NUM_SCENES:-5}"
export SKIP_ABLATION_TABLE="${SKIP_ABLATION_TABLE:-0}"
# If user did not set SCENE_TOKEN / SCENE_TOKENS, run_eval_ttc_mlp.sh picks first COMPARE_NUM_SCENES scenes from ANN_FILE.
# For a single explicit scene: SCENE_TOKEN=... COMPARE_NUM_SCENES=1 or pass one token via SCENE_TOKENS.

# Fail fast if train/val infos are mini (or wrong split) — checks metadata.version in both pkls.
export VALIDATE_NUSCENES_INFOS_EXPECT="${VALIDATE_NUSCENES_INFOS_EXPECT:-v1.0-trainval}"

# Checkpoint: run_eval_ttc_mlp.sh defaults to a fixed iter_*.pth unless CHECKPOINT is set — set full-val default here.
export CHECKPOINT="${CHECKPOINT:-${CKPT:-work_dirs/streampetr_ttc_frozen_20e_4gpu/latest.pth}}"

# 0 = entire val dataloader; default 1000 minibatches (see tools/eval_ttc_mlp.py / eval_ttc_breakdown.py).
export MAX_BATCHES="${MAX_BATCHES:-1000}"
export BREAKDOWN_MAX_BATCHES="${BREAKDOWN_MAX_BATCHES:-1000}"

echo "[eval full] NUSCENES_ROOT=${NUSCENES_ROOT}  (set DATA_FULL or NUSCENES_ROOT to override auto-detect)"
echo "[eval full] ANN_FILE=${ANN_FILE}"
echo "[eval full] STREAMPETR_TTC_PKL=${STREAMPETR_TTC_PKL}"
echo "[eval full] CHECKPOINT=${CHECKPOINT}"
echo "[eval full] MAX_BATCHES=${MAX_BATCHES}  BREAKDOWN_MAX_BATCHES=${BREAKDOWN_MAX_BATCHES}  (set to 0 for full val)"
echo "[eval full] SKIP_COMPARE_SCENE=${SKIP_COMPARE_SCENE}  COMPARE_NUM_SCENES=${COMPARE_NUM_SCENES}  COMPARE_VIDEO=${COMPARE_VIDEO:-1}"

exec bash "${_REPO}/run_eval_ttc_mlp.sh" "$@"
