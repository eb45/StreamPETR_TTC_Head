#!/bin/bash
# TTC MLP eval — nuScenes MINI (v1.0-mini): same val infos + TTC pickle as ./data/nuscenes mini prep.
# For full trainval eval, use run_eval_ttc_mlp_full.sh
#
# (1) tools/eval_ttc_mlp.py mean loss + bar chart
# (2) tools/eval_ttc_breakdown.py per-class / per–GT-bin MAE (JSON + PNG)
# (3) optional: tools/compare_ttc_scene.py — per-scene GT vs physics vs MLP (see COMPARE_NUM_SCENES / SCENE_TOKENS)
#
# Slurm (from repo root):
#   sbatch run_eval_ttc_mlp.sh
#   CHECKPOINT=work_dirs/.../iter_XXX.pth MAX_BATCHES=100 sbatch run_eval_ttc_mlp.sh
#
# Per-scene compare (step 3): default SCENE_TOKEN is the second val scene (see SCENE_TOKEN below).
#   Other scene on this val split: 325cef682f064c55a255f2625c533b75
#   SKIP_COMPARE_SCENE=1 — skip step 3 entirely
#
# Interactive / salloc GPU session (repo root; uses same conda + env as below):
#   bash run_eval_ttc_mlp.sh
#
# Optional env: VALIDATE_NUSCENES_INFOS_EXPECT=v1.0-trainval|v1.0-mini — after conda, checks train+val
#   infos pkls metadata.version (set by run_eval_ttc_mlp_full.sh). SKIP_VALIDATE_NUSCENES_INFOS=1 to disable.
# Optional env: CONFIG, CHECKPOINT (or CKPT), ANN_FILE, MAX_BATCHES, GPU_ID,
#   MAX_BATCHES — default 80; use 0 for **full** val dataloader (all batches / all scenes in split).
#   BREAKDOWN_MAX_BATCHES (default 50; use 0 for full val); breakdown can be slower than loss-only eval.
#   SKIP_TTC_BREAKDOWN=1 to skip eval_ttc_breakdown.py (still runs compare_ttc_scene if SCENE_TOKEN is set),
#   SCENE_TOKEN — single scene (overrides COMPARE_NUM_SCENES list if set alone)
#   SCENE_TOKENS — comma-separated list of scene tokens (overrides COMPARE_NUM_SCENES)
#   COMPARE_NUM_SCENES — if no SCENE_TOKEN/SCENE_TOKENS, take first N unique val scenes from ANN_FILE (default 1)
#   COMPARE_VIDEO_ALL_SCENES — default 0: only first compared scene gets MP4; set 1 for video on every scene
#   SKIP_ABLATION_TABLE — default 0: write ttc_eval_ablation.md/json under checkpoint dir
#   PRETRAINED_BASELINE, COMPARE_SAVE_DIR, SKIP_COMPARE_SCENE,
#   NUSCENES_ROOT or COMPARE_DATA_ROOT — passed as --data-root to eval_ttc_mlp, eval_ttc_breakdown,
#     and compare_ttc_scene (dataset + CAM_FRONT PNGs). Default: ./data/nuscenes (mini).
#   MAX_CAM_PANELS — optional; default 6 (camera + TTC dot rows in ttc_cam_front_panels.png)
#   compare_ttc_scene video (defaults ON; set COMPARE_VIDEO=0 to skip MP4):
#     COMPARE_VIDEO — default 1: --video (MP4; needs OpenCV + nuScenes --data-root)
#     VIDEO_PANELS — default gt,physics,mlp (side-by-side columns; use VIDEO_PANELS=mlp for single column)
#     VIDEO_FPS, VIDEO_PATH, VIDEO_MAX_WIDTH — optional overrides
#     CAM_BBOX_TTC — optional --cam-bbox-ttc (PNG panels; video columns use VIDEO_PANELS)
#   EVAL_LD_NO_USR64=1, STREAMPETR_KEEP_CUDA_HOME=1, USE_SRUN=1
#
# Does not conda install/remove; only sets env for this process.

#SBATCH --job-name=eval_ttc_mlp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=common
# RTX 2080 / 2080 Ti class. If sbatch fails: sinfo -p common -o "%G" — or use #SBATCH --gres=gpu:1
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Pin to one node (optional): sacct -j JOBID -X -n -o NodeList  then:
#   sbatch --nodelist=hostname run_eval_ttc_mlp.sh
# #SBATCH --nodelist=REPLACE_WITH_NODE_FROM_SACCT

set -euo pipefail
mkdir -p logs
# Slurm copies the batch script to /var/spool/slurmd/... — BASH_SOURCE is not the repo. Use submit dir.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  _REPO="${SLURM_SUBMIT_DIR}"
else
  _REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${_REPO}"

for _modinit in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /hpc/group/naderilab/navid/miniconda3/etc/profile.d/modules.sh; do
  [[ -f "${_modinit}" ]] && source "${_modinit}" && break
done
if type module &>/dev/null; then
  module load cuda/11.8 2>/dev/null || echo "[eval] note: module load cuda/11.8 failed or unavailable; continuing without it."
else
  echo "[eval] note: no 'module' command — using conda env + NVIDIA driver only."
fi

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env
if [[ "${EVAL_LD_NO_USR64:-0}" != "1" ]]; then
  export LD_LIBRARY_PATH="/usr/lib64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ "${STREAMPETR_KEEP_CUDA_HOME:-0}" != "1" ]]; then
  if [[ -n "${CUDA_HOME:-}" ]]; then
    echo "[eval] unsetting CUDA_HOME=${CUDA_HOME} for conda PyTorch (set STREAMPETR_KEEP_CUDA_HOME=1 to keep)"
  fi
  unset CUDA_HOME
else
  [[ -n "${CUDA_HOME:-}" ]] && [[ -d "${CUDA_HOME}/lib64" ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$(pwd):$(pwd)/tools:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY

echo "=== Node ==="
echo "hostname=$(hostname)  SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}  SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

echo "=== GPU (before Python) ==="
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
  echo "[eval] set CUDA_VISIBLE_DEVICES=0 (was unset/empty)"
fi
echo "=== CUDA / LD (debug) ==="
env | grep -E '^(CUDA|NVIDIA|LD_LIBRARY)' || true

if ! nvidia-smi -L; then
  echo "FATAL: nvidia-smi failed. Run inside a GPU allocation." >&2
  exit 1
fi

python -u -c "import os; os.environ.setdefault('CUDA_MODULE_LOADING','LAZY'); import torch; print('[eval] torch cuda:', torch.cuda.is_available(), torch.cuda.device_count())" || {
  echo "FATAL: PyTorch cannot see CUDA." >&2
  exit 1
}

CONFIG="${CONFIG:-projects/configs/StreamPETR/stream_petr_vov_ttc_frozen_20e.py}"
CHECKPOINT="${CHECKPOINT:-${CKPT:-work_dirs/streampetr_ttc_frozen_20e/iter_140650.pth}}"
# Mini val infos (override ANN_FILE if your symlink/layout differs).
ANN_FILE="${ANN_FILE:-data/nuscenes/nuscenes2d_temporal_infos_val.pkl}"
# Mini: default ./data/nuscenes — set NUSCENES_ROOT if nuScenes lives elsewhere (passed as --data-root).
NUSCENES_ROOT_RESOLVED="${COMPARE_DATA_ROOT:-${NUSCENES_ROOT:-$(pwd)/data/nuscenes}}"
NUSCENES_ROOT_RESOLVED="${NUSCENES_ROOT_RESOLVED%/}/"
# Wrapper dir (e.g. data_full/) with real nuScenes in data_full/nuscenes/
if [[ ! -f "${NUSCENES_ROOT_RESOLVED}nuscenes2d_temporal_infos_train.pkl" ]] && [[ -f "${NUSCENES_ROOT_RESOLVED}nuscenes/nuscenes2d_temporal_infos_train.pkl" ]]; then
  NUSCENES_ROOT_RESOLVED="${NUSCENES_ROOT_RESOLVED}nuscenes/"
  echo "[eval] Using nested nuscenes/ for data-root: ${NUSCENES_ROOT_RESOLVED}"
fi
if [[ -n "${VALIDATE_NUSCENES_INFOS_EXPECT:-}" ]] && [[ "${SKIP_VALIDATE_NUSCENES_INFOS:-0}" != "1" ]]; then
  echo "[eval] Validating infos pkls expect=${VALIDATE_NUSCENES_INFOS_EXPECT} data_root=${NUSCENES_ROOT_RESOLVED}"
  python -u tools/validate_nuscenes_infos_split.py \
    --expect "${VALIDATE_NUSCENES_INFOS_EXPECT}" \
    --data-root "${NUSCENES_ROOT_RESOLVED}"
fi
# Default scene for tools/compare_ttc_scene.py — must exist in ANN_FILE (mini val has 2 scenes).
# Alternate mini val scene: 325cef682f064c55a255f2625c533b75  (override: SCENE_TOKEN=... bash run_eval_ttc_mlp.sh)
SCENE_TOKEN="${SCENE_TOKEN:-fcbccedd61424f1b85dcbf8f897f9754}"

# Only set if caller did not (e.g. run_eval_ttc_mlp_full.sh exports trainval path).
if [[ -z "${STREAMPETR_TTC_PKL:-}" ]] && [[ -f "$(pwd)/data/nuscenes/ttc_gt_labels_v1_0_mini.pkl" ]]; then
  export STREAMPETR_TTC_PKL="$(pwd)/data/nuscenes/ttc_gt_labels_v1_0_mini.pkl"
fi

COMPARE_VIDEO="${COMPARE_VIDEO:-1}"
VIDEO_PANELS="${VIDEO_PANELS:-gt,physics,mlp}"

echo "[eval] SCENE_TOKEN=${SCENE_TOKEN}  (SKIP_COMPARE_SCENE=${SKIP_COMPARE_SCENE:-0})  COMPARE_VIDEO=${COMPARE_VIDEO}  VIDEO_PANELS=${VIDEO_PANELS}"

cmd=(
  python -u tools/eval_ttc_mlp.py "${CONFIG}" "${CHECKPOINT}"
  --ann-file "${ANN_FILE}" --max-batches "${MAX_BATCHES:-80}" --gpu-id "${GPU_ID:-0}"
)
[[ -n "${NUSCENES_ROOT_RESOLVED}" ]] && cmd+=(--data-root "${NUSCENES_ROOT_RESOLVED}")
[[ -f ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth ]] && cmd+=(--pretrained-baseline ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth)

USE_SRUN="${USE_SRUN:-0}"
if [[ "${USE_SRUN}" == "1" ]] && command -v srun &>/dev/null; then
  srun "${cmd[@]}"
else
  "${cmd[@]}"
fi

if [[ "${SKIP_TTC_BREAKDOWN:-0}" == "1" ]]; then
  echo "[eval] SKIP_TTC_BREAKDOWN=1 — skipping tools/eval_ttc_breakdown.py"
else
  echo ""
  echo "=== TTC breakdown (per-class & GT-TTC bin MAE / RMSE) ==="
  BREAKDOWN_MAX_BATCHES="${BREAKDOWN_MAX_BATCHES:-50}"
  breakdown_cmd=(
    python -u tools/eval_ttc_breakdown.py "${CONFIG}" "${CHECKPOINT}"
    --ann-file "${ANN_FILE}"
    --max-batches "${BREAKDOWN_MAX_BATCHES}"
    --gpu-id "${GPU_ID:-0}"
  )
  [[ -n "${NUSCENES_ROOT_RESOLVED}" ]] && breakdown_cmd+=(--data-root "${NUSCENES_ROOT_RESOLVED}")
  # Same GT TTC pickle as eval_ttc_mlp (avoid resolving a different ttc_gt_labels*.pkl on disk).
  [[ -n "${STREAMPETR_TTC_PKL:-}" ]] && [[ -f "${STREAMPETR_TTC_PKL}" ]] && breakdown_cmd+=(--ttc-pkl "${STREAMPETR_TTC_PKL}")
  if [[ "${USE_SRUN}" == "1" ]] && command -v srun &>/dev/null; then
    srun "${breakdown_cmd[@]}"
  else
    "${breakdown_cmd[@]}"
  fi
fi

# Optional: per-scene compare (GT vs physics TTC vs MLP) — one or many scenes
if [[ "${SKIP_COMPARE_SCENE:-0}" != "1" ]]; then
  PRETRAINED_BASELINE="${PRETRAINED_BASELINE:-}"
  if [[ -z "${PRETRAINED_BASELINE}" ]] && [[ -f ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth ]]; then
    PRETRAINED_BASELINE="ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth"
  fi
  if [[ -z "${PRETRAINED_BASELINE}" ]] || [[ ! -f "${PRETRAINED_BASELINE}" ]]; then
    echo "[eval] FATAL: compare_ttc_scene needs --pretrained-baseline. Set PRETRAINED_BASELINE=/path/to/stream_petr_vov_flash_800_bs2_seq_24e.pth" >&2
    exit 1
  fi
  _ckpt_dir="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
    COMPARE_NUM_SCENES="${COMPARE_NUM_SCENES:-1}"
    declare -a _compare_scenes=()
    if [[ -n "${SCENE_TOKENS:-}" ]]; then
      IFS=',' read -r -a _compare_scenes <<< "${SCENE_TOKENS}"
    elif [[ -n "${SCENE_TOKEN:-}" ]]; then
      _compare_scenes=("${SCENE_TOKEN}")
    elif [[ -f "${ANN_FILE}" ]] && [[ "${COMPARE_NUM_SCENES}" =~ ^[0-9]+$ ]] && [[ "${COMPARE_NUM_SCENES}" -gt 0 ]]; then
      mapfile -t _compare_scenes < <(python -u -c "
import pickle, sys
n = int(sys.argv[1])
p = sys.argv[2]
with open(p, 'rb') as f:
    d = pickle.load(f)
seen = []
for info in d.get('infos', []):
    t = info.get('scene_token')
    if t and t not in seen:
        seen.append(t)
        if len(seen) >= n:
            break
for t in seen:
    print(t)
" "${COMPARE_NUM_SCENES}" "${ANN_FILE}" 2>/dev/null)
    fi

    if [[ ${#_compare_scenes[@]} -eq 0 ]]; then
      echo "[eval] compare_ttc_scene: no scenes (set SCENE_TOKEN, SCENE_TOKENS, or COMPARE_NUM_SCENES + ANN_FILE)"
    else
      echo ""
      echo "=== Per-scene TTC compare: ${#_compare_scenes[@]} scene(s) (tools/compare_ttc_scene.py) ==="
      _csi=0
      for _st in "${_compare_scenes[@]}"; do
        [[ -z "${_st}" ]] && continue
        _st="${_st// /}"
        _csi=$((_csi + 1))
        _save="${_ckpt_dir}/compare_ttc_scene_${_st}"
        echo "[eval] compare scene ${_csi}/${#_compare_scenes[@]} token=${_st} save_dir=${_save}"
        compare_cmd=(
          python -u tools/compare_ttc_scene.py "${CONFIG}" "${CHECKPOINT}"
          --pretrained-baseline "${PRETRAINED_BASELINE}"
          --ann-file "${ANN_FILE}"
          --scene-token "${_st}"
          --save-dir "${_save}"
          --gpu-id "${GPU_ID:-0}"
        )
        if [[ -n "${STREAMPETR_TTC_PKL:-}" ]] && [[ -f "${STREAMPETR_TTC_PKL}" ]]; then
          compare_cmd+=(--ttc-pkl "${STREAMPETR_TTC_PKL}")
        fi
        if [[ -n "${NUSCENES_ROOT_RESOLVED}" ]]; then
          compare_cmd+=(--data-root "${NUSCENES_ROOT_RESOLVED}")
        fi
        if [[ -n "${MAX_CAM_PANELS:-}" ]]; then
          compare_cmd+=(--max-cam-panels "${MAX_CAM_PANELS}")
        fi
        if [[ -n "${CAM_BBOX_TTC:-}" ]]; then
          compare_cmd+=(--cam-bbox-ttc "${CAM_BBOX_TTC}")
        fi
        if [[ "${COMPARE_VIDEO:-1}" == "1" ]]; then
          if [[ "${COMPARE_VIDEO_ALL_SCENES:-0}" == "1" ]] || [[ "${_csi}" -eq 1 ]]; then
            compare_cmd+=(--video)
            [[ -n "${VIDEO_PANELS:-}" ]] && compare_cmd+=(--video-panels "${VIDEO_PANELS}")
            [[ -n "${VIDEO_FPS:-}" ]] && compare_cmd+=(--video-fps "${VIDEO_FPS}")
            [[ -n "${VIDEO_PATH:-}" ]] && compare_cmd+=(--video-path "${VIDEO_PATH}")
            [[ -n "${VIDEO_MAX_WIDTH:-}" ]] && compare_cmd+=(--video-max-width "${VIDEO_MAX_WIDTH}")
          fi
        fi
        if [[ "${USE_SRUN}" == "1" ]] && command -v srun &>/dev/null; then
          srun "${compare_cmd[@]}"
        else
          "${compare_cmd[@]}"
        fi
      done

      if [[ "${SKIP_ABLATION_TABLE:-0}" != "1" ]]; then
        echo ""
        echo "=== TTC ablation summary (ttc_eval_ablation.md) ==="
        python -u tools/aggregate_ttc_eval_ablation.py --work-dir "${_ckpt_dir}" || echo "[eval] WARN: aggregate_ttc_eval_ablation.py failed" >&2
      fi
    fi
fi
