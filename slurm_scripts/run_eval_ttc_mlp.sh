#!/bin/bash
# TTC eval (mini val). Full trainval: run_eval_ttc_mlp_full.sh
# Run from StreamPETR repo root:  cd /path/to/StreamPETR && sbatch run_eval_ttc_mlp.sh
# (Slurm sets SLURM_SUBMIT_DIR to that cwd.) Override: STREAMPETR_REPO=/path/to/StreamPETR
# Env: CONFIG, CHECKPOINT, ANN_FILE, NUSCENES_ROOT, MAX_BATCHES, SKIP_TTC_BREAKDOWN, SKIP_COMPARE_SCENE, SCENE_TOKEN, ...

#SBATCH --job-name=eval_ttc_mlp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail
mkdir -p logs

_root="${STREAMPETR_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "$(cd "${_root}" && pwd)" || { echo "FATAL: cannot cd to ${_root}" >&2; exit 1; }
[[ -f src/tools/eval_ttc_mlp.py ]] || {
  echo "FATAL: not at StreamPETR root (missing src/tools/eval_ttc_mlp.py). Run: cd /path/to/StreamPETR && sbatch run_eval_ttc_mlp.sh" >&2
  echo "  Or set STREAMPETR_REPO to the repo root." >&2
  exit 1
}

# set -u: optional env vars referenced below must have a default before first use
USE_SRUN="${USE_SRUN:-0}"
PRETRAINED_BASELINE="${PRETRAINED_BASELINE:-}"

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
# mmdet3d: use conda/site-packages if installed; else vendored MMDetection3D at repo root.
_PYP="$(pwd):$(pwd)/src:$(pwd)/src/tools"
[[ -d "$(pwd)/mmdetection3d/mmdet3d" ]] && _PYP="$(pwd)/mmdetection3d:${_PYP}"
export PYTHONPATH="${_PYP}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY

echo "[eval] host=$(hostname) job=${SLURM_JOB_ID:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

if ! nvidia-smi -L; then
  echo "FATAL: nvidia-smi failed. Run inside a GPU allocation." >&2
  exit 1
fi

python -u -c "import os; os.environ.setdefault('CUDA_MODULE_LOADING','LAZY'); import torch; print('[eval] torch cuda:', torch.cuda.is_available(), torch.cuda.device_count())" || {
  echo "FATAL: PyTorch cannot see CUDA." >&2
  exit 1
}
python -u -c "import mmdet3d; print('[eval] mmdet3d OK')" || {
  _root="$(pwd)"
  echo "FATAL: Python cannot import mmdet3d (MMDetection3D)." >&2
  echo "  Repo: ${_root}" >&2
  echo "  PYTHONPATH: ${PYTHONPATH:-}" >&2
  if [[ -d "${_root}/mmdetection3d/mmdet3d" ]]; then
    echo "  ${_root}/mmdetection3d exists but is not installed in this env — run: pip install -e ${_root}/mmdetection3d" >&2
  else
    echo "  Missing ${_root}/mmdetection3d — on a node with git+network, from repo root: bash scripts/ensure_mmdet3d.sh" >&2
    echo "  (see docs/setup.md: mmcv-full, mmdet, mmseg, then mmdet3d v1.0.0rc6)" >&2
  fi
  exit 1
}

CONFIG="${CONFIG:-src/projects/configs/StreamPETR_ttc_v3/stream_petr_vov_ttc_frozen_20e.py}"
CHECKPOINT="${CHECKPOINT:-${CKPT:-outputs/streampetr_ttc_frozen_20e/iter_140650.pth}}"
ANN_FILE="${ANN_FILE:-data/nuscenes/nuscenes2d_temporal_infos_val.pkl}"
NUSCENES_ROOT_RESOLVED="${COMPARE_DATA_ROOT:-${NUSCENES_ROOT:-$(pwd)/data/nuscenes}}"
NUSCENES_ROOT_RESOLVED="${NUSCENES_ROOT_RESOLVED%/}/"
if [[ ! -f "${NUSCENES_ROOT_RESOLVED}nuscenes2d_temporal_infos_train.pkl" ]] && [[ -f "${NUSCENES_ROOT_RESOLVED}nuscenes/nuscenes2d_temporal_infos_train.pkl" ]]; then
  NUSCENES_ROOT_RESOLVED="${NUSCENES_ROOT_RESOLVED}nuscenes/"
  echo "[eval] Using nested nuscenes/ for data-root: ${NUSCENES_ROOT_RESOLVED}"
fi
if [[ -n "${VALIDATE_NUSCENES_INFOS_EXPECT:-}" ]] && [[ "${SKIP_VALIDATE_NUSCENES_INFOS:-0}" != "1" ]]; then
  echo "[eval] Validating infos pkls expect=${VALIDATE_NUSCENES_INFOS_EXPECT} data_root=${NUSCENES_ROOT_RESOLVED}"
  python -u src/tools/validate_nuscenes_infos_split.py \
    --expect "${VALIDATE_NUSCENES_INFOS_EXPECT}" \
    --data-root "${NUSCENES_ROOT_RESOLVED}"
fi
SCENE_TOKEN="${SCENE_TOKEN:-fcbccedd61424f1b85dcbf8f897f9754}"

if [[ -z "${STREAMPETR_TTC_PKL:-}" ]] && [[ -f "$(pwd)/data/nuscenes/ttc_gt_labels_v1_0_mini.pkl" ]]; then
  export STREAMPETR_TTC_PKL="$(pwd)/data/nuscenes/ttc_gt_labels_v1_0_mini.pkl"
fi

COMPARE_VIDEO="${COMPARE_VIDEO:-1}"
VIDEO_PANELS="${VIDEO_PANELS:-gt,physics,mlp}"

echo "[eval] SCENE_TOKEN=${SCENE_TOKEN}  (SKIP_COMPARE_SCENE=${SKIP_COMPARE_SCENE:-0})  COMPARE_VIDEO=${COMPARE_VIDEO}  VIDEO_PANELS=${VIDEO_PANELS}"

cmd=(
  python -u src/tools/eval_ttc_mlp.py "${CONFIG}" "${CHECKPOINT}"
  --ann-file "${ANN_FILE}" --max-batches "${MAX_BATCHES:-80}" --gpu-id "${GPU_ID:-0}"
)
[[ -n "${NUSCENES_ROOT_RESOLVED}" ]] && cmd+=(--data-root "${NUSCENES_ROOT_RESOLVED}")
[[ -f ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth ]] && cmd+=(--pretrained-baseline ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth)

if [[ "${USE_SRUN:-0}" == "1" ]] && command -v srun &>/dev/null; then
  srun "${cmd[@]}"
else
  "${cmd[@]}"
fi

if [[ "${SKIP_TTC_BREAKDOWN:-0}" == "1" ]]; then
  echo "[eval] SKIP_TTC_BREAKDOWN=1"
else
  echo "[eval] eval_ttc_breakdown.py"
  BREAKDOWN_MAX_BATCHES="${BREAKDOWN_MAX_BATCHES:-50}"
  breakdown_cmd=(
    python -u src/tools/eval_ttc_breakdown.py "${CONFIG}" "${CHECKPOINT}"
    --ann-file "${ANN_FILE}"
    --max-batches "${BREAKDOWN_MAX_BATCHES}"
    --gpu-id "${GPU_ID:-0}"
  )
  [[ -n "${NUSCENES_ROOT_RESOLVED}" ]] && breakdown_cmd+=(--data-root "${NUSCENES_ROOT_RESOLVED}")
  [[ -n "${STREAMPETR_TTC_PKL:-}" ]] && [[ -f "${STREAMPETR_TTC_PKL}" ]] && breakdown_cmd+=(--ttc-pkl "${STREAMPETR_TTC_PKL}")
  if [[ "${USE_SRUN:-0}" == "1" ]] && command -v srun &>/dev/null; then
    srun "${breakdown_cmd[@]}"
  else
    "${breakdown_cmd[@]}"
  fi
fi

if [[ "${SKIP_COMPARE_SCENE:-0}" != "1" ]]; then
  if [[ -z "${PRETRAINED_BASELINE:-}" ]] && [[ -f ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth ]]; then
    PRETRAINED_BASELINE="ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth"
  fi
  if [[ -z "${PRETRAINED_BASELINE:-}" ]] || [[ ! -f "${PRETRAINED_BASELINE:-}" ]]; then
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
      echo "[eval] compare_ttc_scene x${#_compare_scenes[@]}"
      _csi=0
      for _st in "${_compare_scenes[@]}"; do
        [[ -z "${_st}" ]] && continue
        _st="${_st// /}"
        _csi=$((_csi + 1))
        _save="${_ckpt_dir}/compare_ttc_scene_${_st}"
        echo "[eval] compare scene ${_csi}/${#_compare_scenes[@]} token=${_st} save_dir=${_save}"
        compare_cmd=(
          python -u src/tools/compare_ttc_scene.py "${CONFIG}" "${CHECKPOINT}"
          --pretrained-baseline "${PRETRAINED_BASELINE:-}"
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
        if [[ "${USE_SRUN:-0}" == "1" ]] && command -v srun &>/dev/null; then
          srun "${compare_cmd[@]}"
        else
          "${compare_cmd[@]}"
        fi
      done

      if [[ "${SKIP_ABLATION_TABLE:-0}" != "1" ]]; then
        echo "[eval] aggregate_ttc_eval_ablation.py"
        python -u src/tools/aggregate_ttc_eval_ablation.py --work-dir "${_ckpt_dir}" || echo "[eval] WARN: aggregate_ttc_eval_ablation.py failed" >&2
      fi
    fi
fi
