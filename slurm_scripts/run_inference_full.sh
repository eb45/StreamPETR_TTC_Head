#!/bin/bash
# Slurm: nuScenes bbox eval (mAP / NDS) — full **val** split, not mini.
#
# Prerequisites: full v1.0-trainval blobs + nuscenes2d_temporal_infos_{train,val}.pkl (v1.0, not mini).
# Submit from StreamPETR repo root:  cd /path/to/StreamPETR && sbatch run_streampetr_bbox_eval_slurm.sh
# Override: STREAMPETR_REPO=...  NUSCENES_ROOT=... CONFIG=... CHECKPOINT=... GPUS=4
#
# Layout: code under src/ (src/tools/test.py, src/projects/configs/...).

#SBATCH --job-name=spetr_bbox_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs

_root="${STREAMPETR_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
cd "$(cd "${_root}" && pwd)" || { echo "FATAL: cannot cd to ${_root}" >&2; exit 1; }
[[ -f src/tools/test.py ]] || {
  echo "FATAL: not at StreamPETR root (missing src/tools/test.py). cd to repo then sbatch." >&2
  exit 1
}

for _modinit in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /hpc/group/naderilab/navid/miniconda3/etc/profile.d/modules.sh; do
  [[ -f "${_modinit}" ]] && source "${_modinit}" && break
done
if type module &>/dev/null; then
  module load cuda/11.8 2>/dev/null || echo "[bbox-eval] note: module load cuda/11.8 failed or unavailable; continuing."
else
  echo "[bbox-eval] note: no 'module' command — using conda env + driver only."
fi

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env
if [[ "${EVAL_LD_NO_USR64:-0}" != "1" ]]; then
  export LD_LIBRARY_PATH="/usr/lib64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ "${STREAMPETR_KEEP_CUDA_HOME:-0}" != "1" ]]; then
  [[ -n "${CUDA_HOME:-}" ]] && echo "[bbox-eval] unsetting CUDA_HOME for conda PyTorch (set STREAMPETR_KEEP_CUDA_HOME=1 to keep)"
  unset CUDA_HOME
else
  [[ -n "${CUDA_HOME:-}" ]] && [[ -d "${CUDA_HOME}/lib64" ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi

_PYP="$(pwd):$(pwd)/src:$(pwd)/src/tools"
[[ -d "$(pwd)/mmdetection3d/mmdet3d" ]] && _PYP="$(pwd)/mmdetection3d:${_PYP}"
export PYTHONPATH="${_PYP}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY

echo "[bbox-eval] host=$(hostname) job=${SLURM_JOB_ID:-?} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && export CUDA_VISIBLE_DEVICES=0
if ! nvidia-smi -L; then
  echo "FATAL: nvidia-smi failed. Run inside a GPU allocation." >&2
  exit 1
fi
python -u -c "import os; os.environ.setdefault('CUDA_MODULE_LOADING','LAZY'); import torch; print('[bbox-eval] torch cuda:', torch.cuda.is_available(), torch.cuda.device_count())" || {
  echo "FATAL: PyTorch cannot see CUDA." >&2
  exit 1
}
python -u -c "import mmdet3d; print('[bbox-eval] mmdet3d OK')" || {
  echo "FATAL: cannot import mmdet3d. See docs/setup.md or: pip install -e \$(pwd)/mmdetection3d" >&2
  exit 1
}

CONFIG="${CONFIG:-src/projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py}"
CHECKPOINT="${CHECKPOINT:-ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth}"
GPUS="${GPUS:-1}"

if [[ -z "${NUSCENES_ROOT:-}" ]]; then
  if [[ -d "$(pwd)/data_full/nuscenes/v1.0-trainval" ]]; then
    NUSCENES_ROOT="$(pwd)/data_full/nuscenes"
  else
    NUSCENES_ROOT="/work/eb408/nuscenes/train"
  fi
fi
NUSCENES_ROOT="${NUSCENES_ROOT%/}/"

echo "[bbox-eval] CONFIG=${CONFIG}"
echo "[bbox-eval] CHECKPOINT=${CHECKPOINT}"
echo "[bbox-eval] NUSCENES_ROOT=${NUSCENES_ROOT}"
echo "[bbox-eval] GPUS=${GPUS}"

_DATA_LINK="$(pwd)/data/nuscenes"
if [[ "${SKIP_DATA_NUSCENES_SYMLINK:-0}" != "1" ]]; then
  if [[ -e "${_DATA_LINK}" ]] && [[ ! -L "${_DATA_LINK}" ]]; then
    echo "[bbox-eval] ERROR: ${_DATA_LINK} exists and is a regular directory (not a symlink)." >&2
    echo "         Rename it (e.g. mv data/nuscenes data/nuscenes_mini) so we can ln -s to NUSCENES_ROOT." >&2
    exit 1
  fi
  ln -sfn "$(cd "${NUSCENES_ROOT%/}" && pwd)" "${_DATA_LINK}"
  echo "[bbox-eval] Symlink data/nuscenes -> $(readlink -f "${_DATA_LINK}")"
fi

if [[ "${SKIP_VALIDATE_NUSCENES_INFOS:-0}" != "1" ]]; then
  echo "[bbox-eval] Checking train/val infos are v1.0-trainval (not mini) ..."
  python -u src/tools/validate_nuscenes_infos_split.py --expect v1.0-trainval --data-root "${NUSCENES_ROOT}"
fi

VAL_PKL="${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl"
python -u -c "
import mmcv, sys
p = '${VAL_PKL}'
d = mmcv.load(p)
n = len(d['infos'])
print(f'[bbox-eval] val infos sample count: {n}')
if n < 500:
    print('[bbox-eval] ERROR: val split has <500 samples — this is mini or a bad pkl. Full val is ~6000.', file=sys.stderr)
    sys.exit(1)
"

_CFGOPTS=(
  "data_root=${NUSCENES_ROOT}"
  "data.test.data_root=${NUSCENES_ROOT}"
  "data.test.ann_file=${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl"
  "data.val.data_root=${NUSCENES_ROOT}"
  "data.val.ann_file=${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl"
)

if [[ "${GPUS}" -le 1 ]]; then
  python -u src/tools/test.py "${CONFIG}" "${CHECKPOINT}" \
    --launcher none \
    --eval bbox \
    --cfg-options "${_CFGOPTS[@]}"
else
  bash src/tools/dist_test.sh "${CONFIG}" "${CHECKPOINT}" "${GPUS}" \
    --eval bbox \
    --cfg-options "${_CFGOPTS[@]}"
fi

echo "[bbox-eval] Done."
