#!/bin/bash
# Slurm: nuScenes bbox eval (mAP / NDS) — full **val** split (thousands of samples), not mini (~81).
#
# Prerequisites (or you will get mini-like metrics / wrong NDS):
#   1) Extract ALL v1.0-trainval*_blobs.tgz + meta into NUSCENES_ROOT so samples/ sweeps/ are full.
#   2) nuscenes2d_temporal_infos_{train,val}.pkl built with create_data_nusc.py --version v1.0 (NOT v1.0-mini).
#   3) Point NUSCENES_ROOT at that root (e.g. symlink repo/data_full/nuscenes -> your full tree).
#
# Uses tools/test.py (1 GPU) or tools/dist_test.sh (multi-GPU).
#
# Submit (from repo root):
#   sbatch run_streampetr_bbox_eval_slurm.sh
#
# Overrides:
#   NUSCENES_ROOT=/path/to/full/nuscenes CONFIG=... CHECKPOINT=... GPUS=4 sbatch run_streampetr_bbox_eval_slurm.sh
# Example (absolute path like your screenshot):
#   NUSCENES_ROOT=/data_full/nuscenes sbatch run_streampetr_bbox_eval_slurm.sh
#
# Skip infos version check (e.g. downloaded pkls without metadata):
#   SKIP_VALIDATE_NUSCENES_INFOS=1 sbatch run_streampetr_bbox_eval_slurm.sh
#
# Infos pkls embed paths like ./data/nuscenes/samples/... — this script symlinks ./data/nuscenes -> NUSCENES_ROOT.
# If ./data/nuscenes is a real directory (e.g. mini), rename it first or set SKIP_DATA_NUSCENES_SYMLINK=1 and symlink manually.

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

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-${_REPO}}"

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env

export PYTHONPATH="$(pwd):$(pwd)/tools:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py}"
CHECKPOINT="${CHECKPOINT:-ckpts/stream_petr_vov_flash_800_bs2_seq_24e.pth}"
GPUS="${GPUS:-1}"

# Default data root: explicit NUSCENES_ROOT, else ./data_full/nuscenes if present, else legacy /work path.
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

# Image paths inside temporal_infos *.pkl are stored as ./data/nuscenes/samples/... — not rewritten by --cfg-options.
# Point repo ./data/nuscenes at the real full dataset tree.
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
  python -u tools/validate_nuscenes_infos_split.py --expect v1.0-trainval --data-root "${NUSCENES_ROOT}"
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

# IMPORTANT: Config sets ann_file=data_root+'...' at parse time to ./data/nuscenes/... strings.
# Merging only data_root= does NOT rewrite those paths — test still used mini ./data/nuscenes/.
# Override data.test (and data.val) explicitly:
_CFGOPTS=(
  "data_root=${NUSCENES_ROOT}"
  "data.test.data_root=${NUSCENES_ROOT}"
  "data.test.ann_file=${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl"
  "data.val.data_root=${NUSCENES_ROOT}"
  "data.val.ann_file=${NUSCENES_ROOT}nuscenes2d_temporal_infos_val.pkl"
)

# For GPUS>1, request matching GPUs in #SBATCH --gres (e.g. gpu:4) and set GPUS=4.
if [[ "${GPUS}" -le 1 ]]; then
  python -u tools/test.py "${CONFIG}" "${CHECKPOINT}" \
    --launcher none \
    --eval bbox \
    --cfg-options "${_CFGOPTS[@]}"
else
  bash tools/dist_test.sh "${CONFIG}" "${CHECKPOINT}" "${GPUS}" \
    --eval bbox \
    --cfg-options "${_CFGOPTS[@]}"
fi

echo "[bbox-eval] Done."
