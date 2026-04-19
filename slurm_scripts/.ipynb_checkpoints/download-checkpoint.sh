#!/bin/bash
# Extract nuScenes v1.0-trainval archives on Slurm.
#
# Layout (typical on your cluster):
#   Archives live in:  .../nuscenes/train_og/*.tgz
#   Extract into:      .../nuscenes/train/   (samples/, sweeps/, maps/, v1.0-trainval/)
#
# Usage:
#   sbatch sbatch_extract_nuscenes_trainval.sh
#   TGZ_SRC=/work/eb408/nuscenes/train_og NUSC_ROOT=/work/eb408/nuscenes/train sbatch sbatch_extract_nuscenes_trainval.sh
#
# What this script expects in TGZ_SRC:
#   - v1.0-trainval_meta.tgz
#   - v1.0-trainval01_blobs.tgz ... v1.0-trainval10_blobs.tgz (or equivalent blob parts)
#
# What it creates under NUSC_ROOT:
#   - samples/
#   - sweeps/
#   - maps/
#   - v1.0-trainval/
#
# Notes:
#   - Extraction uses `tar ... -C NUSC_ROOT` so archives can stay in train_og while output goes to train.
#   - Safe to rerun; tar overwrites same paths under NUSC_ROOT.
#   - Forced overwrite: TAR_OVERWRITE=1.

#SBATCH --job-name=nusc_extract_trainval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs

# Where the .tgz files sit (read-only source tree).
TGZ_SRC="${TGZ_SRC:-/work/eb408/nuscenes/train_og}"
TGZ_SRC="${TGZ_SRC%/}"
# Where nuScenes runtime tree should appear (what StreamPETR uses as data root).
NUSC_ROOT="${NUSC_ROOT:-/work/eb408/nuscenes/train}"
NUSC_ROOT="${NUSC_ROOT%/}"

mkdir -p "${NUSC_ROOT}"

echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "TGZ_SRC (archives): ${TGZ_SRC}"
echo "NUSC_ROOT (extract): ${NUSC_ROOT}"
echo "========================================="

META="${TGZ_SRC}/v1.0-trainval_meta.tgz"
if [[ ! -f "${META}" ]]; then
  echo "FATAL: missing ${META}" >&2
  exit 1
fi

shopt -s nullglob
blob_parts=("${TGZ_SRC}"/v1.0-trainval*_blobs.tgz)
if [[ ${#blob_parts[@]} -eq 0 ]]; then
  echo "FATAL: no v1.0-trainval*_blobs.tgz in ${TGZ_SRC}" >&2
  exit 1
fi

tar_flags=(-xzf)
if [[ "${TAR_OVERWRITE:-0}" == "1" ]]; then
  tar_flags=(--overwrite -xzf)
fi

echo "[1/3] Extracting metadata archive into ${NUSC_ROOT} ..."
tar "${tar_flags[@]}" "${META}" -C "${NUSC_ROOT}"

echo "[2/3] Extracting blob parts (${#blob_parts[@]} files) into ${NUSC_ROOT} ..."
for tgz in "${blob_parts[@]}"; do
  echo "  - $(basename "${tgz}")"
  tar "${tar_flags[@]}" "${tgz}" -C "${NUSC_ROOT}"
done

echo "[3/3] Verifying key output folders under ${NUSC_ROOT} ..."
missing=0
for d in samples sweeps maps v1.0-trainval; do
  if [[ -d "${NUSC_ROOT}/${d}" ]]; then
    echo "  OK: ${d}/"
  else
    echo "  MISSING: ${d}/" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "FATAL: extraction incomplete. Check archive set, paths, and free disk space." >&2
  exit 1
fi

echo ""
echo "Top-level sizes (sanity check):"
du -sh "${NUSC_ROOT}/"{samples,sweeps,maps,v1.0-trainval} 2>/dev/null || true

echo ""
echo "Done: $(date)"
