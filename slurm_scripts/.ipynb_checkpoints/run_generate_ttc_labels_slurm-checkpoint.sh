#!/bin/bash
# CPU/RAM job: src/tools/generate_ttc_labels.py → ttc_gt_labels_*.pkl
# sbatch run_generate_ttc_labels_slurm.sh   (override NUSCENES_ROOT, VERSION, OUT)

#SBATCH --job-name=ttc_labels
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=common
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO=""
for _cand in "${SLURM_SUBMIT_DIR:-}" "$(dirname "${SLURM_SUBMIT_DIR:-.}")" "${_script_dir}" "$(dirname "${_script_dir}")"; do
  [[ -z "${_cand}" || "${_cand}" == "." ]] && continue
  _cand="$(cd "${_cand}" 2>/dev/null && pwd)" || continue
  if [[ -f "${_cand}/src/tools/generate_ttc_labels.py" ]]; then
    _REPO="${_cand}"
    break
  fi
done
if [[ -z "${_REPO}" ]]; then
  _d="${_script_dir}"
  while [[ "${_d}" != "/" ]]; do
    if [[ -f "${_d}/src/tools/generate_ttc_labels.py" ]]; then
      _REPO="${_d}"
      break
    fi
    _d="$(dirname "${_d}")"
  done
fi
[[ -n "${_REPO}" ]] || { echo "FATAL: cannot find repo root (need src/tools/generate_ttc_labels.py)." >&2; exit 1; }
cd "${_REPO}"

source /hpc/group/naderilab/navid/miniconda3/bin/activate
conda activate ~/eb408/CS372/streampetr_env

_PYP="$(pwd):$(pwd)/src:$(pwd)/src/tools"
[[ -d "$(pwd)/mmdetection3d/mmdet3d" ]] && _PYP="$(pwd)/mmdetection3d:${_PYP}"
export PYTHONPATH="${_PYP}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

NUSCENES_ROOT="${NUSCENES_ROOT:-${_REPO}/data_full/nuscenes}"
VERSION="${VERSION:-v1.0-trainval}"
OUT="${OUT:-${NUSCENES_ROOT%/}/ttc_gt_labels_v1_0_trainval.pkl}"

echo "[ttc_labels] REPO=${_REPO}"
echo "[ttc_labels] NUSCENES_ROOT=${NUSCENES_ROOT}"
echo "[ttc_labels] VERSION=${VERSION}"
echo "[ttc_labels] OUT=${OUT}"

python -u src/tools/generate_ttc_labels.py \
  --data-root "${NUSCENES_ROOT}" \
  --version "${VERSION}" \
  --out "${OUT}" \
  --progress-every 2000

echo "[ttc_labels] Done. Check: ls -lh ${OUT}"
