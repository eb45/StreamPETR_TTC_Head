#!/bin/bash
# Same queue as slurm_eval_ttc_mlp.sh (common), but ask Slurm for a specific GPU type.
# If sbatch rejects --gres=gpu:2080:1, run:  sinfo -p common -o "%G"
# and match your site’s GRES name (e.g. rtx2080) or use --constraint if your admin uses that.
#
# Submit:  sbatch slurm_eval_ttc_mlp_2080.sh
#
#SBATCH --job-name=eval_ttc_mlp_2080
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=common
#SBATCH --gres=gpu:2080:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail
# Slurm copies only this file to /var/spool/slurmd/... — companion script is not there.
# SLURM_SUBMIT_DIR is the directory you ran sbatch from (repo root); use it for the real path.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
exec bash "${SUBMIT_DIR}/slurm_eval_ttc_mlp.sh"
