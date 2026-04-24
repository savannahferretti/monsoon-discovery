#!/bin/bash
#SBATCH --account=m4334
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --time=06:00:00
#SBATCH --job-name=train_sr
#SBATCH --output=logs/%x_%j.log

set -eo pipefail

export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1
export JULIA_DEPOT_PATH=/global/cfs/cdirs/m4334/sferrett/.julia
export PYTHON_JULIAPKG_PROJECT=/global/cfs/cdirs/m4334/sferrett/.julia/environments/pyjuliapkg
export UCX_ERROR_SIGNALS=""

module load python conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate monsoon-discovery

RUN=${1:-all}
shift || true
scontrol update JobId=${SLURM_JOB_ID} Name=${RUN}
echo "Training model: ${RUN}"

python -m scripts.models.sr.train --runs ${RUN} "$@"
python -m scripts.models.sr.evaluate --runs ${RUN} --split valid