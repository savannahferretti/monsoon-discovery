#!/bin/bash
#SBATCH --account=m4334
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --time=01:30:00
#SBATCH --job-name=train_sr
#SBATCH --output=logs/%x_%j.log

set -eo pipefail

JULIA_DEPOT_CFS=/global/cfs/cdirs/m4334/sferrett/.julia
JULIA_DEPOT_SCRATCH=$SCRATCH/.julia
rsync -a --update $JULIA_DEPOT_CFS/ $JULIA_DEPOT_SCRATCH/

export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_DEPOT_PATH=$JULIA_DEPOT_SCRATCH
export PYTHON_JULIAPKG_PROJECT=$JULIA_DEPOT_SCRATCH/environments/pyjuliapkg
export UCX_ERROR_SIGNALS=""

module load python conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate monsoon-discovery

RUN=${1:-all}
shift || true
scontrol update JobId=${SLURM_JOB_ID} Name=${RUN}
echo "Training model: ${RUN}"

python -m scripts.models.sr.train --runs ${RUN} --timeout 5400 "$@"
python -m scripts.models.sr.evaluate --runs ${RUN} --split valid