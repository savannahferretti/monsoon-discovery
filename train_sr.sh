#!/bin/bash
#SBATCH --account=m4334
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=52
#SBATCH --time=00:40:00
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

python -m scripts.models.sr.train --runs ${RUN} "$@"

scancel $(squeue -u sferrett -h -o "%i") 2>/dev/null
find /global/homes/s/sferrett/ -name "lock.pid" -delete 2>/dev/null
salloc --account=m4334 --constraint=cpu --qos=interactive --nodes=1 --cpus-per-task=52 --time=04:00:00
cd /global/cfs/cdirs/m4334/sferrett/monsoon-discovery
rsync -a --update /global/cfs/cdirs/m4334/sferrett/.julia/ $SCRATCH/.julia/
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1
export JULIA_DEPOT_PATH=$SCRATCH/.julia
export PYTHON_JULIAPKG_PROJECT=$SCRATCH/.julia/environments/pyjuliapkg
export UCX_ERROR_SIGNALS=""
module load python conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate monsoon-discovery
python -m scripts.models.sr.train --runs sr_all --procs 50