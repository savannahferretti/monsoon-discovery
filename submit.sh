#!/bin/bash
#SBATCH --account=m4334
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --job-name=${RUNNAME}
#SBATCH --output=logs/%x_%j.log

export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1
export JULIA_DEPOT_PATH=/pscratch/sd/s/sferrett/.julia
export PYTHON_JULIAPKG_PROJECT=/pscratch/sd/s/sferrett/.julia/environments/pyjuliapkg

module load python conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate monsoon-discovery

python -m scripts.models.sr.train \
    --runs ${RUNNAME:-all} \
    --procs 127 \
    --timeout 19800