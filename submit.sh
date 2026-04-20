#!/bin/bash
#SBATCH --account=m4334
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --job-name=pysr_${RUNNAME}
#SBATCH --output=logs/pysr_%x_%j.out
#SBATCH --error=logs/pysr_%x_%j.err

export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1
export JULIA_DEPOT_PATH=/pscratch/sd/s/sferrett/.julia
export PYTHON_JULIAPKG_PROJECT=/pscratch/sd/s/sferrett/.julia/environments/pyjuliapkg

mkdir -p logs

module load conda
conda activate monsoon-discovery

cd /global/homes/s/sferrett/monsoon-discovery

python -m scripts.models.sr.train \
    --runs ${RUNNAME:-all} \
    --procs 127 \
    --timeout 19800
