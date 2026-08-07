# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code for data-driven discovery of thermodynamic controls on South Asian Monsoon precipitation (Ferretti et al., in prep for JAMES). Runs on NERSC Perlmutter with ERA5 and IMERG V06 data.

## Environment

```bash
conda env create -f environment.yml && conda activate monsoon-discovery
```

Julia is required for PySR. On NERSC, Julia packages live at `/global/cfs/cdirs/m4334/sferrett/.julia` and are synced to `$SCRATCH/.julia` before SR jobs.

## Running Scripts

All scripts run as modules from the repo root:

```bash
# Data pipeline (in order)
python -m scripts.data.download
python -m scripts.data.calculate
python -m scripts.data.split

# Training
python -m scripts.models.pod.train
python -m scripts.models.nn.train --runs all # or comma-separated nn_bl,nn_full

# Evaluation
python -m scripts.models.pod.evaluate --split test
python -m scripts.models.nn.evaluate --runs all --split test
python -m scripts.models.sr.evaluate --runs all --split test

# SR constant optimization
python -m scripts.models.sr.optimize --equations all --splits test
```

On NERSC use the SLURM wrappers: `sbatch train_sr.sh [run_name]`, `sbatch optimize_sr.sh`.

## Verifying Changes Before Pushing

There is no test suite. The paths in `configs.json` point to NERSC CFS directories, so most scripts cannot run end-to-end outside Perlmutter. So, before committing, at minimum:
- `python -m py_compile <changed files>` — catches syntax errors
- `python -c 'import scripts.models.nn.train'` (etc.) — catches bad imports, circular imports, and module-level failures. Note that `architectures.py` and several scripts read `data/splits/stats.json` at import time, so an import check also confirms that file resolves.
- Confirm any new `configs.json` keys are actually read by the code that consumes them, and that existing runs still parse (`python -c 'from scripts.utils import Config; Config()'`)

Where data is available, run the affected script with the smallest possible workload (`--runs <one_run>`, `--iterations 5`, `--subsetfrac 0.001`) rather than a full job. State plainly what was and wasn't verified. If a change could only be checked for syntax and imports, say that — do not describe untested code as working.

## Configuration

`scripts/configs.json` holds all parameters; `scripts/utils.py:Config` exposes them as attributes. Key blocks: `filepaths` (NERSC CFS paths — update locally), `domain` (JJA 2000–2020, 5–25°N 60–90°E), `splits` (train 2000–2014, valid 2015–2017, test 2018–2020), `variables`, and `experiments` (per-run configs for `pod`, `nn`, `sr`). To add a run, add an entry to `experiments.<type>.runs`.

## Architecture

**Data Pipeline:** raw ERA5/IMERG → thermodynamic variables (`rh`, `thetae`, `thetaestar`, `bl`, surface fluxes, `dsig`) → HDF5 splits with raw and normalized versions. Stats saved to `data/splits/stats.json`. All split files use `h5netcdf` engine.

**POD** (`scripts/models/pod/`): ramp model `alpha * max(0, bl - xcrit)`, fit to binned training data. Saved as `.npz`.

**NN** (`scripts/models/nn/`): three variants keyed by `kind` in the run config — `baseline` (flattened profiles + local vars), `nonparametric` (free-form learned vertical kernel), `parametric` (Gaussian kernel with learnable mu/sigma). All share the same 4-layer GELU backbone and output `zmin + ReLU(f(x))` (enforces non-negative precip). Target is z-scored `log1p(tp)`. Kernel models save integration weights to `data/weights/` — reused by SR. Checkpoints: `{run}_{seed}.pth`. Training logged to W&B.

**SR** (`scripts/models/sr/`): two-stage — `train.py` runs PySR search (Julia backend), saving Pareto frontiers (`.pkl`) and equation tables (`.csv`); `optimize.py` fits constants of hand-specified forms from `sr.optimizedeqs` in configs via L-BFGS-B multistart, writing an `optimized_equations.pkl` registry. SR runs can use NN kernel-integrated features (`weightsfrom`) or target residuals from a prior SR equation (`baselinefrom`).

**Predictions:** Saved to `data/predictions/{run}_{split}_predictions.nc`. NN output has a `seed` dimension; SR output has `seed` and `complexity` dims (full Pareto frontier). All in native mm units post-denormalization.

## Code Style

**Python:** No comments; no spaces after commas (`np.sqrt(a,b)`); variables have no underscores (`ntime`, `fieldvars`); functions do (`load_split`, `calc_rh`); single quotes; `if __name__=='__main__'` guards in entry-point scripts only; logging via `logging` module, not `print`; file writes verified by reopening; scripts skip runs where outputs already exist.

**Notebooks:** Imports → ALL_CAPS config fields (no underscores, e.g. `SAVEDIR`, `TARGETVAR`) → helper functions → analysis/plotting. The `notebooks/` directory is for analysis and visualization and is not part of the pipeline.
