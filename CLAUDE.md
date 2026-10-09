# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project Overview

Research code for data-driven discovery of thermodynamic controls on South Asian Monsoon precipitation (Ferretti et al., in prep for JAMES). Runs on NERSC Perlmutter with ERA5 and IMERG V06 data.

## Environment

```bash
conda env create -f environment.yml && conda activate monsoon-discovery
```

Julia is required for PySR. On NERSC, Julia packages live at `/global/cfs/cdirs/m4334/sferrett/.julia`; SR jobs use that depot directly (not `$SCRATCH`, which can hang during scratch outages); the juliapkg project is copied to node-local `/tmp` because its file lock does not work on CFS.

## Running Scripts

All scripts run as modules from the repo root:

```bash
# Data pipeline (in order)
python -m scripts.data.download
python -m scripts.data.calculate
python -m scripts.data.split

# Training
python -m scripts.models.nn.train --runs all # or comma-separated nn_bl,nn_full

# Evaluation
python -m scripts.models.nn.evaluate --runs all --split test

# SR constant optimization
python -m scripts.models.sr.optimize --equations all --splits test
```

NERSC: `sbatch train_sr.sh [run_name]`, `sbatch optimize_sr.sh`.

## Before Committing

No test suite; `configs.json` points to NERSC CFS paths, so most scripts can't run end-to-end off Perlmutter. Minimum checks:

- `python -m py_compile <changed files>` — syntax
- `python -c 'import scripts.models.nn.train'` (etc.) — imports, circular imports, module-level failures. Also confirms `stats.json` resolves (read from `filepaths.splits`; at import time by `architectures.py`).
- New `configs.json` keys are actually read by consuming code; `python -c 'from scripts.utils import Config; Config()'` still parses.
- If data is available: run the affected script at smallest scale (`--runs <one_run>`, `--iterations 5`, `--subsetfrac 0.001`).

State exactly what was/wasn't verified — never describe untested code as working.

## Guardrails

- `data/{raw,interim,splits,predictions,features,weights}/`, `*.nc`, `*.h5`, `*.pkl`, `*.pth`, `*.npz`, and `*.sh` (except `train_sr.sh`/`optimize_sr.sh`) are gitignored — `git add` won't pick them up; don't add new `.sh` files without updating `.gitignore` too.
- `filepaths` in `configs.json` is NERSC-specific — update locally for testing, never commit personal path overrides.
- Scripts skip runs whose outputs already exist — don't delete existing checkpoints/predictions to force a rerun without asking first.

## Git Workflow

Claude Code works on a `claude` branch and cannot run experiments here (no NERSC access from this environment) — treat results as unverified until the user confirms them. Don't commit to or push `main` directly. After changes are committed to `claude`, the user merges `claude` → local `main` → pushes to remote `main`; only then is `main` up to date. If asked to check the "latest" state of something, confirm you're looking at `claude`, not an unmerged `main`.

## Configuration

`scripts/configs.json` holds all parameters; `scripts/utils.py:Config` exposes them as attributes. Key blocks: `filepaths` (NERSC CFS paths — update locally), `domain` (JJA 2000–2020, 5–25°N 60–90°E; hourly ERA5 raw files also include Sep 1 00:00 of each year so the last 3-hourly window is complete), `splits` (train 2000–2014, valid 2015–2017, test 2018–2020), `variables`, `experiments` (per-run configs for `nn`/`sr`). New run → add entry to `experiments.<type>.runs`.

## Architecture

**Data Pipeline:** raw ERA5/IMERG → thermodynamic variables (`rh`, `thetae`, `thetaestar`, `bl`, surface fluxes, `dsig`) → HDF5 splits (`{split}.h5`, physical units). Stats → `data/splits/stats.json`; inputs are standardized on the fly (no normalized files). Splits use `h5netcdf` engine. Concurrent `timewindow`-hour windows starting at T (00, 03, …, 21 UTC): state variables are the trapezoidal mean of hourly values T..T+3 (derived variables computed hourly first), fluxes the mean and `tp` the sum of ERA5 stamps T+1..T+3 (ERA5 stamps accumulations at the end of the hour), IMERG `pr` the mean of half-hourly stamps T..T+2:30. 736 windows per JJA season. Sigma interpolation holds values at the nearest pressure level outside 500–1000 hPa (no extrapolation).

**Precision:** float64 in memory, float32 on disk. Read and write every data file through `scripts/utils.py:load`/`save`.

**NN** (`scripts/models/nn/`): three `kind` variants — `baseline` (flattened profiles + local vars), `nonparametric` (free-form learned vertical kernel), `parametric` (Gaussian kernel, learnable mu/sigma). Shared 4-layer GELU backbone; output `zmin + ReLU(f(x))` (non-negative precip). Target: z-scored `log1p(tp)`. Kernel models save integration weights to `data/weights/`, reused by SR. Checkpoints: `{run}_{seed}.pth`. Logged to W&B.

**SR** (`scripts/models/sr/`): `train.py` runs PySR search (Julia backend) → equation tables (`.csv`). Kernel-integrated features (`weightsfrom`): NN kernels renormalized in float64 (sum(k·Δσ)=1), averaged over seeds, applied to physical profiles, then standardized. `residualfrom` adds a prior optimized SR equation as a predictor (complexity hard-coded to 2 in `train.py`). Per-run `complexityofvariables` overrides `sr.complexity.ofvariables`. `optimize.py` fits constants of hand-specified forms (`sr.optimizedeqs`) via L-BFGS-B multistart: one start from the run's own PySR results (the optimized constants of `initfrom`; else a structural match at `refcomplexity`, averaged over `seeds`; else constants copied by hand from the PySR table into `init`), plus random starts (`sr.nrestarts` total, uniform ±`sr.initscale`); constants rounded to `sr.constantsigfigs` significant figures → `optimized_equations.csv` registry; physical-space predictions are checked against standardized ones (warning if they differ beyond float32 rounding). `equations.py` is the only place equations are evaluated and physical constants derived. `--predict-only` flag on `optimize.py` skips optimization and predicts from existing constants.

**Predictions:** `data/predictions/{name}_{split}_predictions.nc`. NN runs → `seed` dim; optimized SR equations (written by `optimize.py`) → single prediction. Native mm units, post-denormalization.

## Code Style

**Python:** no comments; no spaces after commas (`np.sqrt(a,b)`); variables have no underscores (`ntime`, `fieldvars`), functions do (`load_split`, `calc_rh`); single quotes; `if __name__=='__main__'` in entry-point scripts only; `logging` not `print`; verify file writes by reopening; skip runs whose outputs already exist.

**Notebooks:** imports → ALL_CAPS config fields (no underscores, e.g. `SAVEDIR`, `TARGETVAR`) → helper functions → analysis/plotting. `notebooks/` is for analysis/viz, not the pipeline.
