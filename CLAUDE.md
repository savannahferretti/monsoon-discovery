# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project Overview

Research code for data-driven discovery of thermodynamic controls on South Asian Monsoon precipitation (Ferretti et al., in prep for JAMES). Runs on NERSC Perlmutter with ERA5 and IMERG V06 data.

## Environment

```bash
conda env create -f environment.yml && conda activate monsoon-discovery
```

Julia is required for PySR. On NERSC, Julia packages live at `/global/cfs/cdirs/m4334/sferrett/.julia`, synced to `$SCRATCH/.julia` before SR jobs.

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

NERSC: `sbatch train_sr.sh [run_name]`, `sbatch optimize_sr.sh`.

## Before Committing

No test suite; `configs.json` points to NERSC CFS paths, so most scripts can't run end-to-end off Perlmutter. Minimum checks:

- `python -m py_compile <changed files>` — syntax
- `python -c 'import scripts.models.nn.train'` (etc.) — imports, circular imports, module-level failures. Also confirms `data/splits/stats.json` resolves (read at import time by `architectures.py` and other scripts).
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

`scripts/configs.json` holds all parameters; `scripts/utils.py:Config` exposes them as attributes. Key blocks: `filepaths` (NERSC CFS paths — update locally), `domain` (JJA 2000–2020, 5–25°N 60–90°E), `splits` (train 2000–2014, valid 2015–2017, test 2018–2020), `variables`, `experiments` (per-run configs for `pod`/`nn`/`sr`). New run → add entry to `experiments.<type>.runs`.

## Architecture

**Data Pipeline:** raw ERA5/IMERG → thermodynamic variables (`rh`, `thetae`, `thetaestar`, `bl`, surface fluxes, `dsig`) → HDF5 splits, raw + normalized. Stats → `data/splits/stats.json`. Splits use `h5netcdf` engine.

**POD** (`scripts/models/pod/`): ramp model `alpha * max(0, bl - xcrit)`, fit to binned training data. Saved as `.npz`.

**NN** (`scripts/models/nn/`): three `kind` variants — `baseline` (flattened profiles + local vars), `nonparametric` (free-form learned vertical kernel), `parametric` (Gaussian kernel, learnable mu/sigma). Shared 4-layer GELU backbone; output `zmin + ReLU(f(x))` (non-negative precip). Target: z-scored `log1p(tp)`. Kernel models save integration weights to `data/weights/`, reused by SR. Checkpoints: `{run}_{seed}.pth`. Logged to W&B.

**SR** (`scripts/models/sr/`): `train.py` runs PySR search (Julia backend) → Pareto frontiers (`.pkl`) + equation tables (`.csv`). `optimize.py` fits constants of hand-specified forms (`sr.optimizedeqs` in configs) via L-BFGS-B multistart → `optimized_equations.pkl` registry. Can use NN kernel-integrated features (`weightsfrom`) or target residuals from a prior SR equation (`baselinefrom`).

**Predictions:** `data/predictions/{run}_{split}_predictions.nc`. NN → `seed` dim; SR → `seed` + `complexity` dims (full Pareto frontier). Native mm units, post-denormalization.

## Code Style

**Python:** no comments; no spaces after commas (`np.sqrt(a,b)`); variables have no underscores (`ntime`, `fieldvars`), functions do (`load_split`, `calc_rh`); single quotes; `if __name__=='__main__'` in entry-point scripts only; `logging` not `print`; verify file writes by reopening; skip runs whose outputs already exist.

**Notebooks:** imports → ALL_CAPS config fields (no underscores, e.g. `SAVEDIR`, `TARGETVAR`) → helper functions → analysis/plotting. `notebooks/` is for analysis/viz, not the pipeline.
