# Timing test

Self-contained test of how predictor timing affects skill, kernels, and discovered equations. Reads the existing raw files, interim static fields (`lf`, `dsig`), and the current registry/predictions without changing them. Writes only to `tests/timing/{data,results,logs}`. Delete the folder to remove everything.

## Windows

All three variants use the same target: ERA5 precipitation accumulated over [T, T+3 h], where T = 00, 03, ..., 21 UTC, thresholded at 10⁻⁴ mm. Hours are relative to T. ERA5 hourly accumulations and mean rates stamped *t* cover (t−1 h, t] (`accumlabel: "end"` in `configs.json`; confirm with `check_convention.py`).

| Variant | State (RH, θe, θe*, B_L) | Fluxes (SHF, LHF) |
|---|---|---|
| `lead` | instantaneous at T | mean over [T, T+3] (stamps T+1..T+3) |
| `concurrent` | trapezoidal mean of hourly values over [T, T+3] | mean over [T, T+3] |
| `causal` | trapezoidal mean over [T−3, T] | mean over [T−3, T] (stamps T−2..T) |

Derived variables are computed hourly, then averaged. The first and last window of each season are dropped in every variant (734 instead of 736 per season) because they need hours outside JJA.

`lead` is what the manuscript describes. The current pipeline differs: `floor('3h')` groups hourly stamps T, T+1, T+2, so its fluxes and precipitation cover [T−1, T+2] (see the main reply).

SR-SFC and SR-ALL take SR-ATM as input. For each variant they use the manuscript SR-ATM *form*, with constants refit on that variant's data. All five manuscript forms are refit per variant. The PySR searches run the same as in the main pipeline.

## Run order (from repo root on Perlmutter, `conda activate monsoon-discovery`)

```bash
# 0. Conventions (login node, needs internet; ~minutes)
python tests/timing/check_convention.py 2>&1 | tee tests/timing/logs/convention.log

# 1. Validate the reimplementation against the existing interim files (CPU node)
python tests/timing/calculate.py --check 2018 2>&1 | tee tests/timing/logs/check2018.log

# 2. Current-setup numbers (only reads existing files)
python tests/timing/summarize.py --variants current

# 3. Build variant data (CPU node, all 21 years)
python tests/timing/calculate.py --variants all
python tests/timing/split.py --variants all

# 4. NN-BL and NN-GAUSS (GPU node; W&B project Chapter-3-Timing, group timing_<variant>)
python tests/timing/nn_train.py --variants all
python tests/timing/nn_evaluate.py --variants all

# 5. SR searches, stage 1
for v in lead concurrent causal; do for r in sr_bl sr_atm; do sbatch tests/timing/sr_train.sbatch $v $r; done; done
# 6. Refit SR-BL and SR-ATM (needed by stage 2)
for v in lead concurrent causal; do sbatch tests/timing/sr_optimize.sbatch $v sr_bl_eq,sr_atm_eq; done
# 7. SR searches, stage 2
for v in lead concurrent causal; do for r in sr_sfc sr_all; do sbatch tests/timing/sr_train.sbatch $v $r; done; done
# 8. Refit the remaining forms
for v in lead concurrent causal; do sbatch tests/timing/sr_optimize.sbatch $v sr_sfc_eq,sr_all_eq,sr_all_pc_eq; done

# 9. Summary → tests/timing/results/summary.md, summary.json, <variant>_pareto.txt
python tests/timing/summarize.py
```

Each step skips outputs that already exist. Every script takes `--variants lead,causal` etc. The training scripts also take `--seeds 42` for a cheaper first pass; SR features then average only the NN-GAUSS seeds that exist. Quick-test outputs (`--iterations`, `--subsetfrac`) are written to the same paths, so delete them before the full run.
