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

## Precision

- Files on disk are float32. Every file is read through `timingutils.load_dataset`, which converts it to float64, and written through `timingutils.save_dataset`, which converts it back to float32 and checks the result.
- All arithmetic in between is float64: hourly windowing, statistics, standardization, kernel integration, constant fitting, and conversion back to mm.
- Two exceptions: the NNs train on float32 tensors (converted once in `nn_train.to_tensors`; mixed precision as in the main pipeline), and PySR searches in float32. Only PySR's equation structure is used; constants are refit in float64.
- The NN-GAUSS kernels are renormalized in float64 so that sum(k·dσ) = 1 exactly. Profiles are integrated in physical units, then standardized, so standardized and physical-space equations see the same inputs.
- `equations.py` is the only place equations are evaluated and physical constants are derived. `sr_optimize.py` checks physical-space predictions against standardized-space and saved predictions, and writes the standardized constants, physical constants, and differences to `results/<variant>_<equation>_<split>_constants.json`.
- Constants are rounded to `constantsigfigs` (4) significant figures. The manuscript used 2 decimals; set this to match if needed.
- The normalized `norm_*.h5` files are not written. Inputs are standardized on the fly from `{split}.h5` and `stats.json`.

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

## SR-ALL with cheaper kernel features (`sr_all_k1`)

Same inputs as `sr_all` (SR-ATM plus RH, θe, θe*, LF, SHF, LHF), but RH, θe, and θe* cost 1 complexity unit instead of 2, so PySR is more inclined to use them outside SR-ATM's max. Surface fluxes still cost 2. Set in `configs.json` under `sr.extraruns`. Needs `sr_atm_eq` optimized first:

```bash
sbatch tests/timing/sr_train.sbatch concurrent sr_all_k1
```

Its complexities are not on the same scale as the other runs, so compare its equations by form and loss, not by position on the Pareto frontier.

`sr_all_k1_eq` (SR-ALL-K1) is the form this search found (concurrent, seed 72, complexity 16): `sr_atm_eq + c14·(thetae + c15·shf)·cube(c16 − lf)`. It is the manuscript SR-ALL with a free scale c14 on the correction, so it contains SR-ALL as c14 = 1. `sr_optimize.py` fits it from seed 72's constants, the variant's and the manuscript's SR-ALL constants, and random starts. Physical constants: λ_θe = s_y·c14/s_θe, λ_SHF = s_y·c14·c15/s_SHF, LF_c = c16.

## Lag scan (B_L only)

`lagscan.py` holds the precipitation window fixed and pairs it with B_L at different times: hourly snapshots from T−3 to T+3 h and 3-hour means centred at T−1.5, T−0.5, T+0.5, and T+1.5 h. It does this for both rain windows: `concurrent` [T, T+3] and `current` [T−1, T+2]. For each pairing it fits the SR-BL form on 2000–2017 and reports test R² (all, land, ocean), plus a model-free reference (mean rain in 40 B_L quantile bins). It reads only the raw files and writes `results/lagscan.{csv,md,png}`. Run it on a CPU node (about 30–60 min):

```bash
python tests/timing/lagscan.py 2>&1 | tee tests/timing/logs/lagscan.log
```

Sanity checks: the `current` window with `snapshot T+0` should give about SR-BL's current test R² (0.293), and the `concurrent` window with `mean T+0..T+3` about the concurrent SR-BL (0.275).

For the Fig. 1-style comparison (test R² bars and the Pareto frontier for `current` and each variant), open `tests/timing/pareto.ipynb` from inside `tests/timing/`. It saves `results/pareto.jpg`.

Each step skips outputs that already exist. Every script takes `--variants lead,causal` etc. The training scripts also take `--seeds 42` for a cheaper first pass; SR features then average only the NN-GAUSS seeds that exist. Quick-test outputs (`--iterations`, `--subsetfrac`) are written to the same paths, so delete them before the full run.
