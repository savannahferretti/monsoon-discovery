# IMERG-space notebooks

Copies of `notebooks/*.ipynb` with IMERG V06 (`pr`, mm/hr) as the truth instead of ERA5 (`tp`, mm). Self-contained: delete this folder to remove it.

**Bias correction** (`biascorrect.py`): empirical quantile mapping ERA5 → IMERG fit on paired train+valid samples (`FITSPLITS`, `NQUANTILES=200`), over the full distributions including zeros. ERA5 values at or below the quantile matching IMERG's dry fraction map to 0, so the corrected wet fraction matches IMERG; larger values follow the quantile transfer function. NaN stays NaN. Every model prediction (and ERA5 itself, where shown) is mapped before comparison to IMERG.

**Per-notebook changes**
- `pareto`: R²/MSE vs IMERG on bias-corrected predictions (per seed, full SR frontier), with bias-corrected ERA5 as a reference bar; MSE x-limits set from the data.
- `diurnal`: IMERG is the reference row; bias-corrected ERA5 added as a row.
- `weights`: validation R² vs IMERG; kernel figures unchanged.
- `srbl`, `sratm`: binned IMERG replaces binned ERA5; SR curves/surfaces bias-corrected.
- `srsfc`, `srall`: ΔMSE map, diurnal cycles and the ocean flux panels use IMERG and bias-corrected SR. The analytic "correction-made" surface is replaced by the binned mean of the bias-corrected correction (the mapping is nonlinear, so the closed form no longer applies); the dashed line is its zero contour.
- `constraints`: hypercube tests run on IMERG. Model rows are unchanged because a monotone mapping preserves derivative signs.
- `extremes`: same analysis, using the shared helper (so it also uses the zero-inclusive mapping).
- `equations`: no precipitation data involved; only the config path differs.

Max-difference checks against saved predictions still run in ERA5 space (before mapping). Figures save to `notebooks/imerg/figs/`.
