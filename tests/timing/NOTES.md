# Notes for moving the main pipeline to concurrent timing

Decisions to carry over from the timing test. Copy this file somewhere permanent before deleting `tests/timing/`.

## Windows
- Concurrent: every variable covers [T, T+3 h]. State variables are the trapezoidal mean of hourly snapshots T..T+3 (weights ½, 1, 1, ½), computed hourly and then averaged. Fluxes are the mean, and precipitation the sum, of hourly values stamped T+1..T+3, because ERA5 stamps cover the hour ending at the stamp. Precipitation is then set to 0 below 10⁻⁴ mm.
- Keep the June 1 00:00 window. In the test it is dropped only because the windows are shared with the `causal` variant, which needs May 31.
- Keep the Aug 31 21:00 window. It needs the Sep 1 00:00 hour (the snapshot and the 23:00–24:00 accumulations).
- IMERG already covers [T, T+3]: half-hourly stamps 00:00..02:30 mark the start of each interval. ERA5 then matches it with no shift.

## Download
- `scripts/data/classes/downloader.py` subsets by `months` only, so each season ends at Aug 31 23:00. Change the time selection so a from-scratch run also downloads Sep 1 00:00 of each year for every hourly ERA5 variable.
- For the existing raw files, add the missing hour ad hoc (e.g. in a one-off notebook): download only Sep 1 00:00 for each year and variable, append it to the existing raw files, and verify by reopening. Do not re-download everything.
- `scripts/data/calculate.py`: replace `resample` (`first` / `floor('3h')`) with the concurrent windowing from `tests/timing/calculate.py`, keeping every window that has its required hours.

## Precision
- One rule: float64 in memory, float32 on disk. Convert at the read and write helpers only (`tests/timing/timingutils.py`). Compute stats in float64. Standardize on the fly instead of storing `norm_*.h5`.
- One module holds the SR equations and physical-constant conversions (`tests/timing/equations.py`). Notebooks import it instead of retyping the formulas.
- Kernel weights are renormalized in float64 so sum(k·dσ) = 1 exactly before SR use.
- Round optimized constants to 4 significant figures, not 2 decimals. Update Text S3.
- Table S2: report the physical constants with enough digits to reproduce the predictions, or list the standardized constants plus training mean/std.

## Other open issues
- `interpolate_to_sigma` extrapolates below the lowest pressure level when surface pressure > 1000 hPa: max RH 152%, max θe* 536 K in interim data. Count the affected samples and decide on clipping or constant extrapolation.
- Manuscript lines 91 and 93 describe the timing incorrectly for the current data: precipitation and fluxes cover [T−1, T+2], not the 3 h after the state sample. Rewrite them for whichever timing is final.
- NN training uses TF32 and float16 mixed precision. Disclose in the methods, or disable (`useamp=False`, remove the `allow_tf32` lines).
