# Manuscript Changes

Running list of edits/things still do to for the manuscript now that the new analysis is complete. We need to check each section for inconsistencies with the code, besides the requests specifed below. Note that the code will be available on Zenodo once the paper is published, so we don't need to give every detail on how we got a result (especially in the Methods and Supplemental Information).

Every results section needs two checks: (1) the text against its regenerated figure/table, and (2) the physical form of each equation in the text against the standardized form that was optimized (registry `models/sr/optimized_equations.csv` + `stats.json`, via `scripts/models/sr/equations.py`; `optimize.py` logs a WARNING if the two disagree).

## Status of the SR equations

| Model | Form (standardized) | Status |
|---|---|---|
| SR-BL | `cube(bl+c1)+c2` | Fixed (complexity 7, seeds 42/72/102). |
| SR-ATM | `c3*cube(max(rh,thetae-c4*thetaestar-c5))` | Fixed (complexity 17, seeds 42/102). |
| SR-SFC | `sr_atm_eq+shf*(c6-lf)+c7*lhf` | Chosen (complexity 14, seed 72; same structure in SR-ALL seeds 72 and 102). Old form without the SHF scale constant. |
| SR-ALL | `sr_atm_eq+(shf+c8*thetae)*cube(c9-lf)` | Chosen (complexity 15, seed 72; starts from `init`, no structural match). |
| SR-ALL-PC | — | Removed from `configs.json`; with c9 > 1 SR-ALL may already satisfy PC₂. Decide after the constraint results. |

SR-SFC and SR-ALL forms are updated in `configs.json` and `equations.py`. If SR-ALL-PC returns, add it to both (and its `complexity`, used for Figure 1).

## Notebooks to update before regenerating figures/tables

`equations.ipynb` (Table S2) now reads the registry and `equations.py` directly and lists only optimized equations. The notebooks below still retype the equations with the old constant names (c6–c13) instead of importing `scripts/models/sr/equations.py`, so they must be updated to the final forms (ideally by importing `equations.py`):

- `srsfc.ipynb` (Figure 5) and `srall.ipynb` (Figure 6) now use `equations.py` for constants and predictions; Figure 6 shows SR-ALL (no SR-ALL-PC).
- `diurnal.ipynb` and `constraints.ipynb` now loop over whatever equations are in the registry (`constraints` uses `load_features` and finite-difference gradients through `equations.evaluate`).

## Key Points, Abstract, PLS
- Key points need to be "verified" that they still apply given the new results.
- Abstract and PLS both need to be written.

## Introduction
- Need to add citations to possible reviewers where relevant. Possible reviewers are: Rajat Masiwal, Akshay Deoras, Hao Xu, Dion Hafner, Antonios Mamalakis.
- Otherwise checked against the code; no changes.

## Methods

- **2.3 Experimental Design:** SR-ALL-PC paragraph: check once SR-ALL-PC is defined.
- **Text S3** (kept here with the SR methods):
  - Seeds that found each structure (Constant Optimization paragraph): SR-BL 42, 72, and 102 (complexity 7); SR-ATM 42 and 102 (complexity 17); SR-SFC 72 (complexity 14); SR-ALL 72 (complexity 15).
  - Search settings changed for the rerun of all four searches: "its predictions enter with weight 1, since that equation has already been optimized and adds no degrees of freedom" → the existing equation's output has complexity 2, like the other time-varying predictors; add that denominators are limited to complexity 1 (a constant or LF), alongside the existing limits on exponents and nested compositions.
  - Unchanged (checked against `configs.json`): operators and complexities, max size 20 (10 for SR-BL), depth 10, 20 × 150 populations, 200 iterations, parsimony 0.0025, constant-optimization weight 0.25, ~8.6M combined samples, ~215,000 at 2.5%, 50 initializations, [−5, 5].

## Results

For each item: update numbers and descriptions from the regenerated figure/table, do the physical-form check, and flag any qualitative change from the old version.

- **Section 3.1 (model hierarchy) — Figure 1 (`pareto.ipynb`):** all R² and MSE values; complexity markers (SR complexity + SR-ATM's for SR-SFC/SR-ALL/SR-ALL-PC); dashed line now connects only the plotted models.
- **SR-BL — Figure 2 (`srbl.ipynb`):** equation and physical constants (λ, B_c, β); physical-form check.
- **Kernels — Figure 3 (`weights.ipynb`):** kernel descriptions (new nonparametric RH peaks near the surface and the column top).
- **SR-ATM — Figure 4 (`sratm.ipynb`):** equation and physical constants (λ, κ, γ, θ_c); confirm λ, κ, γ > 0 (Section 3.5 and Text S4 rely on this); physical-form check.
- **SR-SFC — Figure 5 (`srsfc.ipynb`):** new equation and its interpretation (old form had LHF; new may not); physical constants; physical-form check.
- **Section 3.4 (physical constraints) — Table 1 (`constraints.ipynb`):** all satisfaction rates; ERA5 reference choices (land for PC₁) still justified.
- **Section 3.7 (SR-ALL, SR-ALL-PC) — Figure 6 (`srall.ipynb`):** new forms and physical constants; derivatives with respect to each feature and which constraints hold; motivation and construction of SR-ALL-PC; physical-form check for both.
- **Diurnal results — diurnal table (`diurnal.ipynb`):** all values; flag changes in the diurnal cycle (windows are now concurrent, so phases may shift).

## Conclusion
- Check every quantitative statement and model description against the final results, especially SR-SFC, SR-ALL, and SR-ALL-PC.

## Supplemental Information

- **Table S1, Figure S1 (`weights.ipynb`):** confirm inter-seed kernel variability is still negligible before keeping that statement in both captions.
- **Table S2 (`equations.ipynb`):** physical constants of every model from `equations.calc_physical_constants` (registry + `stats.json`), with enough digits to reproduce predictions; physical-form check for every model.
- **Text S4, Table S3, Figure S2 (`constraints.ipynb`):** update all satisfaction rates (99.2, 62.6, 87.2, 97.2, 81.9, 93.3, 95.2%), the 54 violating ocean hypercubes, and "roughly 40%" dry fraction; re-verify that λ, κ, γ are positive (SR-ATM, SR-SFC) and which discovered equations satisfy PC₁ for all inputs; the closing sentence about SR-ALL/SR-ALL-PC derivatives must match the new forms.
- **New SI section — lag scan:** B_L paired with precipitation at different time offsets (`tests/timing/results/lagscan.*`); move the script and results out of `tests/timing` before deleting it.