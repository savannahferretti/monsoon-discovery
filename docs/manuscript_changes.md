# Manuscript changes for the concurrent-timing rerun

Scope: only changes needed for the manuscript to be correct at its level of detail; implementation details (precision, quadrature, regridding method) are left to the released code.

Running list of edits the rerun requires. At the end, send the updated figures/tables and the relevant sections; each item below gets checked against them.

## Already drafted (confirm applied)

- **Section 2.1 (Data), sigma paragraph:** sigma levels outside 1,000–500 hPa take the nearest pressure level instead of extrapolating; "To avoid" → "To minimize the influence of" below-ground levels.
- **Section 2.1 (Data), resolution/timing paragraph:** concurrent 3-hour windows (predictors computed hourly and averaged over each window, precipitation accumulated over the same window); the old "state precedes precipitation" paragraph replaced by one sentence saying the relationships are concurrent and not causal. This covers the old timing description at lines 91 and 93.
- **Section 2.2.4 (Symbolic Regression):** "predictions of an existing equation" → "output of an existing equation" (applied).

## Text S3 (Symbolic Regression Search and Optimization)

- Rounding: "two decimal places" → "four significant figures".
- SR-ALL-PC initialization: "from the SR-ALL constants found by PySR" → "from the optimized SR-ALL constants".
- Add after the variable-complexity sentence: "The exception is the SR-ALL search, in which the kernel-integrated features contribute 1 unit each, so that reusing them in the correction is not penalized relative to the surface variables."
- SR-SFC and SR-ALL now search only for additive corrections to SR-ATM: replace "In searches that build on an existing equation, its predictions enter with weight 1, since that equation has already been optimized and adds no degrees of freedom." with a sentence saying the existing equation's output is added to every candidate, so complexity counts only the correction.
- Dry fraction checked: 19.3% of combined training+validation samples (text's "approximately 19%" stands).
- Seeds that found each structure (Constant Optimization paragraph): SR-BL 42, 72, and 102 (complexity 7); SR-ATM 42 and 102 (complexity 17); SR-SFC and SR-ALL still to come.
- Unchanged (checked against `configs.json`): operators and complexities, max size 20 (10 for SR-BL), depth 10, 20 × 150 populations, 200 iterations, parsimony 0.0025, constant-optimization weight 0.25, ~8.6M combined samples, ~215,000 at 2.5%, 50 initializations, [−5, 5].

## Text S2 (Gaussian Kernel Parameterization), Table S1, Figure S1

- Done: R² (0.512 / 0.509), Table S1 values, θe and θe* descriptions checked against the new Figure S1.
- RH sentence: nonparametric RH now peaks near the surface and near the top of the column (σ ≈ 0.5–0.6), not the lower free troposphere; replacement sentence provided.
- Confirm inter-seed kernel variability is still negligible before keeping that statement in the Table S1 and Figure S1 captions.

## Text S4 (Physical Constraint Evaluation), Table S3, Figure S2

- Update all satisfaction rates (99.2, 62.6, 87.2, 97.2, 81.9, 93.3, 95.2%), the 54 violating ocean hypercubes, and "roughly 40%" dry fraction.
- Re-verify that λ, κ, γ are positive (SR-ATM, SR-SFC) and that each discovered equation satisfies PC₁ for all inputs.

## Section 3 / SR-ALL / SR-ALL-PC

- SR-ALL form is now `sr_atm_eq + c14·(θe + c15·SHF)·cube(c16 − LF)` (free scale c14 on the correction); SR-ALL-PC adds `c14·cube(1 − c16)·θe`. Update the equations, the SR-ALL-PC definition, and the derivatives in Section 3.7.
- The final optimized forms depend on the new SR searches and may differ; update all equations and physical constants once they are fixed.
- **Table S2:** physical constants from `equations.calc_physical_constants` (registry + stats.json) (4 significant figures; enough digits to reproduce predictions).
- All skill numbers, Pareto figures, diurnal and constraint results: update and flag any qualitative change.

## SI additions

- Lag scan (B_L paired with precipitation at different time offsets; `tests/timing/results/lagscan.*`) as a new SI section/figure.

## Open / optional

- If IMERG is described anywhere, its window is the half-hourly values from T to T+2:30.
