# Manuscript Changes

Running list of edits/things still do to for the manuscript now that the new analysis is complete. We need to check each section for inconsistencies with the code, besdies the requests specifed below. Note that the code will be available on Zenodo once the paper is published, so we don't need, especially in the Methods/SI, give very specific details.

## Key Points, Abstract, PLS
- Key points need to be "verified" that they still apply given the new results.
- Abstract and PLS both need to be written.

## Introduction
- Need to add citations to possible reviewers where relevant. Possible reviewers are: Rajat Masiwal, Akshay Deoras, Hao Xu, Dion Hafner, Antonios Mamalakis.

## Methods


- Seeds that found each structure (Constant Optimization paragraph): SR-BL 42, 72, and 102 (complexity 7); SR-ATM 42 and 102 (complexity 17); SR-SFC and SR-ALL still to come.
- Unchanged (checked against `configs.json`): operators and complexities, max size 20 (10 for SR-BL), depth 10, 20 × 150 populations, 200 iterations, parsimony 0.0025, constant-optimization weight 0.25, ~8.6M combined samples, ~215,000 at 2.5%, 50 initializations, [−5, 5].

## Results


## Conclusion


## SI


## Text S4 (Physical Constraint Evaluation), Table S3, Figure S2

- Update all satisfaction rates (99.2, 62.6, 87.2, 97.2, 81.9, 93.3, 95.2%), the 54 violating ocean hypercubes, and "roughly 40%" dry fraction.
- Re-verify that λ, κ, γ are positive (SR-ATM, SR-SFC) and that each discovered equation satisfies PC₁ for all inputs.

## Section 3 / SR-ALL / SR-ALL-PC

- SR-ALL form is now `sr_atm_eq + c14·(θe + c15·SHF)·cube(c16 − LF)` (free scale c14 on the correction); SR-ALL-PC adds `c14·cube(1 − c16)·θe`. Update the equations, the SR-ALL-PC definition, and the derivatives in Section 3.7.
- The final optimized forms depend on the new SR searches and may differ; update all equations and physical constants once they are fixed.
- **Table S2:** physical constants from `equations.calc_physical_constants` (registry + stats.json) (4 significant figures; enough digits to reproduce predictions).
- All skill numbers, Pareto figures, diurnal and constraint results: update and flag any qualitative change.