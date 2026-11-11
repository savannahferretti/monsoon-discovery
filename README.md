Discovering Equations for the Thermodynamics of South Asian Monsoon Precipitation
------------

This repository contains the code used to reproduce the analyses in [Ferretti et al.](https://arxiv.org/abs/2611.14401), a preprint submitted to the *Journal of Advances in Modeling Earth Systems*. The version of the code used to produce the preprint results is [archived on Zenodo](https://doi.org/10.5281/zenodo.24094998). For questions or feedback, contact Savannah Ferretti (savannah.ferretti@asu.edu).

**Authors & Affiliations:**  
Savannah Ferretti<sup>1</sup>, Jane Baldwin<sup>1</sup>, Mike Pritchard<sup>2</sup>, & Tom Beucler<sup>3</sup>  
<sup>1</sup>University of California Irvine, Irvine, CA, United States  
<sup>2</sup>NVIDIA Corporation, Santa Clara, CA, United States  
<sup>3</sup>University of Lausanne, Lausanne, VD, Switzerland  

**Abstract**: South Asian monsoon precipitation depends on the thermodynamic structure of the lower troposphere, but how its ingredients combine to shape precipitation remains unclear, partly because this dependence is nonlocal in the vertical. We use symbolic distillation, which combines neural networks and symbolic regression, to discover interpretable precipitation equations from 21 years of June--August reanalysis data. Neural networks first learn the vertical thermodynamic structure most relevant to precipitation, condensing each variable into a single interpretable feature not tied to a particular vertical grid. Then, symbolic regression derives closed-form equations from these features and the surface state. Our best equation explains 1.5 times as much precipitation variance as a widely used buoyancy-based diagnostic and recovers about 85\% of neural network skill while using about a thousand times fewer parameters. Unlike the buoyancy diagnostic, which adds moisture and instability, the equations treat them as competing controls, with the heaviest precipitation occurring when the lower free troposphere is humid. Over the ocean, sensible heating is associated with additional precipitation mainly when near-surface air is warm and moist, improving predicted precipitation and its diurnal cycle. Although the learned features were optimized only for precipitation prediction, the equations largely respond to each feature as expected from convective physics, consistent with the reanalysis. These responses reduce the dependence of monsoon precipitation on the entire vertical column to a few simple, testable constraints that could guide data-driven precipitation models to make them more physically consistent.

How to Use this Repository
------------

All commands should be run from the repository root after cloning:
```
git clone https://github.com/savannahferretti/monsoon-discovery.git
cd monsoon-discovery
```
Create the Conda environment and activate it:
```
conda env create -f environment.yml
conda activate monsoon-discovery
```

Before running any code, edit `scripts/configs.json` to set valid local paths for `filepaths` (`raw`, `interim`, `splits`, `predictions`, `features`, `weights`, and `models`). All outputs will be written to these locations. Users should also update `metadata` (author and email for output files), and can optionally customize the spatial/temporal domain in `domain`, the year ranges in `splits`, and the neural network and symbolic regression settings in `experiments`. This includes adding, removing, or renaming runs and changing their input variables. The run and equation names used below are the ones from our configuration; if you change them, pass your own names via `--runs` or `--equations`, matching the keys in `experiments.nn.runs`, `experiments.sr.runs`, or `experiments.sr.optimizedeqs`.

Now, download the ERA5 and IMERG V06 datasets into `raw`, compute derived predictor and target variables stored in `interim`, and generate the training, validation, and test splits written to `splits`:
```
python -m scripts.data.download
python -m scripts.data.calculate
python -m scripts.data.split
```
First, train and evaluate all neural networks. Evaluating the kernel neural networks saves the learned kernel weights used by the symbolic regression runs:
```
python -m scripts.models.nn.train --runs all
python -m scripts.models.nn.evaluate --runs all --split test # or --split valid, for tuning
```
Next, run the symbolic regression searches for SR-BL and SR-ATM:
```
python -m scripts.models.sr.train --runs sr_bl,sr_atm
```
Then choose the equation form to optimize for each run from the resulting equation tables (`<run>_<seed>_equations.csv` in `models`). PySR's search is stochastic, so your equations may differ from ours. The forms already in `experiments.sr.optimizedeqs` are the ones from our runs; replace them with forms chosen from your own results. For each equation, set `runfrom` to the symbolic regression run it came from and `refcomplexity` to its PySR complexity. Then optimize the constants of both equations:
```
python -m scripts.models.sr.optimize --equations sr_bl_eq,sr_atm_eq
```
Then, run the symbolic regression searches for SR-SFC and SR-ALL, which use the optimized SR-ATM equation as a predictor (set by `residualfrom`):
```
python -m scripts.models.sr.train --runs sr_sfc,sr_all
```
As before, choose the SR-SFC and SR-ALL forms from your results and add them to `optimizedeqs`. SR-ALL-PC is a hand-specified variant of the SR-ALL form, so it has no search of its own: add it with `runfrom` set to `sr_all` and `initfrom` set to `sr_all_eq`. Then optimize all three:
```
python -m scripts.models.sr.optimize --equations sr_sfc_eq,sr_all_eq,sr_all_pc_eq
```
To visualize your results, run the Jupyter Notebooks in `notebooks/` to replicate the figures used in the manuscript.

Project Organization
------------
```
├── LICENSE.md        <- License for code
├── README.md         <- Top-level information on this code base/manuscript
├── data/
│   ├── raw/          <- Original data
│   ├── interim/      <- Intermediate processed data
│   ├── splits/       <- Training, validation, and test sets
│   ├── predictions/  <- Model predictions
│   └── weights/      <- Learned kernel weights
├── figs/             <- Manuscript figures
├── models/            
│   ├── nn/           <- Saved model checkpoints for NNs
│   └── sr/           <- Saved model checkpoints for PySR models
├── notebooks/        <- Jupyter notebooks for data analysis and visualization
├── scripts/
│   ├── data/         <- Data processing scripts
│   ├── models/       <- Model building, training, and inferencing scripts
│   └── utils.py      <- Configuration and utility functions
└── environment.yml   <- File for reproducing the analysis environment
```

Acknowledgements
-------

The analysis for this work was performed on NERSC’s [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/). This research was supported by [LEAP NSF-STC](https://leap.columbia.edu/); the US DOE [ASCR](https://www.energy.gov/science/ascr/advanced-scientific-computing-research) Program; [NVIDIA](https://www.nvidia.com/en-us/); NASA’s [NIP-ES](https://science.nasa.gov/earth-science/early-career-opportunities/#h-early-career-investigator-program-in-earth-science); and the Horizon Europe [AI4PEX Project](https://ai4pex.org/) through [SERI](https://www.sbfi.admin.ch/en). Additionally, we thank Ariane Bietighoefer, Spencer Hill, Jerry Lin, Sara Shamekh, and Frederick Iat-Hin Tam for valuable scientific discussions that helped inform this work.
