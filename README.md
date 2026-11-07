Discovering Equations for the Thermodynamics of South Asian Monsoon Precipitation
------------

This repository contains the code used to reproduce the analyses in Ferretti et al. (in preparation for *Journal of Advances in Modeling Earth Systems*). The final version will be archived on Zenodo upon acceptance. For questions or feedback, contact Savannah Ferretti (savannahleeferretti@outlook.com).

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
conda activate monsoon-disocvery
```

Before running any code, edit `scripts/configs.json` to set valid local paths for `filepaths` (`raw`, `interim`, `splits`, `predictions`, `weights`, and `models`). All outputs will be written to these locations. Users should also update `metadata` (author and email for output files), and can optionally customize the spatial/temporal domain in `domain`, the year ranges in `splits`, the chosen predictors and target in `variables`, training hyperparameters and seeds in `training`, and the set of model variants in `models`. Model names passed via `--models` must match the `name` fields in the configuration.

Before running any code, edit `scripts/configs.json`:
- `filepaths`: set valid local paths for `raw`, `interim`, `splits`, `predictions`, `weights`, and `models`. All outputs are written to these locations (the defaults point to our NERSC directories).
- `metadata`: set the `author` and `email` written to the output files.


Now, download the ERA5 and IMERG V06 datasets into `data/raw/`, compute derived predictor and target variables stored in `data/interim/`, and generate the training, validation, and test splits written to `data/splits/`:
```
python -m scripts.data.download
python -m scripts.data.calculate
python -m scripts.data.split
```
Train and evaluate all models specified in the configuration file:
```
python -m scripts.models.train
python -m scripts.models.evaluate --split test # or --split valid, for tuning
```
To train and evaluate a single model:
```
python -m scripts.models.train --models baseline_local
python -m scripts.models.evaluate --models baseline_local --split test
```
Or train and evaluate multiple models at once:
```
python -m scripts.models.train --models baseline_local,baseline_nonlocal
python -m scripts.models.evaluate --models baseline_local,baseline_nonlocal --split test
```
To visualize your results, run the Jupyter Notebooks in `notebooks/` to replicate the figures used in the manuscript.

Project Organization
------------
```
├── LICENSE.md        <- License for code
├── README.md         <- Top-level information on this code base/manuscript
├── data/
│   ├── raw/          <- Original ERA5 data
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
