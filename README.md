Data-Driven Equation Discovery of Thermodynamic Relationships with South Asian Monsoon Precipitation
------------

This repository contains the code used to reproduce the analyses in Ferretti et al. (in preparation for *Journal of Advances in Modeling Earth Systems*). The final version will be archived on Zenodo upon acceptance. For questions or feedback, contact Savannah Ferretti (savannah.ferretti@uci.edu).

**Authors & Affiliations:**  
Savannah Ferretti<sup>1</sup>, Tom Beucler<sup>2</sup>, Mike Pritchard<sup>3</sup>, & Jane Baldwin<sup>1,4</sup>  
<sup>1</sup>University of California Irvine, Irvine, CA, United States  
<sup>2</sup>University of Lausanne, Lausanne, VD, Switzerland  
<sup>3</sup>NVIDIA Corporation, Santa Clara, CA, United States  
<sup>4</sup>Lamont-Doherty Earth Observatory, Palisades, NY, United States  

**Abstract**: South Asian monsoon rainfall depends on the thermodynamic structure of the lower troposphere, but how its ingredients combine to shape rainfall remains unclear, partly because this dependence is nonlocal in the vertical. We use symbolic distillation, which combines neural networks and symbolic regression, to discover interpretable rainfall equations from 21 years of June--August reanalysis data. Neural networks first learn the vertical thermodynamic structure most relevant to rainfall, condensing each variable into a single interpretable feature, and symbolic regression then derives closed-form equations from these features and the surface state. These equations outperform a widely used buoyancy-based diagnostic and recover much of the skill of the neural networks while using about a thousand times fewer parameters. They reveal that moisture and instability act as competing rather than additive controls, with the heaviest rainfall occurring when the lower free troposphere is humid. Over the ocean, sensible heating is associated with additional rainfall mainly when near-surface air is warm and moist, improving predicted rainfall and its diurnal cycle. Although the learned features were optimized only for rainfall prediction, the equations largely respond to each feature as expected from convective physics, consistent with the reanalysis. These responses reduce the dependence of monsoon rainfall on the entire vertical column to a few simple, testable constraints that could guide more physically consistent precipitation models.

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

The analysis for this work was performed on NERSC’s [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/). This research was supported by [LEAP NSF-STC](https://leap.columbia.edu/); the US DOE [ASCR](https://www.energy.gov/science/ascr/advanced-scientific-computing-research) Program; [NVIDIA](https://www.nvidia.com/en-us/); NASA’s [NIP-ES](https://science.nasa.gov/earth-science/early-career-opportunities/#h-early-career-investigator-program-in-earth-science); and the Horizon Europe [AI4PEX Project](https://ai4pex.org/) through [SERI](https://www.sbfi.admin.ch/en). Additionally, we thank Jerry Lin for his input in the early stages of this work.
