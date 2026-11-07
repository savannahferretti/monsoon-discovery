Discovering Equations for the Thermodynamics of South Asian Monsoon Precipitation
------------

This repository contains the code used to reproduce the analyses in Ferretti et al. (in preparation for *Journal of Advances in Modeling Earth Systems*). The final version will be archived on Zenodo upon acceptance. For questions or feedback, contact Savannah Ferretti (savannahleeferretti@outlook.com).

**Authors & Affiliations:**  
Savannah Ferretti<sup>1</sup>, Jane Baldwin<sup>1</sup>, Mike Pritchard<sup>2</sup>, & Tom Beucler<sup>3</sup>  
<sup>1</sup>University of California Irvine, Irvine, CA, United States  
<sup>2</sup>NVIDIA Corporation, Santa Clara, CA, United States  
<sup>3</sup>University of Lausanne, Lausanne, VD, Switzerland  

**Abstract**: South Asian monsoon precipitation depends on the thermodynamic structure of the lower troposphere, but how its ingredients combine to shape precipitation remains unclear, partly because this dependence is nonlocal in the vertical. We use symbolic distillation, which combines neural networks and symbolic regression, to discover interpretable precipitation equations from 21 years of June--August reanalysis data. Neural networks first learn the vertical thermodynamic structure most relevant to precipitation, condensing each variable into a single interpretable feature not tied to a particular vertical grid. Then, symbolic regression derives closed-form equations from these features and the surface state. Our best equation explains 1.5 times as much precipitation variance as a widely used buoyancy-based diagnostic and recovers about 85\% of neural network skill while using about a thousand times fewer parameters. Unlike the buoyancy diagnostic, which adds moisture and instability, the equations treat them as competing controls, with the heaviest precipitation occurring when the lower free troposphere is humid. Over the ocean, sensible heating is associated with additional precipitation mainly when near-surface air is warm and moist, improving predicted precipitation and its diurnal cycle. Although the learned features were optimized only for precipitation prediction, the equations largely respond to each feature as expected from convective physics, consistent with the reanalysis. These responses reduce the dependence of monsoon precipitation on the entire vertical column to a few simple, testable constraints that could guide data-driven precipitation models to make them more physically consistent.

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
