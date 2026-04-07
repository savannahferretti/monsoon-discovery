Data-Driven Discovery of Thermodynamic Controls on South Asian Monsoon Precipitation
------------

This repository contains the code used to reproduce the analyses in Ferretti et al. (in preparation for *Journal of Advances in Modeling Earth Systems*). The final version will be archived on Zenodo upon acceptance. For questions or feedback, contact Savannah Ferretti (savannah.ferretti@uci.edu).

**Authors & Affiliations:**  
Savannah L. Ferretti<sup>1</sup>, Tom Beucler<sup>2</sup>, Michael S. Pritchard<sup>3</sup>, & Jane W. Baldwin<sup>1,4</sup>  
<sup>1</sup>University of California Irvine, Irvine, CA, United States  
<sup>2</sup>University of Lausanne, Lausanne, Switzerland  
<sup>3</sup>NVIDIA Corporation, Santa Clara, CA, United States  
<sup>4</sup>Lamont-Doherty Earth Observatory, Palisades, NY, United States  

**Abstract**: Insert abstract text here.

Project Organization
------------
```
├── LICENSE.md         <- License for code
├── README.md          <- Top-level information on this code base/manuscript
├── data/
│   ├── raw/          <- Original ERA5 and IMERG V06 data
│   ├── interim/      <- Intermediate processed data
│   ├── splits/       <- Training, validation, and test sets
│   ├── predictions/  <- Model predictions
│   └── weights/      <- Learned kernel weights
├── figs/             <- Manuscript figures
├── models/            
│   ├── pod/           <- Saved model checkpoints for POD models
│   ├── nn/            <- Saved model checkpoints for NNs
│   └── sr/            <- Saved model checkpoints for PySR models
├── notebooks/         <- Jupyter notebooks for data analysis and visualization
├── scripts/
│   ├── data/         <- Data processing scripts
│   ├── models/       <- Model building, training, and inferencing scripts
│   └── utils.py      <- Configuration and utility functions
└── environment.yml   <- File for reproducing the analysis environment
```

Acknowledgements
-------

The analysis for this work was performed on NERSC’s [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/). This research was supported by [LEAP NSF-STC](https://leap.columbia.edu/); the US DOE, via the [ASCR](https://www.energy.gov/science/ascr/advanced-scientific-computing-research) Program; [NVIDIA](https://www.nvidia.com/en-us/); NASA’s [NIP-ES](https://science.nasa.gov/earth-science/early-career-opportunities/#h-early-career-investigator-program-in-earth-science); and the Horizon Europe [AI4PEX Project](https://ai4pex.org/) through [SERI](https://www.sbfi.admin.ch/en). Additionally, we thank Jerry Lin for their input in the early stages of this work.
