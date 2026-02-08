Data-Driven Discovery of Thermodynamic Controls on South Asian Monsoon Precipitation
------------

Savannah L. Ferretti<sup>1</sup>, Tom Beucler<sup>2,3</sup>,  Sara Shamekh<sup>4</sup>, Michael S. Pritchard<sup>5/sup>, & Jane W. Baldwin<sup>1,6</sup>

<sup>1</sup>Department of Earth System Science, University of California Irvine, Irvine, CA, USA  
<sup>2</sup>Faculty of Geosciences and Environment, University of Lausanne, Lausanne, VD, CH  
<sup>3</sup>Expertise Center for Climate Extremes, University of Lausanne, Lausanne, VD, CH  
<sup>4</sup>Courant Institute for Mathematical Science, New York University, New York, NY, USA  
<sup>5</sup>NVIDIA Corporation, Santa Clara, CA, USA  
<sup>6</sup>Lamont-Doherty Earth Observatory, Palisades, NY, USA  

**Status:** This manuscript is currently in preparation. We welcome any comments, questions, or suggestions. Please email your feedback to Savannah Ferretti (savannah.ferretti@uci.edu).

**Abstract**: Insert abstract text here.

Project Organization
------------
```
├── LICENSE.md         <- License for code
│
├── README.md          <- Top-level information on this code base/manuscript
│
├── data/
│   ├── raw/          <- Original ERA5 and IMERG V06 data
│   ├── interim/      <- Intermediate data that has been transformed
│   ├── splits/       <- Training, validation, and test sets
│   ├── predictions/  <- Model predictions
│   ├── features/     <- Kernel-integrated features
│   └── weights/      <- Learned kernel weights
│
├── figs/              <- Generated figures/graphics 
│
├── models/            
│   ├── pod/           <- Saved POD models
│   ├── nn/            <- Saved NN models
│   └── sr/            <- Saved PySR models
│
├── notebooks/         <- Jupyter notebooks for data analysis and visualizations
│
├── scripts/
│   ├── data/
│   │   ├── classes/      <- Data processing classes
│   │   ├── download.py   <- Execution script for downloading raw data
│   │   ├── calculate.py  <- Execution script for calculating derived variables
│   │   └── split.py      <- Execution script for creating train/valid/test splits
│
└── environment.yml    <- File for reproducing the analysis environment
```

Acknowledgements
-------

The analysis for this work was performed on NERSC’s [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/). This research was supported by [LEAP NSF-STC](https://leap.columbia.edu/); the US DOE, including the [ASCR](https://www.energy.gov/science/ascr/advanced-scientific-computing-research) Program and the [BER](https://www.energy.gov/science/ber/biological-and-environmental-research) Program Office; [NVIDIA](https://www.nvidia.com/en-us/); NASA’s [NIP-ES](https://science.nasa.gov/earth-science/early-career-opportunities/#h-early-career-investigator-program-in-earth-science); and the Horizon Europe [AI4PEX Project](https://ai4pex.org/) through [SERI](https://www.sbfi.admin.ch/en). Additionally, we thank Fiaz Ahmed and Eric Wengrowski for their input during the early stages of this work.

--------
<p><small>This template is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
