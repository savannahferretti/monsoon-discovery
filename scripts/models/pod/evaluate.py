#!/usr/bin/env python

import os
import logging
import warnings
import argparse
import numpy as np
import xarray as xr
from scripts.utils import Config
from scripts.models.pod.model import RampPOD

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def load(splitname,splitsdir,inputvar):
    '''
    Purpose: Load the regular (non-normalized) evaluation data split.
    Args:
    - splitname (str): 'valid' | 'test'
    - splitsdir (str): directory containing split files
    - inputvar (str): input variable name
    Returns:
    - xr.DataArray: input DataArray for evaluation
    '''
    if splitname not in ('valid','test'):
        raise ValueError('Splitname must be `valid` or `test`.')
    filepath = os.path.join(splitsdir,f'{splitname}.h5')
    evalds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar]]
    x = evalds[inputvar].load()
    return x

def fetch(runname,modeldir):
    '''
    Purpose: Load a trained RampPOD from saved .npz file.
    Args:
    - runname (str): model run name
    - modeldir (str): directory containing model files
    Returns:
    - RampPOD: loaded RampPOD instance with fitted parameters
    '''
    filepath = os.path.join(modeldir,f'{runname}.npz')
    with np.load(filepath) as data:
        model = RampPOD(alpha=float(data['alpha']),xcrit=float(data['xcrit']))
    return model

def predict(model,x,targetvar):
    '''
    Purpose: Run the forward pass and return precipitation predictions as an xr.DataArray.
    Args:
    - model (RampPOD): trained RampPOD instance
    - x (xr.DataArray): input DataArray with dims (lat, lon, time)
    - targetvar (str): target variable name ('pr' | 'tp')
    Returns:
    - xr.DataArray: DataArray of predicted precipitation with same shape as x
    '''
    ypredflat = model.forward(x)
    ypred = xr.DataArray(ypredflat.reshape(x.shape),dims=x.dims,coords=x.coords,name=targetvar)
    ypred.attrs = dict(long_name='POD-predicted total precipitation',units='mm')
    return ypred

def save(ypred,runname,splitname,predsdir):
    '''
    Purpose: Save predicted precipitation to a NetCDF file, then verify by reopening.
    Args:
    - ypred (xr.DataArray): DataArray of predicted precipitation
    - runname (str): model run name
    - splitname (str): evaluated split label
    - predsdir (str): output directory
    Returns:
    - bool: True if writing and verification succeed, otherwise False
    '''
    os.makedirs(predsdir,exist_ok=True)
    filename = f'{runname}_{splitname}_predictions.nc'
    filepath = os.path.join(predsdir,filename)
    logger.info(f'      Attempting to save {filename}...')
    try:
        ypred.to_netcdf(filepath,engine='h5netcdf')
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('         File write successful')
        return True
    except Exception:
        logger.exception('         Failed to save or verify')
        return False

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate POD ramp models on a chosen data split.')
    parser.add_argument('--split',required=True,choices=['valid','test'],help='Which split to evaluate: `valid` or `test`.')
    args = parser.parse_args()
    config    = Config()
    pod       = config.pod
    targetvar = config.targetvar
    modeldir  = os.path.join(config.modelsdir,'pod')
    logger.info('Evaluating POD models...')
    for runname,runconfig in pod['runs'].items():
        logger.info(f'   Evaluating `{runname}`...')
        logger.info(f'      Loading {args.split} split...')
        inputvar = runconfig['inputvar']
        x = load(args.split,config.splitsdir,inputvar=inputvar)
        model = fetch(runname,modeldir)
        ypred = predict(model,x,targetvar)
        save(ypred,runname,args.split,config.predsdir)
        del x,model,ypred
