#!/usr/bin/env python

import os
import json
import logging
import warnings
import argparse
import numpy as np
import xarray as xr
from model import PODModel

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS = json.load(f)
FILEDIR     = CONFIGS['paths']['filedir']
MODELDIR    = CONFIGS['paths']['modeldir']
RESULTSDIR  = CONFIGS['paths']['resultsdir']
INPUTVAR    = CONFIGS['dataparams']['inputvar']
LANDVAR     = CONFIGS['dataparams']['landvar']
RUNCONFIGS  = CONFIGS['runs']

def load(splitname,inputvar=INPUTVAR,landvar=LANDVAR,filedir=FILEDIR):
    '''
    Purpose: Load in the evaluation data split (validation or test).
    Args:
    - splitname (str): 'valid' | 'test'
    - inputvar (str): input variable name (defaults to INPUTVAR)
    - landvar (str): land fraction variable name (defaults to LANDVAR)
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - tuple[xr.DataArray,xr.DataArray]: 3D BL/land fraction DataArrays for evaluation
    '''
    if splitname not in ('valid','test'):
        raise ValueError('Splitname must be `valid` or `test`.')
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    evalds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,landvar]]
    x  = evalds[inputvar].load()
    lf = evalds[landvar].load()
    return x,lf

def fetch(runname,modeldir=MODELDIR):
    '''
    Purpose: Load a trained POD model from saved .npz file.
    Args:
    - runname (str): model run name
    - modeldir (str): directory containing model files (defaults to MODELDIR)
    Returns:
    - PODModel: loaded model instance with fitted parameters
    '''
    filename = f'pod_{runname}.npz'
    filepath = os.path.join(modeldir,filename)
    with np.load(filepath) as data:
        mode = str(data['mode'][0])
        if mode=='pooled':
            model = PODModel(
                mode='pooled',
                alphapooled=float(data['alphapooled']),
                blcritpooled=float(data['blcritpooled']))
        elif mode=='regional':
            model = PODModel(
                mode='regional',
                alphaland=float(data['alphaland']),
                blcritland=float(data['blcritland']),
                alphaocean=float(data['alphaocean']),
                blcritocean=float(data['blcritocean']))
    return model

def predict(model,x,lf=None):
    '''
    Purpose: Run the POD forward pass and return precipitation predictions as an xr.DataArray..
    Args:
    - model (PODModel): trained model instance
    - x (xr.DataArray): input 3D BL DataArray
    - lf (xr.DataArray): land fraction DataArray (required for `regional` mode, not used for `pooled`)
    Returns:
    - xr.DataArray: 3D DataArray of predicted precipitation
    '''
    ypredflat = model.forward(x,lf=lf if model.mode=='regional' else None)
    ypred = xr.DataArray(ypredflat.reshape(x.shape),dims=x.dims,coords=x.coords,name='pr')
    ypred.attrs = dict(long_name='POD-predicted precipitation rate',units='mm/hr')
    return ypred

def save(ypred,runname,splitname,resultsdir=RESULTSDIR):
    '''
    Purpose: Save an xr.DataArray of predicted precipitation to a NetCDF file, then verify the write by reopening.
    Args:
    - ypred (xr.DataArray): 3D DataArray of predicted precipitation
    - runname (str): model run name
    - splitname (str): evaluated split label
    - resultsdir (str): output directory (defaults to RESULTSDIR)
    Returns:
    - bool: True if writing and verification succeed, otherwise False
    '''
    os.makedirs(resultsdir,exist_ok=True)
    filename = f'pod_{runname}_{splitname}_pr.nc'
    filepath = os.path.join(resultsdir,filename)
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
    logger.info(f'Loading {args.split} data split...')
    x,lf = load(args.split)
    logger.info('Evaluating POD models...')
    for run in RUNCONFIGS:
        runname     = run['run_name']
        description = run['description']
        logger.info(f'   Evaluating {description}')
        model = fetch(runname)
        ypred = predict(model,x,lf=lf)
        save(ypred,runname,args.split)
        del model,ypred