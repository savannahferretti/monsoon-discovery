#!/usr/bin/env python

import os
import json
import logging
import warnings
import numpy as np
import xarray as xr
from model import PODModel

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS  = json.load(f)
FILEDIR       = CONFIGS['paths']['filedir']
MODELDIR      = CONFIGS['paths']['modeldir']
INPUTVAR      = CONFIGS['dataparams']['inputvar']
TARGETVAR     = CONFIGS['dataparams']['targetvar']
LANDVAR       = CONFIGS['dataparams']['landvar']
LANDTHRESH    = CONFIGS['dataparams']['landthresh']
BINMIN        = CONFIGS['fitparams']['binmin']
BINMAX        = CONFIGS['fitparams']['binmax']
BINWIDTH      = CONFIGS['fitparams']['binwidth']
SAMPLETHRESH  = CONFIGS['fitparams']['samplethresh']
PRMIN         = CONFIGS['fitparams']['prmin']
PRMAX         = CONFIGS['fitparams']['prmax']
RUNCONFIGS    = CONFIGS['runs']

def load(inputvar=INPUTVAR,targetvar=TARGETVAR,landvar=LANDVAR,filedir=FILEDIR):
    '''
    Purpose: Load in training and validation data splits, which we combine for training.
    Args:
    - inputvar (str): input variable name (defaults to INPUTVAR)
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - landvar (str): land fraction variable name (defaults to LANDVAR)
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - tuple[xr.DataArray,xr.DataArray,xr.DataArray]: 3D BL/precipitation/land fraction DataArrays for training
    '''
    dslist = []
    for splitname in ('train','valid'):
        filename = f'{splitname}.h5'
        filepath = os.path.join(filedir,filename)
        ds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar,landvar]]
        dslist.append(ds)
    trainds = xr.concat(dslist,dim='time')
    x  = trainds[inputvar].load()
    y  = trainds[targetvar].load()
    lf = trainds[landvar].load()
    return x,y,lf

def fit(mode,x,y,lf,landthresh=LANDTHRESH,binmin=BINMIN,binmax=BINMAX,binwidth=BINWIDTH,samplethresh=SAMPLETHRESH,prmin=PRMIN,prmax=PRMAX):
    '''
    Purpose: Fit POD ramp model(s) to training data and return model with diagnostic data.
    Args:
    - mode (str): 'pooled' (single ramp) | 'regional' (separate land/ocean ramps)
    - x (xr.DataArray): input BL data
    - y (xr.DataArray): target precipitation data
    - lf (xr.DataArray): land fraction data
    - landthresh (float): threshold for land/ocean classification (defaults to LANDTHRESH)
    - binmin (float): minimum bin edge (defaults to BINMIN)
    - binmax (float): maximum bin edge (defaults to BINMAX)
    - binwidth (float): bin width (defaults to BINWIDTH)
    - samplethresh (int): minimum samples per bin (defaults to SAMPLETHRESH)
    - prmin (float): minimum precipitation for linear fit (defaults to PRMIN)
    - prmax (float): maximum precipitation for linear fit (defaults to PRMAX)
    Returns:
    - tuple[PODModel,dict]: trained model instance and diagnostics dictionary with binning data
    '''    
    binedges   = np.arange(binmin,binmax+binwidth,binwidth)
    bincenters = 0.5*(binedges[:-1]+binedges[1:])
    def ramp(x,y):
        binidxs = np.digitize(x,binedges)-1
        inrange = (binidxs>=0)&(binidxs<bincenters.size)
        counts = np.bincount(binidxs[inrange],minlength=bincenters.size).astype(np.int64)
        sums = np.bincount(binidxs[inrange],weights=y[inrange],minlength=bincenters.size).astype(np.float32)
        with np.errstate(divide='ignore',invalid='ignore'):
            ymeans = sums/counts
        ymeans[counts<samplethresh] = np.nan
        fitrange = np.isfinite(ymeans)&(ymeans>=prmin)&(ymeans<=prmax)
        alpha,intercept = np.polyfit(bincenters[fitrange],ymeans[fitrange],1)
        blcrit = -intercept/alpha
        return float(alpha),float(blcrit),ymeans,fitrange    
    xflat = x.values.ravel()
    yflat = y.values.ravel()
    if mode=='pooled':
        finite  = np.isfinite(xflat)&np.isfinite(yflat)
        results = ramp(xflat[finite],yflat[finite])
        model   = PODModel(mode='pooled',alphapooled=results[0],blcritpooled=results[1])
        diagnostics = {
            'bincenters':bincenters,
            'ymeanpooled':results[2],
            'fitrangepooled':results[3]}
        return model,diagnostics
    elif mode=='regional':
        lfflat = lf.values.ravel()
        finite = np.isfinite(xflat)&np.isfinite(yflat)&np.isfinite(lfflat)
        land   = finite&(lfflat>=landthresh)
        ocean  = finite&(lfflat<landthresh)
        landresults  = ramp(xflat[land],yflat[land])
        oceanresults = ramp(xflat[ocean],yflat[ocean])
        model        = PODModel(mode='regional',alphaland=landresults[0],blcritland=landresults[1],alphaocean=oceanresults[0],blcritocean=oceanresults[1])
        diagnostics = {
            'bincenters':bincenters,
            'ymeanland':landresults[2],
            'fitrangeland':landresults[3],
            'ymeanocean':oceanresults[2],
            'fitrangeocean':oceanresults[3]}
        return model,diagnostics

def save(model,diagnostics,runname,modeldir=MODELDIR):
    '''
    Purpose: Save trained model parameters/configuration and diagnostic data to a .npz file in the specified directory, then verify the write by reopening.
    Args:
    - model (PODModel): trained model instance
    - diagnostics (dict): dictionary containing binning and fitting diagnostic data
    - runname (str): model run name
    - modeldir (str): output directory (defaults to MODELDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''
    os.makedirs(modeldir,exist_ok=True)
    filename = f'pod_{runname}.npz'
    filepath = os.path.join(modeldir,filename)
    logger.info(f'      Attempting to save {filename}...')
    try:
        if model.mode=='pooled':
            np.savez(filepath,
                     mode=np.array([model.mode],dtype='U10'),
                     alphapooled=model.alphapooled,
                     blcritpooled=model.blcritpooled,
                     bincenters=diagnostics['bincenters'],
                     ymeanpooled=diagnostics['ymeanpooled'],
                     fitrangepooled=diagnostics['fitrangepooled'],
                     nparams=np.int32(model.nparams))
        elif model.mode=='regional':
            np.savez(filepath,
                     mode=np.array([model.mode],dtype='U10'),
                     alphaland=model.alphaland,
                     blcritland=model.blcritland,
                     alphaocean=model.alphaocean,
                     blcritocean=model.blcritocean,
                     bincenters=diagnostics['bincenters'],
                     ymeanland=diagnostics['ymeanland'],
                     fitrangeland=diagnostics['fitrangeland'],
                     ymeanocean=diagnostics['ymeanocean'],
                     fitrangeocean=diagnostics['fitrangeocean'],
                     nparams=np.int32(model.nparams))
        with np.load(filepath) as _:
            pass
        logger.info('         File write successful')
        return True
    except Exception:
        logger.exception('         Failed to save or verify')
        return False

if __name__=='__main__':
    logger.info('Loading training + validation data splits combined...')
    x,y,lf = load()
    logger.info('Training and saving ramp-fit POD models...')
    for run in RUNCONFIGS:
        runname     = run['run_name']
        mode        = run['mode']
        description = run['description']
        logger.info(f'   Training {description}')
        model,diagnostics = fit(mode,x,y,lf)
        save(model,diagnostics,runname)
        del model,diagnostics