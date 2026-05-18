#!/usr/bin/env python

import os
import logging
import warnings
import numpy as np
import xarray as xr
from scripts.utils import Config
from scripts.models.pod.model import RampPOD

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def load(splitsdir,inputvar,targetvar):
    '''
    Purpose: Load regular (non-normalized) training and validation data splits combined for POD fitting.
    Args:
    - splitsdir (str): directory containing split files
    - inputvar (str): input variable name
    - targetvar (str): target variable name
    Returns:
    - tuple[xr.DataArray,xr.DataArray]: input and target DataArrays
    '''
    dslist = []
    for splitname in ('train','valid'):
        filepath = os.path.join(splitsdir,f'{splitname}.h5')
        ds = xr.open_dataset(filepath,engine='h5netcdf')[[inputvar,targetvar]]
        dslist.append(ds)
    trainds = xr.concat(dslist,dim='time')
    x = trainds[inputvar].load()
    y = trainds[targetvar].load()
    return x,y

def fit(x,y,bins,fitparams):
    '''
    Purpose: Fit a ramp model to training data and return the model with diagnostic data.
    Args:
    - x (xr.DataArray): input data
    - y (xr.DataArray): target data
    - bins (dict): binning parameters with keys 'min', 'max', 'width', 'minsample'
    - fitparams (dict): fit parameters with keys 'ymin', 'ymax'
    Returns:
    - tuple[RampPOD,dict]: trained RampPOD instance and diagnostics dictionary with binning data
    '''
    binedges     = np.arange(bins['min'],bins['max']+bins['width'],bins['width'])
    bincenters   = 0.5*(binedges[:-1]+binedges[1:])
    samplethresh = bins['minsample']
    ymin = fitparams['ymin']
    ymax = fitparams['ymax']
    xflat  = x.values.ravel()
    yflat  = y.values.ravel()
    finite = np.isfinite(xflat)&np.isfinite(yflat)
    xflat  = xflat[finite]
    yflat  = yflat[finite]
    binidxs = np.digitize(xflat,binedges)-1
    inrange = (binidxs>=0)&(binidxs<bincenters.size)
    counts = np.bincount(binidxs[inrange],minlength=bincenters.size).astype(np.int64)
    sums   = np.bincount(binidxs[inrange],weights=yflat[inrange],minlength=bincenters.size).astype(np.float32)
    with np.errstate(divide='ignore',invalid='ignore'):
        ymeans = sums/counts
    ymeans[counts<samplethresh] = np.nan
    fitrange = np.isfinite(ymeans)&(ymeans>=ymin)&(ymeans<=ymax)
    alpha,intercept = np.polyfit(bincenters[fitrange],ymeans[fitrange],1)
    xcrit = -intercept/alpha
    model = RampPOD(alpha=float(alpha),xcrit=float(xcrit))
    diagnostics = {
        'bincenters':bincenters,
        'ymean':ymeans,
        'fitrange':fitrange}
    return model,diagnostics

def save(model,diagnostics,runname,modeldir):
    '''
    Purpose: Save trained model parameters/configuration and diagnostic data to a .npz file, then verify.
    Args:
    - model (RampPOD): trained RampPOD instance
    - diagnostics (dict): dictionary containing binning and fitting diagnostic data
    - runname (str): model run name
    - modeldir (str): output directory
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''
    os.makedirs(modeldir,exist_ok=True)
    filename = f'{runname}.npz'
    filepath = os.path.join(modeldir,filename)
    logger.info(f'      Attempting to save {filename}...')
    try:
        np.savez(filepath,
                 alpha=model.alpha,
                 xcrit=model.xcrit,
                 bincenters=diagnostics['bincenters'],
                 ymean=diagnostics['ymean'],
                 fitrange=diagnostics['fitrange'],
                 nparams=np.int32(model.nparams))
        with np.load(filepath) as _:
            pass
        logger.info('         File write successful')
        return True
    except Exception:
        logger.exception('         Failed to save or verify')
        return False

if __name__=='__main__':
    config    = Config()
    pod       = config.pod
    targetvar = config.targetvar
    modeldir  = os.path.join(config.modelsdir,'pod')
    logger.info('Training and saving ramp-fit POD models...')
    cachedvars = None
    cacheddata = None
    for runname,runconfig in pod['runs'].items():
        inputvar = runconfig['inputvar']
        if (inputvar,targetvar)!=cachedvars:
            logger.info(f'Loading combined training and validation splits...')
            x,y = load(config.splitsdir,inputvar=inputvar,targetvar=targetvar)
            cachedvars = (inputvar,targetvar)
            cacheddata = (x,y)
        else:
            x,y = cacheddata
        logger.info(f'   Training `{runname}`...')
        model,diagnostics = fit(x,y,pod['bins'],pod['fit'])
        save(model,diagnostics,runname,modeldir)
        del model,diagnostics
