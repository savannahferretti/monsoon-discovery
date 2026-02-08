#!/usr/bin/env python

import os
import logging
import warnings
import numpy as np
import xarray as xr
from scripts.utils import Config
from scripts.models.pod.model import PODModel

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def load(splitsdir):
    '''
    Purpose: Load regular (non-normalized) training and validation data splits combined for POD fitting.
    Args:
    - splitsdir (str): directory containing split files
    Returns:
    - tuple[xr.DataArray,xr.DataArray,xr.DataArray]: BL/precipitation/land fraction DataArrays
    '''
    dslist = []
    for splitname in ('train','valid'):
        filepath = os.path.join(splitsdir,f'{splitname}.h5')
        ds = xr.open_dataset(filepath,engine='h5netcdf')[['bl','pr','lf']]
        dslist.append(ds)
    trainds = xr.concat(dslist,dim='time')
    bl = trainds['bl'].load()
    pr = trainds['pr'].load()
    lf = trainds['lf'].load()
    return bl,pr,lf

def fit(mode,bl,pr,lf,landthresh,bins,fitparams):
    '''
    Purpose: Fit POD ramp model(s) to training data and return model with diagnostic data.
    Args:
    - mode (str): 'pooled' (single ramp) | 'regional' (separate land/ocean ramps)
    - bl (xr.DataArray): input BL data
    - pr (xr.DataArray): target precipitation data
    - lf (xr.DataArray): land fraction data
    - landthresh (float): threshold for land/ocean classification
    - bins (dict): binning parameters with keys 'min', 'max', 'width', 'minsample'
    - fitparams (dict): fit parameters with keys 'prmin', 'prmax'
    Returns:
    - tuple[PODModel,dict]: trained model instance and diagnostics dictionary with binning data
    '''
    binedges   = np.arange(bins['min'],bins['max']+bins['width'],bins['width'])
    bincenters = 0.5*(binedges[:-1]+binedges[1:])
    samplethresh = bins['minsample']
    prmin = fitparams['prmin']
    prmax = fitparams['prmax']
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
    xflat = bl.values.ravel()
    yflat = pr.values.ravel()
    if mode=='pooled':
        finite  = np.isfinite(xflat)&np.isfinite(yflat)
        results = ramp(xflat[finite],yflat[finite])
        model   = PODModel(mode='pooled',landthresh=landthresh,alphapooled=results[0],blcritpooled=results[1])
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
        model        = PODModel(mode='regional',landthresh=landthresh,alphaland=landresults[0],blcritland=landresults[1],alphaocean=oceanresults[0],blcritocean=oceanresults[1])
        diagnostics = {
            'bincenters':bincenters,
            'ymeanland':landresults[2],
            'fitrangeland':landresults[3],
            'ymeanocean':oceanresults[2],
            'fitrangeocean':oceanresults[3]}
        return model,diagnostics

def save(model,diagnostics,runname,modeldir):
    '''
    Purpose: Save trained model parameters/configuration and diagnostic data to a .npz file, then verify.
    Args:
    - model (PODModel): trained model instance
    - diagnostics (dict): dictionary containing binning and fitting diagnostic data
    - runname (str): model run name
    - modeldir (str): output directory
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
    config    = Config()
    pod       = config.pod
    modeldir  = os.path.join(config.modelsdir,'pod')
    logger.info('Loading regular training + validation data splits combined...')
    bl,pr,lf = load(config.splitsdir)
    logger.info('Training and saving ramp-fit POD models...')
    for runname,runconfig in pod['runs'].items():
        mode = runconfig['mode']
        logger.info(f'   Training `{runname}` ({mode})...')
        model,diagnostics = fit(mode,bl,pr,lf,pod['landthresh'],pod['bins'],pod['fit'])
        save(model,diagnostics,runname,modeldir)
        del model,diagnostics
