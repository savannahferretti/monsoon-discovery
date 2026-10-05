#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import xarray as xr
from timingutils import load_dataset,save_dataset,load_stats,standardize,precip_to_z
from equations import evaluate,load_registry

def load_split(config,split):
    '''
    Purpose: Load a (non-normalized) split as float64.
    Args:
    - config (TimingConfig): configuration object
    - split (str): 'train' | 'valid' | 'test'
    Returns:
    - xr.Dataset: split Dataset
    '''
    return load_dataset(os.path.join(config.splitsdir,f'{split}.h5'))

def flatten(da,ntime):
    '''
    Purpose: Flatten a (time, lat, lon) or static (lat, lon) field to samples ordered by (time, lat, lon).
    Args:
    - da (xr.DataArray): field
    - ntime (int): number of timesteps
    Returns:
    - np.ndarray: flat array with shape (ntime*nlat*nlon,)
    '''
    if 'time' in da.dims:
        return da.transpose('time','lat','lon').values.reshape(-1)
    return np.tile(da.transpose('lat','lon').values,(ntime,1,1)).reshape(-1)

def flatten_profile(da):
    '''
    Purpose: Flatten a (time, lat, lon, sig) field to (samples, sig).
    Args:
    - da (xr.DataArray): profile field
    Returns:
    - np.ndarray: array with shape (ntime*nlat*nlon, nsig)
    '''
    return da.transpose('time','lat','lon','sig').values.reshape(-1,da.sizes['sig'])

def load_kernels(config,dsig,run='nn_gauss'):
    '''
    Purpose: Load NN kernel weights for all available seeds, renormalize each so that sum(k·dsig) = 1 exactly in
        float64, and average across seeds.
    Args:
    - config (TimingConfig): configuration object
    - dsig (np.ndarray): sigma thickness weights
    - run (str): NN run name
    Returns:
    - dict[str, np.ndarray]: field variable → kernel weights with shape (nsig,)
    '''
    kernels = []
    for seed in config.nn['seeds']:
        filepath = os.path.join(config.weightsdir,f'{run}_{seed}_weights.nc')
        if os.path.exists(filepath):
            ds = load_dataset(filepath)
            k  = ds['k'].values
            kernels.append(k/(k*dsig[None,:]).sum(axis=1,keepdims=True))
            fieldvars = [str(var) for var in ds['field'].values]
    if not kernels:
        raise FileNotFoundError(f'No `{run}` kernel weights in {config.weightsdir}')
    return dict(zip(fieldvars,np.mean(kernels,axis=0)))

def load_physical(config,split,fieldvars,localvars,weightsfrom=None):
    '''
    Purpose: Physical-unit predictors for one split: kernel-integrated profiles (if weightsfrom), scalar fields,
        and local variables, all float64 and flattened to (time, lat, lon) order.
    Args:
    - config (TimingConfig): configuration object
    - split (str): split name
    - fieldvars (list[str]): field variables
    - localvars (list[str]): local variables
    - weightsfrom (str | None): NN run whose kernels integrate profile variables
    Returns:
    - tuple[dict[str, np.ndarray], np.ndarray, xr.DataArray]: physical predictors, precipitation (mm), and the
        truth DataArray (time, lat, lon) for coordinates
    '''
    ds    = load_split(config,split)
    ntime = ds.sizes['time']
    inputs = {}
    if weightsfrom and fieldvars:
        dsig    = ds['dsig'].values
        kernels = load_kernels(config,dsig,weightsfrom)
        for var in fieldvars:
            inputs[var] = (flatten_profile(ds[var])*(kernels[var]*dsig)[None,:]).sum(axis=1)
    else:
        for var in fieldvars:
            inputs[var] = flatten(ds[var],ntime)
    for var in localvars:
        inputs[var] = flatten(ds[var],ntime)
    truth = ds['tp'].transpose('time','lat','lon')
    return inputs,flatten(truth,ntime),truth

def load_features(config,split,runconfig,timeoffset=0):
    '''
    Purpose: Standardized SR predictors and target for one split. Profiles are kernel-integrated in physical units
        and then standardized, so standardized and physical-space equations see identical inputs.
    Args:
    - config (TimingConfig): configuration object
    - split (str): split name
    - runconfig (dict): SR run configuration
    - timeoffset (int): offset added to the time index (to keep train and valid timesteps distinct)
    Returns:
    - tuple[pd.DataFrame, np.ndarray, xr.DataArray, np.ndarray]: features (with 'timeidx'), standardized target,
        truth DataArray, and valid-sample mask
    '''
    stats     = load_stats(config)
    fieldvars = runconfig['fieldvars']
    localvars = runconfig.get('localvars',[])
    inputs,tp,truth = load_physical(config,split,fieldvars,localvars,runconfig.get('weightsfrom'))
    columns = {var:(values if var=='lf' else standardize(values,stats,var)) for var,values in inputs.items()}
    residualfrom = runconfig.get('residualfrom')
    if residualfrom:
        basename = config.eqname(residualfrom) or residualfrom
        entry    = load_registry(config)[basename]
        baserun  = config.sr['runs'][config.sr['optimizedeqs'][basename]['runfrom']]
        basex,_,_,_ = load_features(config,split,baserun)
        columns[residualfrom] = evaluate(entry['form'],{c:basex[c].values for c in basex.columns if c!='timeidx'},entry['constants'])
    features = pd.DataFrame(columns)
    nspace   = truth.sizes['lat']*truth.sizes['lon']
    features['timeidx'] = np.repeat(np.arange(truth.sizes['time']),nspace)+timeoffset
    target = precip_to_z(tp,stats)
    valid  = np.isfinite(features.drop(columns=['timeidx']).values).all(axis=1)&np.isfinite(target)
    return features,target,truth,valid

def load_nn_arrays(config,split,runconfig):
    '''
    Purpose: Standardized NN inputs and target for one split (float64; convert to float32 tensors in the NN code).
    Args:
    - config (TimingConfig): configuration object
    - split (str): split name
    - runconfig (dict): NN run configuration
    Returns:
    - tuple: fields (nsamp, nfieldvars, nlevs), local (nsamp, nlocalvars), target (nsamp,), dsig (nlevs,), nlevs,
        valid-sample mask over all grid samples, and truth DataArray — fields, local, and target already masked
    '''
    stats     = load_stats(config)
    fieldvars = runconfig['fieldvars']
    localvars = runconfig.get('localvars',[])
    ds        = load_split(config,split)
    ntime     = ds.sizes['time']
    truth     = ds['tp'].transpose('time','lat','lon')
    nsamp     = truth.size
    if fieldvars and 'sig' in ds[fieldvars[0]].dims:
        fields = np.stack([standardize(flatten_profile(ds[var]),stats,var) for var in fieldvars],axis=1)
        dsig   = ds['dsig'].values
    elif fieldvars:
        fields = np.stack([standardize(flatten(ds[var],ntime),stats,var)[:,None] for var in fieldvars],axis=1)
        dsig   = np.ones(1)
    else:
        fields = np.empty((nsamp,0,1))
        dsig   = np.ones(1)
    local  = np.stack([flatten(ds[var],ntime) if var=='lf' else standardize(flatten(ds[var],ntime),stats,var) for var in localvars],axis=1) if localvars else np.empty((nsamp,0))
    target = precip_to_z(flatten(truth,ntime),stats)
    valid  = np.isfinite(fields).all(axis=(1,2))&np.isfinite(local).all(axis=1)&np.isfinite(target)
    return fields[valid],local[valid],target[valid],dsig,fields.shape[2],valid,truth

def unflatten(flat,valid,truth):
    '''
    Purpose: Place valid-sample values back on the (time, lat, lon) grid, NaN elsewhere.
    Args:
    - flat (np.ndarray): values for valid samples
    - valid (np.ndarray): valid-sample mask
    - truth (xr.DataArray): reference grid
    Returns:
    - np.ndarray: float64 grid
    '''
    grid = np.full(valid.shape,np.nan)
    grid[valid] = flat
    return grid.reshape(truth.shape)

def save_predictions(config,name,split,grids,truth,seeds=None):
    '''
    Purpose: Save gridded precipitation predictions (mm) as {name}_{split}_predictions.nc.
    Args:
    - config (TimingConfig): configuration object
    - name (str): model or equation name
    - split (str): split name
    - grids (np.ndarray | list[np.ndarray]): one (time, lat, lon) grid, or one per seed
    - truth (xr.DataArray): reference grid
    - seeds (list[int] | None): seeds matching grids, if a list
    '''
    coords = {dim:truth[dim] for dim in ('time','lat','lon')}
    if seeds is None:
        da = xr.DataArray(grids,dims=('time','lat','lon'),coords=coords)
    else:
        da = xr.DataArray(np.stack(grids,axis=-1),dims=('time','lat','lon','seed'),coords={**coords,'seed':seeds})
    da.attrs = dict(long_name='Predicted total precipitation',units='mm')
    save_dataset(da.to_dataset(name='tp'),os.path.join(config.predsdir,f'{name}_{split}_predictions.nc'))
