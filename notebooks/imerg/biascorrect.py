#!/usr/bin/env python

import os
import numpy as np
import xarray as xr

def load_pairs(splitsdir,splits=('train','valid')):
    '''
    Purpose: Load paired ERA5 total precipitation and IMERG V06 precipitation rate samples from the given splits.
    Args:
    - splitsdir (str): directory containing the split HDF5 files
    - splits (tuple[str]): split names to pool
    Returns:
    - tuple[np.ndarray, np.ndarray]: finite paired ERA5 (mm) and IMERG (mm/hr) values
    '''
    era5,imerg = [],[]
    for split in splits:
        with xr.open_dataset(os.path.join(splitsdir,f'{split}.h5'),engine='h5netcdf') as ds:
            tp,pr = xr.align(ds['tp'],ds['pr'],join='inner')
            era5.append(tp.transpose('time','lat','lon').values.ravel())
            imerg.append(pr.transpose('time','lat','lon').values.ravel())
    era5,imerg = np.concatenate(era5),np.concatenate(imerg)
    finite     = np.isfinite(era5)&np.isfinite(imerg)
    return era5[finite],imerg[finite]

def fit_qm(source,target,nquantiles=200):
    '''
    Purpose: Fit an empirical quantile mapping over the full distributions, zeros included, so the corrected wet fraction matches the target.
    Args:
    - source (np.ndarray): values to correct (ERA5)
    - target (np.ndarray): reference values (IMERG)
    - nquantiles (int): number of quantiles defining the mapping above the dry fraction
    Returns:
    - callable: mapping that sends values at or below the source quantile matching the larger dry fraction to 0, maps larger values through the quantile transfer function, and keeps NaN as NaN
    '''
    dryfrac   = max(np.mean(source<=0),np.mean(target<=0))
    quantiles = np.linspace(dryfrac,1,nquantiles)
    sourceq   = np.maximum.accumulate(np.quantile(source,quantiles))
    targetq   = np.maximum.accumulate(np.maximum(np.quantile(target,quantiles),0.0))
    def qm(x):
        x   = np.asarray(x,dtype=float)
        wet = x>sourceq[0]
        out = np.where(wet,np.interp(np.where(wet,x,sourceq[0]),sourceq,targetq),0.0)
        return np.where(np.isnan(x),np.nan,out)
    return qm

def apply_qm(mapping,da):
    '''
    Purpose: Apply a quantile mapping to a DataArray, keeping its dims, coords, and attrs.
    Args:
    - mapping (callable): mapping returned by fit_qm
    - da (xr.DataArray): values to correct
    Returns:
    - xr.DataArray: bias-corrected values
    '''
    return xr.DataArray(mapping(da.values),dims=da.dims,coords=da.coords,attrs=da.attrs)
