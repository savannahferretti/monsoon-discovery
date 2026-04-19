#!/usr/bin/env python

import os
import json
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

TARGETMETA = {
    'pr':{'longname':'Predicted precipitation rate','units':'mm/hr'},
    'tp':{'longname':'Predicted total precipitation','units':'mm'}}

class PredictionWriter:

    def __init__(self,statsdir,targetvar='pr'):
        '''
        Purpose: Initialize PredictionWriter with denormalization statistics and target metadata.
        Args:
        - statsdir (str): directory containing stats.json
        - targetvar (str): target variable name ('pr' | 'tp')
        '''
        filepath = os.path.join(statsdir,'stats.json')
        with open(filepath,'r',encoding='utf-8') as f:
            stats = json.load(f)
        self.targetvar = targetvar
        self.mean      = stats[f'{targetvar}_mean']
        self.std       = stats[f'{targetvar}_std']
        self.longname  = TARGETMETA[targetvar]['longname']
        self.units     = TARGETMETA[targetvar]['units']

    def unflatten(self,flat,valid,refda):
        '''
        Purpose: Place flat values back onto the (time, lat, lon) grid, filling invalid samples with NaN.
        Args:
        - flat (np.ndarray): flat values with shape (nsamples,)
        - valid (np.ndarray): boolean array with shape (nsamples,) indicating kept samples
        - refda (xr.DataArray): reference DataArray with (time, lat, lon) coordinates
        Returns:
        - np.ndarray: gridded array with shape (time, lat, lon)
        '''
        arr = np.full(valid.shape,np.nan,dtype=np.float32)
        arr[valid] = flat
        return arr.reshape(refda.shape)

    def predictions_to_dataset(self,predslist,valid,refda):
        '''
        Purpose: Unflatten, denormalize, and wrap a list of per-seed predictions into an xr.Dataset.
        Args:
        - predslist (list[np.ndarray]): per-seed flat predictions, each with shape (nsamples,)
        - valid (np.ndarray): boolean array with shape (nsamples,) indicating kept samples
        - refda (xr.DataArray): reference DataArray with (time, lat, lon) coordinates
        Returns:
        - xr.Dataset: Dataset with predictions in native units on a (time, lat, lon, seed) grid
        '''
        predstack = np.stack([np.expm1(self.unflatten(preds,valid,refda)*self.std+self.mean) for preds in predslist],axis=-1)
        coords = {dim:refda.coords[dim].values for dim in refda.dims}
        coords['seed'] = np.arange(len(predslist))
        da = xr.DataArray(predstack,dims=('time','lat','lon','seed'),coords=coords,name=self.targetvar,
                          attrs=dict(long_name=self.longname,units=self.units))
        return da.to_dataset()

    @staticmethod
    def weights_to_dataset(weights,fieldvars,refds,components=None):
        '''
        Purpose: Wrap a kernel weight array into an xr.Dataset with a seed dimension.
        Args:
        - weights (np.ndarray): normalized kernel weights with shape (nfieldvars, nlevs, nseed)
        - fieldvars (list[str]): predictor field variable names
        - refds (xr.Dataset): reference Dataset for sig coordinates
        - components (np.ndarray | None): mixture kernel components with shape (2, nfieldvars, nlevs, nseed),
          or None if not applicable; NaN entries indicate fields without a mixture kernel
        Returns:
        - xr.Dataset: Dataset with normalized kernel weights, and mixture components if provided
        '''
        coords = {'field':fieldvars}
        coords['sig']  = refds.coords['sig'].values if 'sig' in refds.coords else np.arange(weights.shape[1])
        coords['seed'] = np.arange(weights.shape[-1])
        da = xr.DataArray(weights,dims=('field','sig','seed'),coords=coords,
                          attrs=dict(long_name='Normalized kernel weights',units='N/A'))
        ds = da.to_dataset(name='k')
        if components is not None:
            comp_coords = {'component':[0,1],**coords}
            comp_da = xr.DataArray(components,dims=('component','field','sig','seed'),coords=comp_coords,
                                   attrs=dict(long_name='Normalized mixture kernel components',units='N/A'))
            ds['kc'] = comp_da
        return ds

    def save(self,ds,name,kind,split,savedir,timechunksize=736):
        '''
        Purpose: Save an xr.Dataset to NetCDF and verify by reopening.
        Args:
        - ds (xr.Dataset): Dataset to save
        - name (str): run name
        - kind (str): 'predictions' | 'features' | 'weights'
        - split (str): 'train' | 'valid' | 'test'
        - savedir (str): output directory
        - timechunksize (int): chunk size for time dimension (defaults to 736 for 3-month chunks on 3-hourly data)
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(savedir,exist_ok=True)
        filename = f'{name}_{split}_{kind}.nc'
        filepath = os.path.join(savedir,filename)
        logger.info(f'   Attempting to save {filename}...')
        ds.load()
        encoding = {}
        for varname,da in ds.data_vars.items():
            chunks = []
            for dim,size in zip(da.dims,da.shape):
                chunks.append(min(timechunksize,size) if dim=='time' else size)
            encoding[varname] = {'chunksizes':tuple(chunks)}
        try:
            ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
            xr.open_dataset(filepath,engine='h5netcdf').close()
            logger.info('      File write successful')
            return True
        except Exception:
            logger.exception('      Failed to save or verify')
            return False
