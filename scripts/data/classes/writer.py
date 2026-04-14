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

    def features_to_dataset(self,featslist,fieldvars,valid,refda):
        '''
        Purpose: Unflatten and wrap a list of per-seed kernel-integrated features into an xr.Dataset.
        Args:
        - featslist (list[np.ndarray]): per-seed features, each with shape (nsamples, nfieldvars)
        - fieldvars (list[str]): predictor field variable names
        - valid (np.ndarray): boolean array with shape (nsamples,) indicating kept samples
        - refda (xr.DataArray): reference DataArray with (time, lat, lon) coordinates
        Returns:
        - xr.Dataset: Dataset with a per-field DataArray on a (time, lat, lon, seed) grid
        '''
        coords = {dim:refda.coords[dim].values for dim in refda.dims}
        coords['seed'] = np.arange(len(featslist))
        ds = xr.Dataset()
        for i,var in enumerate(fieldvars):
            featstack = np.stack([self.unflatten(feats[:,i],valid,refda) for feats in featslist],axis=-1)
            ds[var] = xr.DataArray(featstack,dims=('time','lat','lon','seed'),coords=coords,name=var,
                                   attrs=dict(long_name=f'Kernel-integrated {var}',units='N/A'))
        return ds

    @staticmethod
    def weights_to_dataset(components,fieldvars,refds):
        '''
        Purpose: Wrap kernel weight component arrays into an xr.Dataset with a seed dimension.
        Args:
        - components (list[np.ndarray]): list of component arrays, each with shape (nfieldvars, nlevs, nseed)
        - fieldvars (list[str]): predictor field variable names
        - refds (xr.Dataset): reference Dataset for sig coordinates
        Returns:
        - xr.Dataset: Dataset with normalized kernel weight components
        '''
        coords = {'field':fieldvars}
        coords['sig']  = refds.coords['sig'].values if 'sig' in refds.coords else np.arange(components[0].shape[1])
        coords['seed'] = np.arange(components[0].shape[-1])
        ds = xr.Dataset()
        for i,comp in enumerate(components):
            da = xr.DataArray(comp,dims=('field','sig','seed'),coords=coords,
                              attrs=dict(long_name=f'Normalized kernel weights (component {i+1})',units='N/A'))
            ds[f'k{i+1}'] = da
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
