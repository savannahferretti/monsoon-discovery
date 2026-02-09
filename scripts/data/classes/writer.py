#!/usr/bin/env python

import os
import json
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

class PredictionWriter:

    def __init__(self,statsdir):
        '''
        Purpose: Initialize PredictionWriter with denormalization statistics.
        Args:
        - statsdir (str): directory containing stats.json
        '''
        filepath = os.path.join(statsdir,'stats.json')
        with open(filepath,'r',encoding='utf-8') as f:
            stats = json.load(f)
        self.pr_mean = stats['pr_mean']
        self.pr_std  = stats['pr_std']

    def to_array(self,preds,valid,refda):
        '''
        Purpose: Place flat predictions back onto the (lat, lon, time) grid and denormalize to mm/hr.
        Args:
        - preds (np.ndarray): flat predictions with shape (nsamp,)
        - valid (np.ndarray): boolean array with shape (nlat*nlon*ntime,) indicating kept samples
        - refda (xr.DataArray): reference DataArray with (time, lat, lon) coordinates
        Returns:
        - np.ndarray: denormalized predictions on the full grid with shape (time, lat, lon)
        '''
        arr = np.full(valid.shape,np.nan,dtype=np.float32)
        arr[valid] = preds
        arr = arr.reshape(refda.shape)
        arr = np.expm1(arr*self.pr_std+self.pr_mean)
        return arr

    def to_dataset(self,predstack,refda):
        '''
        Purpose: Wrap a stacked prediction array into an xr.Dataset with a seed dimension.
        Args:
        - predstack (np.ndarray): predictions with shape (time, lat, lon, nseed)
        - refda (xr.DataArray): reference DataArray with (time, lat, lon) coordinates
        Returns:
        - xr.Dataset: Dataset with precipitation predictions in mm/hr
        '''
        dims   = ('time','lat','lon','seed')
        coords = {dim:refda.coords[dim].values for dim in refda.dims}
        coords['seed'] = np.arange(predstack.shape[-1])
        da = xr.DataArray(predstack,dims=dims,coords=coords,name='pr')
        da.attrs = dict(long_name='Predicted precipitation rate',units='mm/hr')
        return da.to_dataset()

    @staticmethod
    def save(name,ds,split,savedir):
        '''
        Purpose: Save an xr.Dataset to NetCDF and verify by reopening.
        Args:
        - name (str): run name
        - ds (xr.Dataset): Dataset to save
        - split (str): train | valid | test
        - savedir (str): output directory
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(savedir,exist_ok=True)
        filename = f'{name}_{split}_predictions.nc'
        filepath = os.path.join(savedir,filename)
        logger.info(f'   Attempting to save {filename}...')
        try:
            ds.to_netcdf(filepath,engine='h5netcdf')
            xr.open_dataset(filepath,engine='h5netcdf').close()
            logger.info('      File write successful')
            return True
        except Exception:
            logger.exception('      Failed to save or verify')
            return False
