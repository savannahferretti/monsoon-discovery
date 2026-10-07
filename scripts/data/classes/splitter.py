#!/usr/bin/env python

import os
import json
import glob
import logging
import numpy as np
import xarray as xr
from scripts.utils import load,save

logger = logging.getLogger(__name__)

class DataSplitter:

    def __init__(self,filedir,savedir,trainrange,validrange,testrange):
        '''
        Purpose: Initialize DataSplitter with configuration parameters.
        Args:
        - filedir (str): directory containing NetCDF files
        - savedir (str): directory to save split files
        - trainrange (tuple[int,int]): inclusive year range for training split
        - validrange (tuple[int,int]): inclusive year range for validation split
        - testrange (tuple[int,int]): inclusive year range for test split
        '''
        self.filedir    = filedir
        self.savedir    = savedir
        self.trainrange = trainrange
        self.validrange = validrange
        self.testrange  = testrange

    def combine(self):
        '''
        Purpose: Load all interim NetCDF files (float64) into a single xr.Dataset.
        Returns:
        - xr.Dataset: Dataset with every interim variable
        '''
        datavars = {}
        for filepath in sorted(glob.glob(os.path.join(self.filedir,'*.nc'))):
            for name,da in load(filepath).data_vars.items():
                datavars[name] = da.transpose(*[dim for dim in ('lat','lon','sig','time') if dim in da.dims])
        return xr.Dataset(datavars)

    def split(self,ds,splitrange):
        '''
        Purpose: Select the years of a split.
        Args:
        - ds (xr.Dataset): Dataset from combine()
        - splitrange (tuple[int,int]): inclusive year range for the split
        Returns:
        - xr.Dataset: split Dataset
        '''
        return ds.sel(time=(ds.time.dt.year>=splitrange[0])&(ds.time.dt.year<=splitrange[1]))

    def calc_stats(self,trainds):
        '''
        Purpose: Compute training-set statistics (float64) for each variable, save to JSON, and verify by reopening.
        Args:
        - trainds (xr.Dataset): training Dataset
        Returns:
        - dict[str,float]: training set mean and standard deviation for select variables
        '''
        stats = {}
        for varname,da in trainds.data_vars.items():
            if varname in ('dsig','lf'):
                continue
            arr = np.log1p(da.values) if varname in ('pr','tp') else da.values
            stats[f'{varname}_mean'] = float(np.nanmean(arr))
            stats[f'{varname}_std']  = float(np.nanstd(arr))
        os.makedirs(self.savedir,exist_ok=True)
        filepath = os.path.join(self.savedir,'stats.json')
        with open(filepath,'w',encoding='utf-8') as f:
            json.dump(stats,f)
        with open(filepath,'r',encoding='utf-8') as f:
            if json.load(f)!=stats:
                raise ValueError(f'{filepath} does not match the computed statistics')
        logger.info('   Wrote statistics to stats.json')
        return stats

    def save(self,ds,splitname,timechunksize=736):
        '''
        Purpose: Save an xr.Dataset to an HDF5 file and verify by reopening.
        Args:
        - ds (xr.Dataset): Dataset to save
        - splitname (str): 'train' | 'valid' | 'test'
        - timechunksize (int): chunk size for time dimension (defaults to 736 for 3-month chunks on 3-hourly data)
        '''
        save(ds,os.path.join(self.savedir,f'{splitname}.h5'),timechunksize)