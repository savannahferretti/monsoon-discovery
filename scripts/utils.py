#!/usr/bin/env python

import os
import json
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

def load(filepath,lazy=False):
    '''
    Purpose: Read a NetCDF/HDF5 file and convert floating-point data variables to float64.
    Args:
    - filepath (str): file path
    - lazy (bool): if True, open with dask and defer reading (for files too large to hold in memory at once)
    Returns:
    - xr.Dataset: Dataset with float64 data variables
    '''
    if lazy:
        ds = xr.open_dataset(filepath,engine='h5netcdf',chunks={})
    else:
        with xr.open_dataset(filepath,engine='h5netcdf') as ds:
            ds = ds.load()
    return ds.assign({name:da.astype(np.float64) for name,da in ds.data_vars.items() if da.dtype.kind=='f'})

def save(ds,filepath,timechunksize=736):
    '''
    Purpose: Convert floating-point data variables to float32, save to NetCDF/HDF5, and verify by reopening.
    Args:
    - ds (xr.Dataset): Dataset to save
    - filepath (str): output path
    - timechunksize (int): chunk size for time dimension (defaults to 736 for 3-month chunks on 3-hourly data)
    '''
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    logger.info(f'   Attempting to save {os.path.basename(filepath)}...')
    ds = ds.copy(deep=False).assign({name:da.astype(np.float32) for name,da in ds.data_vars.items() if da.dtype.kind=='f'})
    for variable in ds.variables.values():
        variable.encoding = {}
    encoding = {name:{'chunksizes':tuple(min(timechunksize,size) if dim=='time' else size for dim,size in zip(da.dims,da.shape))}
                for name,da in ds.data_vars.items() if da.ndim>0}
    ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
    with xr.open_dataset(filepath,engine='h5netcdf') as check:
        wrong = {name:str(da.dtype) for name,da in check.data_vars.items() if da.dtype.kind=='f' and da.dtype!=np.float32}
    if wrong:
        raise TypeError(f'{filepath} has non-float32 variables: {wrong}')
    logger.info('      File write successful')

def load_stats(splitsdir):
    '''
    Purpose: Load training statistics (mean and standard deviation of each variable) from stats.json.
    Args:
    - splitsdir (str): directory containing stats.json
    Returns:
    - dict[str,float]: statistics keyed by '{var}_mean' and '{var}_std'
    '''
    with open(os.path.join(splitsdir,'stats.json'),'r',encoding='utf-8') as f:
        return json.load(f)

def standardize(values,var,stats):
    '''
    Purpose: Standardize a variable with training statistics. Precipitation is log1p-transformed first and land
    fraction is returned unchanged.
    Args:
    - values (np.ndarray): values in physical units
    - var (str): variable name
    - stats (dict[str,float]): training statistics
    Returns:
    - np.ndarray: standardized values
    '''
    if var=='lf':
        return values
    if var in ('pr','tp'):
        values = np.log1p(values)
    return (values-stats[f'{var}_mean'])/stats[f'{var}_std']

def flatten(da,ntime):
    '''
    Purpose: Flatten a (time, lat, lon) field, or a static (lat, lon) field repeated in time, to samples.
    Args:
    - da (xr.DataArray): field
    - ntime (int): number of timesteps
    Returns:
    - np.ndarray: values with shape (ntime*nlat*nlon,)
    '''
    if 'time' in da.dims:
        return da.transpose('time','lat','lon').values.reshape(-1)
    return np.tile(da.transpose('lat','lon').values,(ntime,1,1)).reshape(-1)

class Config:

    def __init__(self,path=None):
        '''
        Purpose: Load configurations from a JSON file and expose commonly used blocks/paths as attributes.
        '''
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'configs.json')
        with open(path,'r',encoding='utf-8') as f:
            config = json.load(f)
            self.filepaths   = config['filepaths']
            self.metadata    = config['metadata']
            self.domain      = config['domain']
            self.splits      = config['splits']
            self.experiments = config['experiments']

    @property
    def rawdir(self):
        return self.filepaths['raw']

    @property
    def interimdir(self):
        return self.filepaths['interim']

    @property
    def splitsdir(self):
        return self.filepaths['splits']

    @property
    def predsdir(self):
        return self.filepaths['predictions']

    @property
    def featsdir(self):
        return self.filepaths['features']

    @property
    def weightsdir(self):
        return self.filepaths['weights']

    @property
    def modelsdir(self):
        return self.filepaths['models']

    @property
    def author(self):
        return self.metadata['author']

    @property
    def email(self):
        return self.metadata['email']

    @property
    def latrange(self):
        latmin,latmax = self.domain['latrange']
        return float(latmin),float(latmax)

    @property
    def lonrange(self):
        lonmin,lonmax = self.domain['lonrange']
        return float(lonmin),float(lonmax)

    @property
    def levrange(self):
        levmin,levmax = self.domain['levrange']
        return float(levmin),float(levmax)

    @property
    def years(self):
        return self.domain['years']

    @property
    def months(self):
        return self.domain['months']

    @property
    def timewindow(self):
        return int(self.domain['timewindow'])

    @property
    def trainrange(self):
        trainstart,trainend = self.splits['trainyears']
        return int(trainstart),int(trainend)

    @property
    def validrange(self):
        validstart,validend = self.splits['validyears']
        return int(validstart),int(validend)

    @property
    def testrange(self):
        teststart,testend = self.splits['testyears']
        return int(teststart),int(testend)

    @property
    def targetvar(self):
        return self.domain['target']

    @property
    def nn(self):
        return self.experiments['nn']

    @property
    def sr(self):
        return self.experiments['sr']
