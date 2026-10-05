#!/usr/bin/env python

import os
import sys
import json
import logging
import numpy as np
import xarray as xr

TIMINGDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR   = os.path.dirname(os.path.dirname(TIMINGDIR))
if REPODIR not in sys.path:
    sys.path.insert(0,REPODIR)

from scripts.utils import Config

OUTPUTDIRS   = ('interim','splits','predictions','features','weights','models')
COMPUTEDTYPE = np.float64
STOREDTYPE   = np.float32

logger = logging.getLogger(__name__)

class TimingConfig(Config):

    def __init__(self,variant=None):
        '''
        Purpose: Load the main configurations and redirect all output directories to tests/timing/data/{variant}.
            With variant=None, all paths point to the main (current setup) data, which is only read.
        Args:
        - variant (str | None): timing variant name from tests/timing/configs.json
        '''
        super().__init__()
        with open(os.path.join(TIMINGDIR,'configs.json'),'r',encoding='utf-8') as f:
            self.timing = json.load(f)
        self.mainfilepaths = dict(self.filepaths)
        self.variant       = variant
        if variant is not None:
            if variant not in self.timing['variants']:
                raise ValueError(f'Unknown variant `{variant}`; must be one of {list(self.timing["variants"])}')
            datadir = os.path.join(TIMINGDIR,'data',variant)
            self.filepaths = {**self.mainfilepaths,**{key:os.path.join(datadir,key) for key in OUTPUTDIRS}}

    @property
    def datadir(self):
        return os.path.join(TIMINGDIR,'data',self.variant) if self.variant else None

    @property
    def resultsdir(self):
        return os.path.join(TIMINGDIR,'results')

    @property
    def nnruns(self):
        return {name:self.nn['runs'][name] for name in self.timing['nn']['runs']}

    @property
    def srruns(self):
        extra = self.timing['sr'].get('extraruns',{})
        return {name:extra[name] if name in extra else self.sr['runs'][name] for name in self.timing['sr']['runs']}

    @property
    def srequations(self):
        return {name:self.sr['optimizedeqs'][name] for name in self.timing['sr']['optimizedeqs']}

def parse_names(arg,allnames):
    '''
    Purpose: Turn a comma-separated CLI argument into an ordered list of names.
    Args:
    - arg (str): comma-separated names, or `all`
    - allnames (list[str]): valid names in order
    Returns:
    - list[str]: selected names
    '''
    if arg=='all':
        return list(allnames)
    names = [name.strip() for name in arg.split(',') if name.strip()]
    unknown = [name for name in names if name not in allnames]
    if unknown:
        raise ValueError(f'Unknown name(s) {unknown}; must be from {list(allnames)}')
    return names

def load_dataset(filepath):
    '''
    Purpose: Read a NetCDF/HDF5 file fully into memory and convert every floating-point data variable to
        COMPUTEDTYPE. This is the only place files are read in the timing test.
    Args:
    - filepath (str): file path
    Returns:
    - xr.Dataset: Dataset with float64 data variables
    '''
    with xr.open_dataset(filepath,engine='h5netcdf') as ds:
        ds = ds.load()
    return ds.assign({name:ds[name].astype(COMPUTEDTYPE) for name in ds.data_vars if ds[name].dtype.kind=='f'})

def save_dataset(ds,filepath,timechunksize=736):
    '''
    Purpose: Convert every floating-point data variable to STOREDTYPE, write the file, and verify by reopening.
        This is the only place files are written in the timing test.
    Args:
    - ds (xr.Dataset): Dataset to save
    - filepath (str): output path
    - timechunksize (int): chunk size along time
    '''
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    ds = ds.assign({name:ds[name].astype(STOREDTYPE) for name in ds.data_vars if ds[name].dtype.kind=='f'})
    for variable in ds.variables.values():
        variable.encoding = {}
    encoding = {name:{'chunksizes':tuple(min(timechunksize,size) if dim=='time' else size for dim,size in zip(da.dims,da.shape))}
                for name,da in ds.data_vars.items() if da.ndim>0}
    ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
    with xr.open_dataset(filepath,engine='h5netcdf') as check:
        wrong = {name:str(check[name].dtype) for name in check.data_vars if check[name].dtype.kind=='f' and check[name].dtype!=STOREDTYPE}
    if wrong:
        raise TypeError(f'{filepath} has non-{STOREDTYPE.__name__} variables: {wrong}')
    logger.info(f'      Saved {filepath}')

def load_stats(config):
    '''
    Purpose: Load training statistics for the configured splits directory.
    Args:
    - config (TimingConfig): configuration object
    Returns:
    - dict[str,float]: flat statistics dictionary
    '''
    with open(os.path.join(config.splitsdir,'stats.json'),'r',encoding='utf-8') as f:
        return json.load(f)

def standardize(values,stats,var):
    '''
    Purpose: Standardize a predictor with training statistics.
    Args:
    - values (np.ndarray): physical values
    - stats (dict): training statistics
    - var (str): variable name
    Returns:
    - np.ndarray: standardized values
    '''
    return (values-stats[f'{var}_mean'])/stats[f'{var}_std']

def precip_to_z(tp,stats):
    '''
    Purpose: Transform precipitation (mm) to the standardized log1p target.
    Args:
    - tp (np.ndarray): precipitation (mm)
    - stats (dict): training statistics
    Returns:
    - np.ndarray: standardized target
    '''
    return (np.log1p(tp)-stats['tp_mean'])/stats['tp_std']

def z_to_precip(z,stats):
    '''
    Purpose: Invert precip_to_z, clipping at zero.
    Args:
    - z (np.ndarray): standardized target values
    - stats (dict): training statistics
    Returns:
    - np.ndarray: precipitation (mm)
    '''
    return np.clip(np.expm1(z*stats['tp_std']+stats['tp_mean']),0.0,None)

def calc_zmin(stats):
    '''
    Purpose: Standardized value of zero precipitation.
    Args:
    - stats (dict): training statistics
    Returns:
    - float: zmin
    '''
    return (0.0-stats['tp_mean'])/stats['tp_std']

def restrict_kernel_seeds(config,kernelrun='nn_gauss'):
    '''
    Purpose: Limit config.nn['seeds'] to seeds with saved kernel weights, so SR features average only over
        trained kernels (e.g. when the NN was trained with --seeds).
    Args:
    - config (TimingConfig): configuration object (modified in place)
    - kernelrun (str): NN run whose kernel weights are used
    Returns:
    - list[int]: seeds with saved kernel weights (config unchanged if none)
    '''
    seeds = [seed for seed in config.nn['seeds'] if os.path.exists(os.path.join(config.weightsdir,f'{kernelrun}_{seed}_weights.nc'))]
    if seeds:
        config.nn['seeds'] = seeds
    return seeds

def calc_r2(truth,pred,mask=None):
    '''
    Purpose: Coefficient of determination over samples where both arrays are finite.
    Args:
    - truth (np.ndarray): true values
    - pred (np.ndarray): predicted values with the same shape
    - mask (np.ndarray | None): optional boolean mask with the same shape
    Returns:
    - float: R² (NaN if no valid samples)
    '''
    valid = np.isfinite(truth)&np.isfinite(pred)
    if mask is not None:
        valid = valid&mask
    if not valid.any():
        return np.nan
    ytrue,ypred = truth[valid],pred[valid]
    return float(1-np.sum((ypred-ytrue)**2)/np.sum((ytrue-ytrue.mean())**2))
