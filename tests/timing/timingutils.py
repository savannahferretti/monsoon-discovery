#!/usr/bin/env python

import os
import sys
import json
import numpy as np

TIMINGDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR   = os.path.dirname(os.path.dirname(TIMINGDIR))
if REPODIR not in sys.path:
    sys.path.insert(0,REPODIR)

from scripts.utils import Config

OUTPUTDIRS = ('interim','splits','predictions','features','weights','models')

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
        return {name:self.sr['runs'][name] for name in self.timing['sr']['runs']}

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
