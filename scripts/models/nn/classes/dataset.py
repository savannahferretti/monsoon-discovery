#!/usr/bin/env python

import os
import torch
import numpy as np
from scripts.utils import load,load_stats,standardize,flatten

class FieldDataset(torch.utils.data.Dataset):

    def __init__(self,fields,local,target,dsig=None):
        '''
        Purpose: Dataset of individual (time, lat, lon) samples for NN training and inference.
        Args:
        - fields (torch.Tensor): profiles with shape (nsamples, nfieldvars, nlevs)
        - local (torch.Tensor): local variables with shape (nsamples, nlocalvars)
        - target (torch.Tensor): standardized target with shape (nsamples,)
        - dsig (torch.Tensor | None): sigma thickness weights with shape (nlevs,), required for kernel models
        '''
        self.fields = fields
        self.local  = local
        self.target = target
        self.dsig   = dsig

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self,idx):
        batch = {
            'fields':self.fields[idx],
            'local':self.local[idx],
            'target':self.target[idx]}
        if self.dsig is not None:
            batch['dsig'] = self.dsig
        return batch

def load_split(splitname,fieldvars,localvars,splitsdir,targetvar='tp'):
    '''
    Purpose: Load a split, standardize it with training statistics, and return float32 tensors of the valid samples.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - fieldvars (list[str]): profile (or scalar field) variables
    - localvars (list[str]): local variables (e.g. ['lf','shf','lhf'])
    - splitsdir (str): directory containing the split files and stats.json
    - targetvar (str): target variable name (defaults to 'tp')
    Returns:
    - tuple: fields (nsamples, nfieldvars, nlevs), local (nsamples, nlocalvars), target (nsamples,), dsig (nlevs,),
        nlevs, boolean mask of valid samples over the full grid, and the target DataArray with (time, lat, lon)
        coordinates
    '''
    stats  = load_stats(splitsdir)
    ds     = load(os.path.join(splitsdir,f'{splitname}.h5'))
    ntime  = ds.sizes['time']
    target = standardize(flatten(ds[targetvar],ntime),targetvar,stats)
    if not fieldvars:
        nlevs  = 1
        fields = np.empty((target.size,0,1))
        dsig   = np.ones(1)
    elif 'sig' in ds[fieldvars[0]].dims:
        nlevs  = ds.sizes['sig']
        fields = np.stack([standardize(ds[var].transpose('time','lat','lon','sig').values.reshape(-1,nlevs),var,stats) for var in fieldvars],axis=1)
        dsig   = ds['dsig'].values
    else:
        nlevs  = 1
        fields = np.stack([standardize(flatten(ds[var],ntime),var,stats)[:,None] for var in fieldvars],axis=1)
        dsig   = np.ones(1)
    local = np.stack([standardize(flatten(ds[var],ntime),var,stats) for var in localvars],axis=1) if localvars else np.empty((target.size,0))
    valid = np.isfinite(fields).all(axis=(1,2))&np.isfinite(local).all(axis=1)&np.isfinite(target)
    refda = ds[targetvar].transpose('time','lat','lon')
    totensor = lambda arr:torch.from_numpy(np.ascontiguousarray(arr,dtype=np.float32))
    return totensor(fields[valid]),totensor(local[valid]),totensor(target[valid]),totensor(dsig),nlevs,valid,refda