#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr

class FieldDataset(torch.utils.data.Dataset):

    def __init__(self,fields,lf,target,dlev=None):
        '''
        Purpose: Dataset for NN training/inference at individual grid cell/timestep samples.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nsamp, nfieldvars, nlevs)
        - lf (torch.Tensor): land fraction with shape (nsamp,)
        - target (torch.Tensor): target precipitation with shape (nsamp,)
        - dlev (torch.Tensor | None): vertical thickness weights with shape (nlevs,), required for kernel models
        '''
        self.fields = fields
        self.lf     = lf
        self.target = target
        self.dlev   = dlev

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self,idx):
        batch = {
            'fields':self.fields[idx],
            'lf':self.lf[idx],
            'target':self.target[idx]}
        if self.dlev is not None:
            batch['dlev'] = self.dlev
        return batch


def load_split(splitname,fieldvars,splitsdir):
    '''
    Purpose: Load a normalized data split and return tensors shaped for NN training/inference.
        Each sample corresponds to a single (lat, lon, time) grid cell and timestep.
        For profile variables (with a lev dimension), the full vertical column at that grid cell
        is included. For scalar variables (without lev), each field is a single value per sample.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - fieldvars (list[str]): predictor field variable names from run config
    - splitsdir (str): directory containing normalized split HDF5 files
    Returns:
    - tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        (fields, lf, target, dlev, nlevs) where:
        - fields: (nsamp, nfieldvars, nlevs) predictor fields
        - lf: (nsamp,) land fraction
        - target: (nsamp,) normalized precipitation
        - dlev: (nlevs,) vertical thickness weights, or None for scalar inputs
        - nlevs: number of vertical levels (1 for scalar inputs)
    '''
    filepath = os.path.join(splitsdir,f'norm_{splitname}.h5')
    ds = xr.open_dataset(filepath,engine='h5netcdf')
    has_lev = 'lev' in ds[fieldvars[0]].dims
    ntime = ds.sizes['time']
    if has_lev:
        nlevs = ds.sizes['lev']
        field_arrays = []
        for v in fieldvars:
            da  = ds[v].transpose('time','lat','lon','lev')
            arr = da.values.reshape(-1,nlevs)
            field_arrays.append(arr)
        fields = np.stack(field_arrays,axis=1)
        dlev = torch.from_numpy(ds['dlev'].values.astype(np.float32))
    else:
        nlevs = 1
        field_arrays = []
        for v in fieldvars:
            da  = ds[v].transpose('time','lat','lon')
            arr = da.values.reshape(-1,1)
            field_arrays.append(arr)
        fields = np.stack(field_arrays,axis=1)
        dlev = None
    pr = ds['pr'].transpose('time','lat','lon').values.reshape(-1)
    lf_2d = ds['lf'].values
    lf = np.tile(lf_2d,(ntime,1,1)).reshape(-1)
    valid = np.isfinite(fields).all(axis=(1,2))&np.isfinite(lf)&np.isfinite(pr)
    fields = torch.from_numpy(fields[valid].astype(np.float32))
    lf     = torch.from_numpy(lf[valid].astype(np.float32))
    pr     = torch.from_numpy(pr[valid].astype(np.float32))
    return fields,lf,pr,dlev,nlevs
