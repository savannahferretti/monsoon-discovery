#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr

class FieldDataset(torch.utils.data.Dataset):

    def __init__(self,fields,local,target,dsig=None):
        '''
        Purpose: Dataset for NN training/inference at individual grid cell/timestep samples.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nsamp, nfieldvars, nlevs)
        - local (torch.Tensor): local input variables with shape (nsamp, nlocalvars)
        - target (torch.Tensor): target precipitation with shape (nsamp,)
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

def load_split(splitname,fieldvars,localvars,splitsdir,targetvar='pr',subset=None):
    '''
    Purpose: Load a normalized data split and return tensors shaped for NN training/inference.
        Each sample corresponds to a single (lat, lon, time) grid cell and timestep.
        For profile variables (with a sig dimension), the full vertical column at that grid cell
        is included. For scalar variables (without sig), each field is a single value per sample.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - fieldvars (list[str]): predictor field variable names from run config
    - localvars (list[str]): local input variable names (e.g. ['lf','shf','lhf'])
    - splitsdir (str): directory containing normalized split HDF5 files
    - targetvar (str): target variable name ('pr' or 'tp') — must match run config
    - subset (dict | None): optional data subset filter with keys 'var', 'op', 'val'
        (e.g. {'var':'lf','op':'>=','val':0.5} for land-only samples)
    Returns:
    - tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, np.ndarray, xr.DataArray]:
        (fields, local, target, dsig, nlevs, valid, refda) where:
        - fields: (nsamp, nfieldvars, nlevs) predictor fields
        - local: (nsamp, nlocalvars) local input variables
        - target: (nsamp,) normalized precipitation
        - dsig: (nlevs,) sigma thickness weights (unit weight for scalar inputs)
        - nlevs: number of vertical levels (1 for scalar inputs)
        - valid: (nlat*nlon*ntime,) boolean array indicating which flattened samples were kept
        - refda: reference DataArray with (lat, lon, time) coordinates for reconstructing the grid
    '''
    filepath = os.path.join(splitsdir,f'norm_{splitname}.h5')
    ds = xr.open_dataset(filepath,engine='h5netcdf')
    hassig = 'sig' in ds[fieldvars[0]].dims
    ntime = ds.sizes['time']
    if hassig:
        nlevs = ds.sizes['sig']
        fieldarrays = []
        for v in fieldvars:
            da  = ds[v].transpose('time','lat','lon','sig')
            arr = da.values.reshape(-1,nlevs)
            fieldarrays.append(arr)
        fields = np.stack(fieldarrays,axis=1)
        dsig = torch.from_numpy(ds['dsig'].values.astype(np.float32))
    else:
        nlevs = 1
        fieldarrays = []
        for v in fieldvars:
            da  = ds[v].transpose('time','lat','lon')
            arr = da.values.reshape(-1,1)
            fieldarrays.append(arr)
        fields = np.stack(fieldarrays,axis=1)
        dsig = torch.tensor([1.0],dtype=torch.float32)
    pr = ds[targetvar].transpose('time','lat','lon').values.reshape(-1)
    ntotal = fields.shape[0]
    if localvars:
        localarrays = []
        for v in localvars:
            if 'time' in ds[v].dims:
                arr = ds[v].transpose('time','lat','lon').values.reshape(-1)
            else:
                arr = np.tile(ds[v].values,(ntime,1,1)).reshape(-1)
            localarrays.append(arr)
        local = np.stack(localarrays,axis=1)
    else:
        local = np.empty((ntotal,0),dtype=np.float32)
    valid = np.isfinite(fields).all(axis=(1,2))&np.isfinite(local).all(axis=1)&np.isfinite(pr)
    if subset:
        subvar = ds[subset['var']]
        if 'time' in subvar.dims:
            subarr = subvar.transpose('time','lat','lon').values.reshape(-1)
        else:
            subarr = np.tile(subvar.values,(ntime,1,1)).reshape(-1)
        if subset['op']=='>=':
            valid = valid&(subarr>=subset['val'])
        elif subset['op']=='<':
            valid = valid&(subarr<subset['val'])
    refda = ds[targetvar].transpose('time','lat','lon')
    fields = torch.from_numpy(fields[valid].astype(np.float32))
    local  = torch.from_numpy(local[valid].astype(np.float32))
    pr     = torch.from_numpy(pr[valid].astype(np.float32))
    return fields,local,pr,dsig,nlevs,valid,refda
