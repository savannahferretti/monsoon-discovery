#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr

class FieldDataset(torch.utils.data.Dataset):

    def __init__(self,fields,lf,target,dlev=None,mask=None):
        '''
        Purpose: Dataset for NN training/inference at individual grid cell/timestep samples.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nsamp, nfieldvars, nlevs)
        - lf (torch.Tensor): land fraction with shape (nsamp,)
        - target (torch.Tensor): target precipitation with shape (nsamp,)
        - dlev (torch.Tensor | None): vertical thickness weights with shape (nlevs,), required for kernel models
        - mask (torch.Tensor | None): surface validity mask with shape (nsamp, nlevs), required for profile inputs
        '''
        self.fields = fields
        self.lf     = lf
        self.target = target
        self.dlev   = dlev
        self.mask   = mask

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self,idx):
        batch = {
            'fields':self.fields[idx],
            'lf':self.lf[idx],
            'target':self.target[idx]}
        if self.dlev is not None:
            batch['dlev'] = self.dlev
        if self.mask is not None:
            batch['mask'] = self.mask[idx]
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
    - tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None, np.ndarray, xr.DataArray]:
        (fields, lf, target, dlev, nlevs, mask, valid, refda) where:
        - fields: (nsamp, nfieldvars, nlevs) predictor fields
        - lf: (nsamp,) land fraction
        - target: (nsamp,) normalized precipitation
        - dlev: (nlevs,) vertical thickness weights (unit weight for scalar inputs)
        - nlevs: number of vertical levels (1 for scalar inputs)
        - mask: (nsamp, nlevs) surface validity mask, or None for scalar inputs
        - valid: (nlat*nlon*ntime,) boolean array indicating which flattened samples were kept
        - refda: reference DataArray with (lat, lon, time) coordinates for reconstructing the grid
    '''
    filepath = os.path.join(splitsdir,f'norm_{splitname}.h5')
    ds = xr.open_dataset(filepath,engine='h5netcdf')
    haslev = 'lev' in ds[fieldvars[0]].dims
    ntime = ds.sizes['time']
    if haslev:
        nlevs = ds.sizes['lev']
        fieldarrays = []
        for v in fieldvars:
            da  = ds[v].transpose('time','lat','lon','lev')
            arr = da.values.reshape(-1,nlevs)
            fieldarrays.append(arr)
        fields = np.stack(fieldarrays,axis=1)
        dlev = torch.from_numpy(ds['dlev'].values.astype(np.float32))
        maskarr = ds['surfmask'].transpose('time','lat','lon','lev').values.reshape(-1,nlevs)
    else:
        nlevs = 1
        fieldarrays = []
        for v in fieldvars:
            da  = ds[v].transpose('time','lat','lon')
            arr = da.values.reshape(-1,1)
            fieldarrays.append(arr)
        fields = np.stack(fieldarrays,axis=1)
        dlev = torch.tensor([1.0],dtype=torch.float32)
        maskarr = None
    pr = ds['pr'].transpose('time','lat','lon').values.reshape(-1)
    lf2d = ds['lf'].values
    lf = np.tile(lf2d,(ntime,1,1)).reshape(-1)
    valid = np.isfinite(fields).all(axis=(1,2))&np.isfinite(lf)&np.isfinite(pr)
    refda = ds['pr'].transpose('time','lat','lon')
    fields = torch.from_numpy(fields[valid].astype(np.float32))
    lf     = torch.from_numpy(lf[valid].astype(np.float32))
    pr     = torch.from_numpy(pr[valid].astype(np.float32))
    if maskarr is not None:
        mask = torch.from_numpy(maskarr[valid].astype(np.float32))
    else:
        mask = None
    return fields,lf,pr,dlev,nlevs,mask,valid,refda
