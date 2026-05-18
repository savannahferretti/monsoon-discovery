#!/usr/bin/env python

import numpy as np

class EmpiricalRampPOD:

    def __init__(self,alpha=None,xcrit=None,bincenters=None,ymeans=None):
        self.alpha      = alpha
        self.xcrit      = xcrit
        self.bincenters = bincenters
        self.ymeans     = ymeans
        self.nparams    = 2

    def forward(self,x):
        xflat      = x.values.ravel()
        ypred      = np.full(xflat.shape,np.nan,dtype=np.float32)
        finite_idx = np.where(np.isfinite(xflat))[0]
        xf         = xflat[finite_idx]
        lo,hi      = self.bincenters[0],self.bincenters[-1]
        dense_mask = (xf>=lo)&(xf<=hi)
        if dense_mask.any():
            ypred[finite_idx[dense_mask]] = np.interp(
                xf[dense_mask],self.bincenters,self.ymeans).astype(np.float32)
        out_mask = ~dense_mask
        if out_mask.any():
            ypred[finite_idx[out_mask]] = (
                self.alpha*np.maximum(0.0,xf[out_mask]-self.xcrit)).astype(np.float32)
        return ypred

class RampPOD:

    def __init__(self,alpha=None,xcrit=None):
        '''
        Purpose: Initialize a ramp-based POD for precipitation prediction using Eq. 8 from Ahmed F., Adames A.F., &
        Neelin J.D. (2020), J. Atmos. Sci.
        Args:
        - alpha (float): slope of the ramp fit
        - xcrit (float): critical BL value at which the ramp activates
        '''
        self.alpha   = alpha
        self.xcrit   = xcrit
        self.nparams = 2

    def forward(self,x):
        '''
        Purpose: Forward pass through the ramp function.
        Args:
        - x (xr.DataArray): input BL DataArray with dims (lat, lon, time)
        Returns:
        - np.ndarray: predicted precipitation array of shape (x.size,)
        '''
        xflat  = x.values.ravel()
        ypred  = np.full(xflat.shape,np.nan,dtype=np.float32)
        finite = np.isfinite(xflat)
        ypred[finite] = self.alpha*np.maximum(0.0,xflat[finite]-self.xcrit).astype(np.float32)
        return ypred
