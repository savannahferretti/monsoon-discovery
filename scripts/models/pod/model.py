#!/usr/bin/env python

import numpy as np

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
