#!/usr/bin/env python

import numpy as np

class PODModel:

    def __init__(self,withlf,landthresh=0.5,alpha=None,blcrit=None,alphaland=None,blcritland=None,alphaocean=None,blcritocean=None):
        '''
        Purpose: Initialize a ramp-based POD model for precipitation prediction using Eq. 8 from Ahmed F., Adames A.F., &
        Neelin J.D. (2020), J. Atmos. Sci.
        Args:
        - withlf (bool): False for a single ramp fit, True for separate land/ocean ramp fits
        - landthresh (float): threshold for land/ocean classification (defaults to 0.5)
        - alpha (float): slope for single fit (optional, used when withlf=False)
        - blcrit (float): critical BL for single fit (optional, used when withlf=False)
        - alphaland (float): slope for land fit (optional, used when withlf=True)
        - blcritland (float): critical BL for land fit (optional, used when withlf=True)
        - alphaocean (float): slope for ocean fit (optional, used when withlf=True)
        - blcritocean (float): critical BL for ocean fit (optional, used when withlf=True)
        '''
        self.withlf       = bool(withlf)
        self.landthresh   = float(landthresh)
        self.alpha        = alpha
        self.blcrit       = blcrit
        self.alphaland    = alphaland
        self.blcritland   = blcritland
        self.alphaocean   = alphaocean
        self.blcritocean  = blcritocean
        self.nparams      = 4 if withlf else 2

    def forward(self,x,lf=None):
        '''
        Purpose: Forward pass through the POD ramp.
        Args:
        - x (xr.DataArray): input BL DataArray with dims (lat, lon, time)
        - lf (xr.DataArray): land fraction DataArray (required when withlf=True)
        Returns:
        - np.ndarray: predicted precipitation array of shape (x.size,)
        '''
        xflat  = x.values.ravel()
        ypred  = np.full(xflat.shape,np.nan,dtype=np.float32)
        finite = np.isfinite(xflat)
        if not self.withlf:
            ypred[finite] = self.alpha*np.maximum(0.0,xflat[finite]-self.blcrit).astype(np.float32)
        else:
            lfflat = lf.values.ravel()
            land   = (lfflat[finite]>=self.landthresh)
            ypredland     = self.alphaland*np.maximum(0.0,xflat[finite]-self.blcritland)
            ypredocean    = self.alphaocean*np.maximum(0.0,xflat[finite]-self.blcritocean)
            ypred[finite] = np.where(land,ypredland,ypredocean).astype(np.float32)
        return ypred
