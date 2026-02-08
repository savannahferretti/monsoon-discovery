#!/usr/bin/env python

import json
import numpy as np

class PODModel:
    
    def __init__(self,mode,alphapooled=None,blcritpooled=None,alphaland=None,blcritland=None,alphaocean=None,blcritocean=None):
        '''
        Purpose: Initialize a ramp-based POD model for precipitation prediction using Eq. 8 from Ahmed F., Adames Á.F., & 
        Neelin J.D. (2020), J. Atmos. Sci.
        Args: 
        - mode (str): 'pooled' (single ramp) | 'regional' (separate land/ocean ramps)
        - alphapooled (float): slope for pooled mode (optional)
        - blcritpooled (float): critical BL for pooled mode (optional)
        - alphaland (float): slope for land in regional mode (optional)
        - blcritland (float): critical BL for land in regional mode (optional)
        - alphaocean (float): slope for ocean in regional mode (optional)
        - blcritocean (float): critical BL for ocean in regional mode (optional)
        Returns:
        - None
        '''
        with open('configs.json','r',encoding='utf8') as f:
            configs = json.load(f)
        self.mode         = str(mode)
        self.landthresh   = float(configs['dataparams']['landthresh'])
        self.alphapooled  = alphapooled
        self.blcritpooled = blcritpooled
        self.alphaland    = alphaland
        self.blcritland   = blcritland
        self.alphaocean   = alphaocean
        self.blcritocean  = blcritocean
        self.nparams      = 2 if mode=='pooled' else 4

    def forward(self,x,lf=None):
        '''
        Purpose: Forward pass through the POD ramp. 
        Args:
        - x (xr.DataArray): input 3D BL DataArray
        - lf (xr.DataArray): land fraction for routing in 'regional' mode (same shape as 'x')
        Returns:
        - np.ndarray: predicted precipitation array of shape (x.size,)
        '''
        xflat  = x.values.ravel()
        ypred  = np.full(xflat.shape,np.nan,dtype=np.float32)
        finite = np.isfinite(xflat)
        if self.mode=='pooled':
            ypred[finite] = self.alphapooled*np.maximum(0.0,xflat[finite]-self.blcritpooled).astype(np.float32)
        elif self.mode=='regional':
            lfflat = lf.values.ravel()
            land   = (lfflat[finite]>=self.landthresh)
            ypredland     = self.alphaland*np.maximum(0.0,xflat[finite]-self.blcritland)
            ypredocean    = self.alphaocean*np.maximum(0.0,xflat[finite]-self.blcritocean)
            ypred[finite] = np.where(land,ypredland,ypredocean).astype(np.float32)
        return ypred