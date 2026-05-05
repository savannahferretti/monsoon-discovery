#!/usr/bin/env python

import numpy as np

def fit_empirical_qm(a,b,nquantiles=200):
    '''
    Purpose: Fit an empirical quantile-mapping function from distribution `a` to distribution `b`. Only positive values are 
    used to build the quantile mapping; zeros and negative inputs map to zero.
    Args:
    - a (np.ndarray): finite 1D NumPy array of source distribution values (e.g., predicted ERA5 total precipitation in mm)
    - b (np.ndarray): finite 1D NumPy array of target distribution values (e.g., IMERG V06 precipitation rate in mm/hr)
    - nquantiles (int): number of quantile levels used to build the mapping (defaults to 200)
    Returns:
    - callable: function that maps distribution `a` to distribution `b`
    '''
    apos      = a[a>0]
    bpos      = b[b>0]
    quantiles = np.linspace(0,1,nquantiles)
    aq        = np.quantile(apos,quantiles) if len(apos)>0 else np.array([0.0])
    bq        = np.quantile(bpos,quantiles) if len(bpos)>0 else np.array([0.0])
    aq        = np.maximum.accumulate(aq)
    bq        = np.maximum.accumulate(bq)
    def qm(x):
        x       = np.maximum(np.asarray(x,dtype=float),0.0)
        out     = np.zeros_like(x)
        out[x>0] = np.interp(x[x>0],aq,bq,left=bq[0],right=bq[-1])
        return out
    return qm