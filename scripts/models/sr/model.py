#!/usr/bin/env python

import numpy as np

class SRModel:

    def __init__(self,equation,feature_names):
        '''
        Purpose: Wrapper around a symbolic regression equation for precipitation prediction.
        Args:
        - equation (callable): callable that maps feature arrays to predictions
        - feature_names (list[str]): names of features the equation expects
        '''
        self.equation      = equation
        self.feature_names = list(feature_names)
        self.nparams       = None

    def forward(self,features):
        '''
        Purpose: Forward pass through the symbolic regression equation.
        Args:
        - features (dict[str,np.ndarray]): mapping of feature names to 1D arrays
        Returns:
        - np.ndarray: predicted precipitation array
        '''
        args = [features[name] for name in self.feature_names]
        return self.equation(*args)
