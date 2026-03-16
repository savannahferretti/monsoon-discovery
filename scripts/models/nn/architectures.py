#!/usr/bin/env python

import torch
import torch.nn.functional as F
from scripts.models.nn.kernels import NonparametricKernelLayer,ParametricKernelLayer

TARGETSTATS = {'tp':{'mean':0.3376164138317108,'std':0.5284096002578735}}

class MainNN(torch.nn.Module):

    def __init__(self,nfeatures,mean,std):
        '''
        Purpose: Initialize a feed-forward neural network that nonlinearly maps a feature vector to a scalar prediction.
        Args:
        - nfeatures (int): number of input features per sample
        - mean (float): target variable log1p mean (from training stats)
        - std (float): target variable log1p std (from training stats)
        '''
        super().__init__()
        nfeatures = int(nfeatures)
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.tensor(std,dtype=torch.float32))
        self.register_buffer('zmin',torch.tensor((0.0-mean)/std,dtype=torch.float32))
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(nfeatures,256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256,128),       torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128,64),        torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64,32),         torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(32,1))

    def forward(self,X):
        '''
        Purpose: Forward pass through MainNN.
        Args:
        - X (torch.Tensor): input features with shape (nbatch, nfeatures)
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        return self.layers(X).squeeze().clamp(min=self.zmin)

class BaselineNN(torch.nn.Module):

    def __init__(self,nfieldvars,nlevs,nlocalvars,hasmask=False,mean=TARGETSTATS['tp']['mean'],std=TARGETSTATS['tp']['std']):
        '''
        Purpose: Initialize a baseline neural network that flattens vertical profiles and concatenates local variables.
        Args:
        - nfieldvars (int): number of predictor field variables
        - nlevs (int): number of vertical levels (1 for scalar inputs like bl, cape, subsat)
        - nlocalvars (int): number of local input variables (e.g. land fraction, heat fluxes)
        - hasmask (bool): whether to include surface validity mask as an extra input channel (defaults to False)
        - mean (float): target variable log1p mean (from training stats)
        - std (float): target variable log1p std (from training stats)
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.nlevs      = int(nlevs)
        self.nlocalvars = int(nlocalvars)
        self.hasmask    = bool(hasmask)
        if hasmask:
            nfeatures = (self.nfieldvars+1)*self.nlevs+self.nlocalvars
        else:
            nfeatures = self.nfieldvars*self.nlevs+self.nlocalvars
        self.model = MainNN(nfeatures,mean,std)

    def forward(self,fields,lf,mask=None):
        '''
        Purpose: Forward pass through BaselineNN.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - lf (torch.Tensor): local input variables with shape (nbatch, nlocalvars)
        - mask (torch.Tensor | None): surface validity mask with shape (nbatch, nlevs), used when hasmask=True
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        if self.hasmask and mask is not None:
            fields = fields*mask.unsqueeze(1)
            X = torch.cat([fields,mask.unsqueeze(1)],dim=1)
            X = torch.cat([X.flatten(1),lf],dim=1)
        else:
            X = torch.cat([fields.flatten(1),lf],dim=1)
        return self.model(X)

class KernelNN(torch.nn.Module):

    def __init__(self,kernel,nfieldvars,nlocalvars,mean=TARGETSTATS['tp']['mean'],std=TARGETSTATS['tp']['std']):
        '''
        Purpose: Initialize a kernel-based neural network that integrates over the vertical dimension.
        Args:
        - kernel (torch.nn.Module): instance of NonparametricKernelLayer or ParametricKernelLayer
        - nfieldvars (int): number of predictor field variables
        - nlocalvars (int): number of local input variables (e.g. land fraction, heat fluxes)
        - mean (float): target variable log1p mean (from training stats)
        - std (float): target variable log1p std (from training stats)
        '''
        super().__init__()
        self.kernel     = kernel
        self.nfieldvars = int(nfieldvars)
        self.nlocalvars = int(nlocalvars)
        nfeatures = self.nfieldvars+self.nlocalvars
        self.model = MainNN(nfeatures,mean,std)
    
    def forward(self,fields,dlev,lf,mask=None):
        '''
        Purpose: Forward pass through KernelNN.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - lf (torch.Tensor): local input variables with shape (nbatch, nlocalvars)
        - mask (torch.Tensor | None): surface validity mask with shape (nbatch, nlevs), used during integration
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        features = self.kernel(fields,dlev,mask=mask)
        X = torch.cat([features,lf],dim=1)
        return self.model(X)