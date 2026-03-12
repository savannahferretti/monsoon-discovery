#!/usr/bin/env python

import torch
import torch.nn.functional as F
from scripts.models.nn.kernels import NonparametricKernelLayer,ParametricKernelLayer

TARGETSTATS = {
    'pr':{'mean':0.11673472821712494,'std':0.34830141067504883},
    'tp':{'mean':0.33761656284332275,'std':0.5284095406532288}}

class QuantileLoss(torch.nn.Module):
    def __init__(self,q=0.75):
        super().__init__()
        assert 0 <q<1,'Quantile q must satisfy 0 < q < 1'
        self.q = q
    def forward(self,output,target):
        err  = target-output
        loss = torch.maximum(self.q*err,(self.q-1)*err)
        return loss.mean()

class TweedieLoss(torch.nn.Module):
    def __init__(self,p=1.5,mean=TARGETSTATS['pr']['mean'],std=TARGETSTATS['pr']['std']):
        super().__init__()
        assert 1<p<2,'Tweedie power p must satisfy 1 < p < 2'
        self.p = p
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.tensor(std,dtype=torch.float32))
    def forward(self,output,target):
        ypred = torch.expm1(output*self.std+self.mean).clamp(min=1e-6)
        ytrue = torch.expm1(target*self.std+self.mean).clamp(min=0)
        return (ypred.pow(2-self.p)/(2-self.p)-ytrue*ypred.pow(1-self.p)/(1-self.p)).mean()

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

    def __init__(self,nfieldvars,nlevs,nlocalvars,hasmask=False,mean=TARGETSTATS['pr']['mean'],std=TARGETSTATS['pr']['std']):
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

    def __init__(self,kernel,nfieldvars,nlocalvars,mean=TARGETSTATS['pr']['mean'],std=TARGETSTATS['pr']['std']):
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