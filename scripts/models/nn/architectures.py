#!/usr/bin/env python

import torch
import torch.nn.functional as F
from scripts.models.nn.kernels import NonparametricKernelLayer,ParametricKernelLayer

class MainNN(torch.nn.Module):

    def __init__(self,nfeatures):
        '''
        Purpose: Initialize a feed-forward neural network that nonlinearly maps a feature vector to a scalar prediction.
        Args:
        - nfeatures (int): number of input features per sample
        '''
        super().__init__()
        nfeatures = int(nfeatures)
        self.register_buffer('prmean',torch.tensor(0.1292700618505478,dtype=torch.float32))
        self.register_buffer('prstd',torch.tensor(0.343968003988266,dtype=torch.float32))
        self.register_buffer('zmin',(0.0-self.prmean)/self.prstd)
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
        z = self.layers(X).squeeze()
        return z.clamp(min=self.zmin)
        # return self.layers(X).squeeze()
        

class BaselineNN(torch.nn.Module):

    def __init__(self,nfieldvars,nlevs,nlocalvars,hasmask=False):
        '''
        Purpose: Initialize a baseline neural network that flattens vertical profiles and concatenates local variables.
        Args:
        - nfieldvars (int): number of predictor field variables
        - nlevs (int): number of vertical levels (1 for scalar inputs like bl, cape, subsat)
        - nlocalvars (int): number of local input variables (e.g. land fraction, heat fluxes)
        - hasmask (bool): whether to include surface validity mask as an extra input channel (defaults to False)
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
        self.model = MainNN(nfeatures)

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

    def __init__(self,kernel,nfieldvars,nlocalvars):
        '''
        Purpose: Initialize a kernel-based neural network that integrates over the vertical dimension.
        Args:
        - kernel (torch.nn.Module): instance of NonparametricKernelLayer or ParametricKernelLayer
        - nfieldvars (int): number of predictor field variables
        - nlocalvars (int): number of local input variables (e.g. land fraction, heat fluxes)
        '''
        super().__init__()
        self.kernel     = kernel
        self.nfieldvars = int(nfieldvars)
        self.nlocalvars = int(nlocalvars)
        nfeatures = self.nfieldvars+self.nlocalvars
        self.model = MainNN(nfeatures)

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

class WeightedMSELoss(torch.nn.Module):

    def forward(self,pred,target):
        '''
        Purpose: MSE loss with per-sample weights that emphasize higher-precipitation events.
        Targets are in standardized log1p space, so relu(target) selects above-average wet events
        and assigns them linearly increasing weight. Dry samples retain weight 1.
        '''
        weight = 1.0 + F.relu(target.detach())
        return (weight * (pred - target) ** 2).mean()

class LogCoshLoss(torch.nn.Module):

    def forward(self,pred,target):
        '''
        Purpose: Log-cosh loss. Behaves like MSE for small errors and MAE for large ones,
        smoothly downweighting the influence of extreme outliers relative to MSE.
        '''
        return torch.log(torch.cosh(pred - target)).mean()

class QuantileLoss(torch.nn.Module):

    def __init__(self,q=0.5):
        '''
        Purpose: Pinball (quantile) loss that targets the q-th conditional quantile.
        Args:
        - q (float): quantile to target, in (0, 1). Defaults to 0.5 (median = MAE).
                     Use q > 0.5 to penalise under-prediction more (e.g. missing rain events).
        '''
        super().__init__()
        self.q = q

    def forward(self,pred,target):
        err = target - pred
        return torch.where(err >= 0, self.q * err, (self.q - 1) * err).mean()

class DiceLoss(torch.nn.Module):

    def forward(self,pred,target):
        '''
        Purpose: Soft Dice loss adapted for continuous regression targets.
        Applies sigmoid to map standardized predictions and targets into (0, 1), then
        computes 1 - Dice coefficient. Encourages spatial overlap of high-precipitation
        predictions with observed events rather than pointwise accuracy.
        '''
        p = torch.sigmoid(pred)
        t = torch.sigmoid(target)
        intersection = (p * t).sum()
        return 1.0 - (2.0 * intersection) / (p.sum() + t.sum() + 1e-8)