#!/usr/bin/env python

import torch
import torch.nn.functional as F
from scripts.models.nn.kernels import NonparametricKernelLayer,ParametricKernelLayer

class LogCoshLoss(torch.nn.Module):
    def forward(self,output,target):
        return torch.log(torch.cosh(output-target)).mean()

class QuantileLoss(torch.nn.Module):
    def __init__(self,q=0.50):
        super().__init__()
        self.q = q
    def forward(self,output,target):
        err = target-output
        return torch.where(err>=0,self.q*err,(self.q-1)*err).mean()

class MainNN(torch.nn.Module):

    def __init__(self,nfeatures):
        '''
        Purpose: Initialize a feed-forward neural network that nonlinearly maps a feature vector to a scalar prediction.
        Args:
        - nfeatures (int): number of input features per sample
        '''
        super().__init__()
        nfeatures = int(nfeatures)
        self.register_buffer('prmean',torch.tensor(0.12733465433120728,dtype=torch.float32))
        self.register_buffer('prstd',torch.tensor(0.3446449339389801,dtype=torch.float32))
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
        return self.layers(X).squeeze().clamp(min=self.zmin)

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

class HurdleBaselineNN(torch.nn.Module):

    is_hurdle = True

    def __init__(self,nfieldvars,nlevs,nlocalvars,hasmask=False):
        '''
        Purpose: Initialize a hurdle neural network with a shared backbone and separate classification and regression heads.
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
        self.register_buffer('prmean',torch.tensor(0.12733465433120728,dtype=torch.float32))
        self.register_buffer('prstd',torch.tensor(0.3446449339389801,dtype=torch.float32))
        self.register_buffer('zmin',(torch.tensor(0.0)-torch.tensor(0.12733465433120728))/torch.tensor(0.3446449339389801))
        if hasmask:
            nfeatures = (self.nfieldvars+1)*self.nlevs+self.nlocalvars
        else:
            nfeatures = self.nfieldvars*self.nlevs+self.nlocalvars
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(nfeatures,256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256,128),       torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128,64),        torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64,32),         torch.nn.GELU(), torch.nn.Dropout(0.1))
        self.classifier = torch.nn.Linear(32,1)
        self.regressor  = torch.nn.Linear(32,1)

    def forward(self,fields,lf,mask=None):
        '''
        Purpose: Forward pass through HurdleBaselineNN.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - lf (torch.Tensor): local input variables with shape (nbatch, nlocalvars)
        - mask (torch.Tensor | None): surface validity mask with shape (nbatch, nlevs), used when hasmask=True
        Returns:
        - tuple[torch.Tensor, torch.Tensor]: (logit, amount) each with shape (nbatch,), where logit is the
            rain/no-rain classification logit and amount is the clamped normalized log precipitation
        '''
        if self.hasmask and mask is not None:
            fields = fields*mask.unsqueeze(1)
            X = torch.cat([fields,mask.unsqueeze(1)],dim=1)
            X = torch.cat([X.flatten(1),lf],dim=1)
        else:
            X = torch.cat([fields.flatten(1),lf],dim=1)
        h      = self.backbone(X)
        logit  = self.classifier(h).squeeze()
        amount = self.regressor(h).squeeze().clamp(min=self.zmin)
        return logit,amount

    def predict_expected(self,logit,amount):
        '''
        Purpose: Combine classifier and regressor outputs into a single expected-value prediction in normalized log-space, compatible with PredictionWriter denormalization.
        Args:
        - logit (torch.Tensor): rain/no-rain logit with shape (nbatch,)
        - amount (torch.Tensor): normalized log precipitation with shape (nbatch,), clamped at zmin
        Returns:
        - torch.Tensor: expected precipitation in normalized log-space with shape (nbatch,)
        '''
        prob        = torch.sigmoid(logit)
        amount_mm   = torch.expm1(amount*self.prstd+self.prmean)
        expected_mm = prob*amount_mm
        return (torch.log1p(expected_mm.clamp(min=0))-self.prmean)/self.prstd

class HurdleLoss(torch.nn.Module):

    def __init__(self,alpha=1.0):
        '''
        Purpose: Initialize a compound loss for hurdle models combining binary cross-entropy for rain/no-rain classification and MSE for precipitation amount on rainy samples.
        Args:
        - alpha (float): weight applied to the regression loss relative to the classification loss (defaults to 1.0)
        '''
        super().__init__()
        self.alpha = alpha
        self.bce   = torch.nn.BCEWithLogitsLoss()
        self.register_buffer('zmin',torch.tensor((0.0-0.12733465433120728)/0.3446449339389801,dtype=torch.float32))

    def forward(self,output,target):
        '''
        Purpose: Compute hurdle loss from classifier logit and regressor amount against normalized log precipitation target.
        Args:
        - output (tuple[torch.Tensor, torch.Tensor]): (logit, amount) from HurdleBaselineNN.forward()
        - target (torch.Tensor): normalized log precipitation with shape (nbatch,)
        Returns:
        - torch.Tensor: scalar combined loss
        '''
        logit,amount = output
        rain_mask    = (target>self.zmin).float()
        cls_loss     = self.bce(logit,rain_mask)
        rain_idx     = rain_mask.bool()
        if rain_idx.any():
            reg_loss = F.mse_loss(amount[rain_idx],target[rain_idx])
        else:
            reg_loss = torch.tensor(0.0,device=logit.device)
        return cls_loss+self.alpha*reg_loss