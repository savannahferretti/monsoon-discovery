#!/usr/bin/env python

import torch
import torch.nn.functional as F

class MainNN(torch.nn.Module):

    def __init__(self,nfeatures,mean,std):
        '''
        Purpose: Initialize the feed-forward network shared by all models, which maps a feature vector to a standardized
        log1p precipitation prediction.
        Args:
        - nfeatures (int): number of input features per sample
        - mean (float): training mean of log1p(precipitation)
        - std (float): training standard deviation of log1p(precipitation)
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
        Purpose: Predict zmin + ReLU(f(X)), so that precipitation is never negative.
        Args:
        - X (torch.Tensor): input features with shape (nbatch, nfeatures)
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        return self.zmin + F.relu(self.layers(X).squeeze())

class BaselineNN(torch.nn.Module):

    def __init__(self,nfieldvars,nlevs,nlocalvars,mean,std):
        '''
        Purpose: Initialize a model that flattens the profiles and appends the local variables.
        Args:
        - nfieldvars (int): number of profile variables
        - nlevs (int): number of vertical levels (1 for scalar inputs such as bl)
        - nlocalvars (int): number of local variables
        - mean (float): training mean of log1p(precipitation)
        - std (float): training standard deviation of log1p(precipitation)
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.nlevs      = int(nlevs)
        self.nlocalvars = int(nlocalvars)
        nfeatures = self.nfieldvars*self.nlevs+self.nlocalvars
        self.model = MainNN(nfeatures,mean,std)

    def forward(self,fields,local):
        '''
        Purpose: Predict from flattened profiles and local variables.
        Args:
        - fields (torch.Tensor): profiles with shape (nbatch, nfieldvars, nlevs)
        - local (torch.Tensor): local variables with shape (nbatch, nlocalvars)
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        X = torch.cat([fields.flatten(1),local],dim=1)
        return self.model(X)

class KernelNN(torch.nn.Module):

    def __init__(self,kernel,nfieldvars,nlocalvars,mean,std):
        '''
        Purpose: Initialize a model that integrates each profile with a learned vertical kernel and appends the local
        variables.
        Args:
        - kernel (torch.nn.Module): NonparametricKernelLayer or ParametricKernelLayer
        - nfieldvars (int): number of profile variables
        - nlocalvars (int): number of local variables
        - mean (float): training mean of log1p(precipitation)
        - std (float): training standard deviation of log1p(precipitation)
        '''
        super().__init__()
        self.kernel     = kernel
        self.nfieldvars = int(nfieldvars)
        self.nlocalvars = int(nlocalvars)
        nfeatures = self.nfieldvars+self.nlocalvars
        self.model = MainNN(nfeatures,mean,std)

    def forward(self,fields,dsig,local):
        '''
        Purpose: Predict from kernel-integrated profiles and local variables.
        Args:
        - fields (torch.Tensor): profiles with shape (nbatch, nfieldvars, nlevs)
        - dsig (torch.Tensor): sigma thickness weights with shape (nlevs,)
        - local (torch.Tensor): local variables with shape (nbatch, nlocalvars)
        Returns:
        - torch.Tensor: predictions with shape (nbatch,)
        '''
        features = self.kernel(fields,dsig)
        X = torch.cat([features,local],dim=1)
        return self.model(X)
