#!/usr/bin/env python

import torch

class KernelModule:

    @staticmethod
    def normalize(kernel,dsig,epsilon=1e-6):
        '''
        Purpose: Scale each kernel so that sum(k·Δσ) = 1 over the column.
        Args:
        - kernel (torch.Tensor): unnormalized kernels with shape (nfieldvars, nlevs)
        - dsig (torch.Tensor): sigma thickness weights with shape (nlevs,)
        - epsilon (float): stabilizer to avoid divide-by-zero (defaults to 1e-6)
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nlevs)
        '''
        kernelsum = (kernel*dsig.unsqueeze(0)).sum(dim=1)
        weights   = kernel/(kernelsum.unsqueeze(1)+epsilon)
        checksum  = (weights*dsig.unsqueeze(0)).sum(dim=1)
        assert torch.allclose(checksum,torch.ones_like(checksum),atol=1e-2),f'Kernel normalization failed, weights sum to {checksum.mean().item():.6f} instead of 1.0'
        return weights

    @staticmethod
    def integrate(fields,weights,dsig):
        '''
        Purpose: Integrate each profile over the column with its kernel, sum(k·φ·Δσ).
        Args:
        - fields (torch.Tensor): profiles with shape (nbatch, nfieldvars, nlevs)
        - weights (torch.Tensor): normalized kernel weights with shape (nfieldvars, nlevs)
        - dsig (torch.Tensor): sigma thickness weights with shape (nlevs,)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        weighted = fields*weights.unsqueeze(0)*dsig.unsqueeze(0).unsqueeze(0)
        return weighted.sum(dim=2)

class NonparametricKernelLayer(torch.nn.Module):

    def __init__(self,nfieldvars,nlevs):
        '''
        Purpose: Initialize nonparametric kernels, with a free weight at each level.
        Args:
        - nfieldvars (int): number of profile variables
        - nlevs (int): number of vertical levels
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.nlevs      = int(nlevs)
        self.norm       = None
        self.features   = None
        raw = torch.ones(self.nfieldvars,self.nlevs)
        raw = raw+torch.randn_like(raw)*0.2
        self.raw = torch.nn.Parameter(raw)

    def get_weights(self,dsig,device):
        '''
        Purpose: Return the normalized nonparametric kernel weights (also stored in self.norm).
        Args:
        - dsig (torch.Tensor): sigma thickness weights with shape (nlevs,)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nlevs)
        '''
        dsig      = dsig.to(device)
        self.raw  = self.raw.to(device)
        self.norm = KernelModule.normalize(self.raw,dsig)
        return self.norm

    def forward(self,fields,dsig):
        '''
        Purpose: Integrate a batch of profiles with the nonparametric kernels.
        Args:
        - fields (torch.Tensor): profiles with shape (nbatch, nfieldvars, nlevs)
        - dsig (torch.Tensor): sigma thickness weights with shape (nlevs,)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        norm  = self.get_weights(dsig,fields.device)
        feats = KernelModule.integrate(fields,norm,dsig)
        self.features = feats
        return feats

class ParametricKernelLayer(torch.nn.Module):

    class GaussianKernel(torch.nn.Module):

        def __init__(self,nfieldvars):
            '''
            Purpose: Initialize the center and log-width of a Gaussian kernel for each profile variable.
            Args:
            - nfieldvars (int): number of profile variables
            '''
            super().__init__()
            self.mu     = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.logstd = torch.nn.Parameter(torch.zeros(int(nfieldvars)))

        def forward(self,nlevs,device):
            '''
            Purpose: Evaluate the Gaussian kernels on a coordinate running from -1 to 1 across the levels.
            Args:
            - nlevs (int): number of vertical levels
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: Gaussian kernel values with shape (nfieldvars, nlevs)
            '''
            coord    = torch.linspace(-1.0,1.0,steps=nlevs,device=device)
            std      = torch.exp(self.logstd)
            kernel1D = torch.exp(-0.5*((coord[None,:]-self.mu[:,None])/std[:,None])**2)
            return kernel1D

    kerneltypes = {'gaussian':GaussianKernel}

    def __init__(self,nfieldvars,kernelspec):
        '''
        Purpose: Initialize parametric kernels of a given shape.
        Args:
        - nfieldvars (int): number of profile variables
        - kernelspec (str): kernel shape ('gaussian')
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.norm       = None
        self.features   = None
        if kernelspec not in self.kerneltypes:
            raise ValueError(f'Unknown kernel type `{kernelspec}`; must be one of {list(self.kerneltypes.keys())}')
        self.function = self.kerneltypes[kernelspec](self.nfieldvars)

    def get_weights(self,dsig,device):
        '''
        Purpose: Return the normalized parametric kernel weights (also stored in self.norm).
        Args:
        - dsig (torch.Tensor): sigma thickness weights with shape (nlevs,)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nlevs)
        '''
        dsig  = dsig.to(device)
        nlevs = dsig.numel()
        kernel = self.function(nlevs,device)
        self.norm = KernelModule.normalize(kernel,dsig)
        return self.norm

    def forward(self,fields,dsig):
        '''
        Purpose: Integrate a batch of profiles with the parametric kernels.
        Args:
        - fields (torch.Tensor): profiles with shape (nbatch, nfieldvars, nlevs)
        - dsig (torch.Tensor): sigma thickness weights with shape (nlevs,)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        norm  = self.get_weights(dsig,fields.device)
        feats = KernelModule.integrate(fields,norm,dsig)
        self.features = feats
        return feats
