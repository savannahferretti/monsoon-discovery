#!/usr/bin/env python

import torch

class KernelModule:

    @staticmethod
    def normalize(kernel,dlev,epsilon=1e-6):
        '''
        Purpose: Normalize a 1D vertical kernel so that sum(k * dlev) = 1 for each field over the full column.
        Args:
        - kernel (torch.Tensor): unnormalized kernel with shape (nfieldvars, nlevs)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - epsilon (float): stabilizer to avoid divide-by-zero (defaults to 1e-6)
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nlevs)
        '''
        kernelsum = (kernel*dlev.unsqueeze(0)).sum(dim=1)
        weights   = kernel/(kernelsum.unsqueeze(1)+epsilon)
        checksum  = (weights*dlev.unsqueeze(0)).sum(dim=1)
        assert torch.allclose(checksum,torch.ones_like(checksum),atol=1e-2),f'Kernel normalization failed, weights sum to {checksum.mean().item():.6f} instead of 1.0'
        return weights

    @staticmethod
    def integrate(fields,weights,dlev,mask=None):
        '''
        Purpose: Integrate predictor fields using normalized kernel weights over the vertical dimension.
            When a surface validity mask is provided, sub-surface levels are zeroed out so the effective
            integral is less than 1 at high-elevation grid cells.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - weights (torch.Tensor): normalized kernel weights with shape (nfieldvars, nlevs)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - mask (torch.Tensor | None): surface validity mask with shape (nbatch, nlevs) or None
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        weighted = fields*weights.unsqueeze(0)*dlev.unsqueeze(0).unsqueeze(0)
        if mask is not None:
            weighted = weighted*mask.unsqueeze(1)
        return weighted.sum(dim=2)


class NonparametricKernelLayer(torch.nn.Module):

    def __init__(self,nfieldvars,nlevs):
        '''
        Purpose: Initialize free-form (non-parametric) vertical kernels.
        Args:
        - nfieldvars (int): number of predictor fields
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

    def get_weights(self,dlev,device):
        '''
        Purpose: Obtain normalized non-parametric kernel weights.
        Args:
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nlevs)
        '''
        dlev      = dlev.to(device)
        self.raw  = self.raw.to(device)
        self.norm = KernelModule.normalize(self.raw,dlev)
        return self.norm

    def forward(self,fields,dlev,mask=None):
        '''
        Purpose: Apply non-parametric kernels to a batch of vertical profiles.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - mask (torch.Tensor | None): surface validity mask with shape (nbatch, nlevs), or None
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        norm  = self.get_weights(dlev,fields.device)
        feats = KernelModule.integrate(fields,norm,dlev,mask=mask)
        self.features = feats
        return feats


class ParametricKernelLayer(torch.nn.Module):

    class GaussianKernel(torch.nn.Module):

        def __init__(self,nfieldvars):
            '''
            Purpose: Initialize parameters of a 1D Gaussian kernel.
            Args:
            - nfieldvars (int): number of predictor fields
            '''
            super().__init__()
            self.mu     = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.logstd = torch.nn.Parameter(torch.zeros(int(nfieldvars)))

        def forward(self,nlevs,device):
            '''
            Purpose: Evaluate the Gaussian kernel over vertical levels.
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

    class ExponentialKernel(torch.nn.Module):

        def __init__(self,nfieldvars):
            '''
            Purpose: Initialize parameters of an exponential-decay kernel.
            Args:
            - nfieldvars (int): number of predictor fields
            '''
            super().__init__()
            self.logtau     = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.logitalpha = torch.nn.Parameter(torch.zeros(int(nfieldvars)))

        def forward(self,nlevs,device):
            '''
            Purpose: Evaluate the exponential-decay kernel over vertical levels.
            Args:
            - nlevs (int): number of vertical levels
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: exponential kernel values with shape (nfieldvars, nlevs)
            '''
            tau      = torch.exp(self.logtau).clamp(min=1e-4,max=100.0)
            alpha    = torch.sigmoid(self.logitalpha)
            j        = torch.arange(nlevs,device=device,dtype=torch.float32)
            distance = (1.0-alpha[:,None])*j[None,:]+(alpha[:,None])*(nlevs-1-j[None,:])
            kernel1D = torch.exp(-distance/tau[:,None])
            return kernel1D

    kerneltypes = {'gaussian':GaussianKernel,'exponential':ExponentialKernel}

    def __init__(self,nfieldvars,kerneltype):
        '''
        Purpose: Initialize a parametric vertical kernel.
        Args:
        - nfieldvars (int): number of predictor fields
        - kerneltype (str): 'gaussian' | 'exponential'
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.kerneltype = str(kerneltype)
        self.norm       = None
        self.features   = None
        if kerneltype not in self.kerneltypes:
            raise ValueError(f'Unknown kernel type `{kerneltype}`; must be one of {list(self.kerneltypes.keys())}')
        self.function = self.kerneltypes[kerneltype](self.nfieldvars)

    def get_weights(self,dlev,device):
        '''
        Purpose: Obtain normalized parametric kernel weights.
        Args:
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nlevs)
        '''
        dlev   = dlev.to(device)
        nlevs  = dlev.numel()
        kernel = self.function(nlevs,device)
        self.norm = KernelModule.normalize(kernel,dlev)
        return self.norm

    def forward(self,fields,dlev,mask=None):
        '''
        Purpose: Apply parametric kernel to a batch of vertical profiles.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - mask (torch.Tensor | None): surface validity mask with shape (nbatch, nlevs), or None
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        norm  = self.get_weights(dlev,fields.device)
        feats = KernelModule.integrate(fields,norm,dlev,mask=mask)
        self.features = feats
        return feats