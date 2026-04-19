#!/usr/bin/env python

import torch

class KernelModule:

    @staticmethod
    def normalize(kernel,dsig,epsilon=1e-6):
        '''
        Purpose: Normalize a 1D vertical kernel so that sum(k * dsig) = 1 for each field over the full column.
        Args:
        - kernel (torch.Tensor): unnormalized kernel with shape (nfieldvars, nlevs)
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
        Purpose: Integrate predictor fields using normalized kernel weights over the vertical dimension.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
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

    def get_weights(self,dsig,device):
        '''
        Purpose: Obtain normalized non-parametric kernel weights.
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
        Purpose: Apply non-parametric kernels to a batch of vertical profiles.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
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

    class MixtureGaussianKernel(torch.nn.Module):

        def __init__(self,nfieldvars):
            '''
            Purpose: Initialize a mixture-of-Gaussians kernel.
            Args:
            - nfieldvars (int): number of predictor fields
            '''
            super().__init__()
            self.mu1     = torch.nn.Parameter(torch.full((int(nfieldvars),),-0.5))
            self.mu2     = torch.nn.Parameter(torch.full((int(nfieldvars),), 0.5))
            self.logstd1 = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.logstd2 = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.weight1 = torch.nn.Parameter(torch.ones(int(nfieldvars)))
            self.weight2 = torch.nn.Parameter(torch.ones(int(nfieldvars)))

        def get_components(self,nlevs,device):
            '''
            Purpose: Compute the two Gaussian components separately (for visualization).
            Args:
            - nlevs (int): number of vertical levels
            - device (str | torch.device): device to use
            Returns:
            - tuple[torch.Tensor,torch.Tensor]: each component with shape (nfieldvars, nlevs)
            '''
            coord = torch.linspace(-1.0,1.0,steps=nlevs,device=device)
            std1  = torch.exp(self.logstd1).clamp(min=0.1,max=2.0)
            std2  = torch.exp(self.logstd2).clamp(min=0.1,max=2.0)
            c1 = self.weight1[:,None]*torch.exp(-(coord[None,:]-self.mu1[:,None])**2/(2*std1[:,None]**2))
            c2 = self.weight2[:,None]*torch.exp(-(coord[None,:]-self.mu2[:,None])**2/(2*std2[:,None]**2))
            return c1,c2

        def forward(self,nlevs,device):
            '''
            Purpose: Evaluate the mixture-of-Gaussians kernel over vertical levels.
            Args:
            - nlevs (int): number of vertical levels
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: mixture kernel values with shape (nfieldvars, nlevs)
            '''
            c1,c2 = self.get_components(nlevs,device)
            return c1+c2+1e-8

    kerneltypes = {'gaussian':GaussianKernel,'exponential':ExponentialKernel,'mixgaussian':MixtureGaussianKernel}

    def __init__(self,nfieldvars,kernelspec):
        '''
        Purpose: Initialize a parametric vertical kernel, optionally with per-field kernel types.
        Args:
        - nfieldvars (int): number of predictor fields
        - kernelspec (str | list[str]): a single kernel type for all fields, or a list of per-field
          kernel types; valid types are 'gaussian' | 'exponential' | 'mixgaussian'
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.kernelspec = kernelspec
        self.norm       = None
        self.features   = None
        self.perfield   = isinstance(kernelspec,list)
        if self.perfield:
            if len(kernelspec)!=self.nfieldvars:
                raise ValueError(f'Per-field kernel list must have length {self.nfieldvars}, got {len(kernelspec)}')
            for ktype in kernelspec:
                if ktype not in self.kerneltypes:
                    raise ValueError(f'Unknown kernel type `{ktype}`; must be one of {list(self.kerneltypes.keys())}')
            self.functions = torch.nn.ModuleList([self.kerneltypes[ktype](1) for ktype in kernelspec])
        else:
            if kernelspec not in self.kerneltypes:
                raise ValueError(f'Unknown kernel type `{kernelspec}`; must be one of {list(self.kerneltypes.keys())}')
            self.function = self.kerneltypes[kernelspec](self.nfieldvars)

    def get_weights(self,dsig,device):
        '''
        Purpose: Obtain normalized parametric kernel weights.
        Args:
        - dsig (torch.Tensor): sigma thickness weights with shape (nlevs,)
        - device (str | torch.device): device to use
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nlevs)
        '''
        dsig  = dsig.to(device)
        nlevs = dsig.numel()
        if self.perfield:
            kernel = torch.cat([f(nlevs,device) for f in self.functions],dim=0)
        else:
            kernel = self.function(nlevs,device)
        self.norm = KernelModule.normalize(kernel,dsig)
        return self.norm

    def forward(self,fields,dsig):
        '''
        Purpose: Apply parametric kernel to a batch of vertical profiles.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - dsig (torch.Tensor): sigma thickness weights with shape (nlevs,)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        norm  = self.get_weights(dsig,fields.device)
        feats = KernelModule.integrate(fields,norm,dsig)
        self.features = feats
        return feats
