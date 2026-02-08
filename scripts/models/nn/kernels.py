#!/usr/bin/env python

import torch

class KernelModule:

    @staticmethod
    def normalize(kernel,dlev,epsilon=1e-6):
        '''
        Purpose: Normalize a 1D vertical kernel so that sum(k * dlev) = 1 for each field.
        Args:
        - kernel (torch.Tensor): unnormalized kernel with shape (nfieldvars, nlevs)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - epsilon (float): stabilizer to avoid divide-by-zero (defaults to 1e-6)
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nlevs)
        '''
        kernelsum = (kernel*dlev.unsqueeze(0)).sum(dim=1)
        weights = kernel/(kernelsum.unsqueeze(1)+epsilon)
        checksum = (weights*dlev.unsqueeze(0)).sum(dim=1)
        assert torch.allclose(checksum,torch.ones_like(checksum),atol=1e-2),f'Kernel normalization failed, weights sum to {checksum.mean().item():.6f} instead of 1.0'
        return weights

    @staticmethod
    def integrate(fields,weights,dlev):
        '''
        Purpose: Integrate predictor fields using normalized kernel weights over the vertical dimension.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - weights (torch.Tensor): normalized kernel weights with shape (nfieldvars, nlevs)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        weighted = fields*weights.unsqueeze(0)*dlev.unsqueeze(0).unsqueeze(0)
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
        self.raw = self.raw.to(device)
        dlev     = dlev.to(device)
        self.norm = KernelModule.normalize(self.raw,dlev)
        return self.norm

    def forward(self,fields,dlev):
        '''
        Purpose: Apply non-parametric kernels to a batch of vertical profiles.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        norm = self.get_weights(dlev,fields.device)
        feats = KernelModule.integrate(fields,norm,dlev)
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

    class TopHatKernel(torch.nn.Module):

        def __init__(self,nfieldvars):
            '''
            Purpose: Initialize parameters of a 1D top-hat kernel.
            Args:
            - nfieldvars (int): number of predictor fields
            '''
            super().__init__()
            self.lower = torch.nn.Parameter(torch.full((int(nfieldvars),),-0.5))
            self.upper = torch.nn.Parameter(torch.full((int(nfieldvars),),0.5))

        def forward(self,nlevs,device):
            '''
            Purpose: Evaluate the top-hat kernel over vertical levels.
            Args:
            - nlevs (int): number of vertical levels
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: top-hat kernel values with shape (nfieldvars, nlevs)
            '''
            coord = torch.linspace(-1.0,1.0,steps=nlevs,device=device)
            s1,s2 = torch.min(self.lower,self.upper),torch.max(self.lower,self.upper)
            width = s2-s1
            widthconstrained = torch.where(width>1.5,1.5*torch.tanh(width/1.5),width)
            leftedge  = torch.sigmoid((coord[None,:]-s1[:,None])/0.02)
            rightedge = torch.sigmoid(((s1+widthconstrained)[:,None]-coord[None,:])/0.02)
            kernel1D  = leftedge*rightedge+1e-8
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
            self.mu1      = torch.nn.Parameter(torch.full((int(nfieldvars),),-0.5))
            self.mu2      = torch.nn.Parameter(torch.full((int(nfieldvars),),0.5))
            self.logstd1  = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.logstd2  = torch.nn.Parameter(torch.zeros(int(nfieldvars)))
            self.weight1  = torch.nn.Parameter(torch.ones(int(nfieldvars)))
            self.weight2  = torch.nn.Parameter(torch.ones(int(nfieldvars)))

        def get_components(self,nlevs,device):
            '''
            Purpose: Compute individual Gaussian components separately (for visualization).
            Args:
            - nlevs (int): number of vertical levels
            - device (str | torch.device): device to use
            Returns:
            - tuple: tuple of individual Gaussian kernels each with shape (nfieldvars, nlevs)
            '''
            coord = torch.linspace(-1.0,1.0,steps=nlevs,device=device)
            std1  = torch.exp(self.logstd1).clamp(min=0.1,max=2.0)
            std2  = torch.exp(self.logstd2).clamp(min=0.1,max=2.0)
            component1 = self.weight1[:,None]*(torch.exp(-(coord[None,:]-self.mu1[:,None])**2/(2*std1[:,None]**2)))
            component2 = self.weight2[:,None]*(torch.exp(-(coord[None,:]-self.mu2[:,None])**2/(2*std2[:,None]**2)))
            return component1,component2

        def forward(self,nlevs,device):
            '''
            Purpose: Evaluate a mixture-of-Gaussians kernel over vertical levels.
            Args:
            - nlevs (int): number of vertical levels
            - device (str | torch.device): device to use
            Returns:
            - torch.Tensor: mixture kernel values with shape (nfieldvars, nlevs)
            '''
            component1,component2 = self.get_components(nlevs,device)
            kernel1D = component1+component2
            kernel1D = kernel1D+1e-8
            return kernel1D

    KERNEL_TYPES = {
        'gaussian':GaussianKernel,
        'tophat':TopHatKernel,
        'exponential':ExponentialKernel,
        'mixgaussian':MixtureGaussianKernel}

    def __init__(self,nfieldvars,kerneltype):
        '''
        Purpose: Initialize a parametric vertical kernel.
        Args:
        - nfieldvars (int): number of predictor fields
        - kerneltype (str): 'gaussian' | 'tophat' | 'exponential' | 'mixgaussian'
        '''
        super().__init__()
        self.nfieldvars = int(nfieldvars)
        self.kerneltype = str(kerneltype)
        self.norm       = None
        self.components = None
        self.features   = None
        if kerneltype not in self.KERNEL_TYPES:
            raise ValueError(f'Unknown kernel type `{kerneltype}`; must be one of {list(self.KERNEL_TYPES.keys())}')
        self.function = self.KERNEL_TYPES[kerneltype](self.nfieldvars)

    def get_weights(self,dlev,device,decompose=False):
        '''
        Purpose: Obtain normalized parametric kernel weights.
        Args:
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - device (str | torch.device): device to use
        - decompose (bool): whether to decompose mixture kernels into individual components (default: False)
        Returns:
        - torch.Tensor: normalized kernel weights with shape (nfieldvars, nlevs)
        '''
        dlev = dlev.to(device)
        nlevs = dlev.numel()
        kernel = self.function(nlevs,device)
        self.norm = KernelModule.normalize(kernel,dlev)
        self.components = None
        if decompose and isinstance(self.function,self.MixtureGaussianKernel):
            c1,c2 = self.function.get_components(nlevs,device)
            normc1 = KernelModule.normalize(c1,dlev)
            normc2 = KernelModule.normalize(c2,dlev)
            self.components = torch.stack([normc1,normc2],dim=0)
        return self.norm

    def forward(self,fields,dlev):
        '''
        Purpose: Apply parametric kernel to a batch of vertical profiles.
        Args:
        - fields (torch.Tensor): predictor fields with shape (nbatch, nfieldvars, nlevs)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        Returns:
        - torch.Tensor: kernel-integrated features with shape (nbatch, nfieldvars)
        '''
        norm = self.get_weights(dlev,fields.device)
        feats = KernelModule.integrate(fields,norm,dlev)
        self.features = feats
        return feats
