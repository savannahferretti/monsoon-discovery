#!/usr/bin/env python

from scripts.models.nn.architectures import BaselineNN,KernelNN
from scripts.models.nn.kernels import NonparametricKernelLayer,ParametricKernelLayer

def build_model(name,runconfig,nlevs):
    '''
    Purpose: Build a model instance from a run configuration.
    Args:
    - name (str): model name
    - runconfig (dict): run configuration from configs.json experiments.nn.runs
    - nlevs (int): number of vertical levels (1 for scalar inputs)
    Returns:
    - torch.nn.Module: initialized model
    '''
    kind       = runconfig['kind']
    nfieldvars = len(runconfig['fieldvars'])
    if kind=='baseline':
        model = BaselineNN(nfieldvars,nlevs)
    elif kind=='nonparametric':
        kernel = NonparametricKernelLayer(nfieldvars,nlevs)
        model  = KernelNN(kernel,nfieldvars)
    elif kind=='parametric':
        kerneltype = runconfig['kernel']
        kernel = ParametricKernelLayer(nfieldvars,kerneltype)
        model  = KernelNN(kernel,nfieldvars)
    else:
        raise ValueError(f'Unknown model kind `{kind}`')
    model.nparams = sum(param.numel() for param in model.parameters())
    return model
