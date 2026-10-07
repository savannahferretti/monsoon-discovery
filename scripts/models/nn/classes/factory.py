#!/usr/bin/env python

from scripts.utils import Config,load_stats
from scripts.models.nn.architectures import BaselineNN,KernelNN
from scripts.models.nn.kernels import NonparametricKernelLayer,ParametricKernelLayer

def build_model(runconfig,nlevs):
    '''
    Purpose: Build an untrained model for a run, with the target's training statistics.
    Args:
    - runconfig (dict): run configuration from configs.json experiments.nn.runs
    - nlevs (int): number of vertical levels (1 for scalar inputs)
    Returns:
    - torch.nn.Module: initialized model
    '''
    config     = Config()
    stats      = load_stats(config.splitsdir)
    mean       = stats[f'{config.targetvar}_mean']
    std        = stats[f'{config.targetvar}_std']
    kind       = runconfig['kind']
    nfieldvars = len(runconfig['fieldvars'])
    nlocalvars = len(runconfig.get('localvars',[]))
    if kind=='baseline':
        model = BaselineNN(nfieldvars,nlevs,nlocalvars,mean,std)
    elif kind=='nonparametric':
        model = KernelNN(NonparametricKernelLayer(nfieldvars,nlevs),nfieldvars,nlocalvars,mean,std)
    elif kind=='parametric':
        model = KernelNN(ParametricKernelLayer(nfieldvars,runconfig['kernel']),nfieldvars,nlocalvars,mean,std)
    else:
        raise ValueError(f'Unknown model kind `{kind}`')
    model.nparams = sum(param.numel() for param in model.parameters())
    return model