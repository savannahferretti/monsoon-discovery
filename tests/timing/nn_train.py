#!/usr/bin/env python

import os
import torch
import logging
import argparse
import numpy as np
import xarray as xr
from timingutils import TimingConfig,parse_names,load_stats
from scripts.data.classes import PredictionWriter
from scripts.models.nn.architectures import BaselineNN,KernelNN
from scripts.models.nn.kernels import NonparametricKernelLayer,ParametricKernelLayer
from scripts.models.nn.classes.dataset import FieldDataset,load_split
from scripts.models.nn.classes.trainer import Trainer

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def setup(seed):
    '''
    Purpose: Set random seeds for reproducibility and configure compute device.
    Args:
    - seed (int): random seed for NumPy and PyTorch
    Returns:
    - str: device to use
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    return device

def build_model(runconfig,nlevs,stats,targetvar):
    '''
    Purpose: Build a model as in scripts/models/nn/classes/factory.py, but with the variant's target statistics.
    Args:
    - runconfig (dict): run configuration
    - nlevs (int): number of vertical levels
    - stats (dict): variant training statistics
    - targetvar (str): target variable name
    Returns:
    - torch.nn.Module: initialized model
    '''
    kind       = runconfig['kind']
    nfieldvars = len(runconfig['fieldvars'])
    nlocalvars = len(runconfig.get('localvars',[]))
    mean       = stats[f'{targetvar}_mean']
    std        = stats[f'{targetvar}_std']
    if kind=='baseline':
        model = BaselineNN(nfieldvars,nlevs,nlocalvars,mean=mean,std=std)
    elif kind=='nonparametric':
        model = KernelNN(NonparametricKernelLayer(nfieldvars,nlevs),nfieldvars,nlocalvars,mean=mean,std=std)
    elif kind=='parametric':
        model = KernelNN(ParametricKernelLayer(nfieldvars,runconfig['kernel']),nfieldvars,nlocalvars,mean=mean,std=std)
    else:
        raise ValueError(f'Unknown model kind `{kind}`')
    model.nparams = sum(param.numel() for param in model.parameters())
    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train NN models for the timing variants.')
    parser.add_argument('--variants',type=str,default='all',help='Comma-separated variant names, or `all`')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated NN run names, or `all`')
    parser.add_argument('--seeds',type=str,default=None,help='Comma-separated seeds overriding the config (e.g. 42 for a quick test)')
    args = parser.parse_args()
    baseconfig = TimingConfig()
    variants   = parse_names(args.variants,list(baseconfig.timing['variants']))
    for variant in variants:
        config    = TimingConfig(variant)
        nn        = config.nn
        targetvar = config.targetvar
        stats     = load_stats(config)
        seeds     = [int(seed) for seed in args.seeds.split(',')] if args.seeds else nn['seeds']
        runs      = config.nnruns
        os.environ['WANDB_DIR']       = config.datadir
        os.environ['WANDB_RUN_GROUP'] = f'timing_{variant}'
        for name in parse_names(args.runs,list(runs)):
            runconfig = runs[name]
            fieldvars = runconfig['fieldvars']
            localvars = runconfig.get('localvars',[])
            todo = [seed for seed in seeds if not os.path.exists(os.path.join(config.modelsdir,'nn',f'{name}_{seed}.pth'))]
            if not todo:
                logger.info(f'[{variant}] Skipping `{name}`, all checkpoints already exist')
                continue
            logger.info(f'[{variant}] Loading normalized splits for `{name}`...')
            trainfields,trainlocal,trainpr,dsig,nlevs,_,_ = load_split('train',fieldvars,localvars,config.splitsdir,targetvar=targetvar)
            validfields,validlocal,validpr,_,_,_,_        = load_split('valid',fieldvars,localvars,config.splitsdir,targetvar=targetvar)
            trainloader = torch.utils.data.DataLoader(FieldDataset(trainfields,trainlocal,trainpr,dsig),batch_size=nn['batchsize'],shuffle=True,num_workers=nn['workers'],pin_memory=True)
            validloader = torch.utils.data.DataLoader(FieldDataset(validfields,validlocal,validpr,dsig),batch_size=nn['batchsize'],shuffle=False,num_workers=nn['workers'],pin_memory=True)
            for seed in todo:
                runid = f'{name}_{seed}'
                logger.info(f'[{variant}] Training `{runid}`...')
                device = setup(seed)
                model  = build_model(runconfig,nlevs,stats,targetvar).to(device)
                trainer = Trainer(
                    model=model,
                    trainloader=trainloader,
                    validloader=validloader,
                    device=device,
                    modeldir=os.path.join(config.modelsdir,'nn'),
                    project=config.timing['nn']['projectname'],
                    seed=seed,
                    lr=nn['learningrate'],
                    patience=nn['patience'],
                    criterion=runconfig.get('criterion',nn['criterion']),
                    criterionkwargs=runconfig.get('criterionkwargs',nn.get('criterionkwargs',{})),
                    epochs=nn['epochs'],
                    useamp=True,
                    accumsteps=1,
                    compile=False)
                trainer.fit(name)
                if hasattr(model,'kernel'):
                    logger.info(f'   Saving kernel weights for `{runid}`...')
                    model.eval()
                    with torch.no_grad():
                        model.kernel.get_weights(dsig.to(device),device)
                    weights = model.kernel.norm.detach().cpu().numpy().astype(np.float32)
                    with xr.open_dataset(os.path.join(config.splitsdir,'norm_train.h5'),engine='h5netcdf') as refds:
                        ds = PredictionWriter.weights_to_dataset(weights,fieldvars,refds)
                    os.makedirs(config.weightsdir,exist_ok=True)
                    wpath = os.path.join(config.weightsdir,f'{runid}_weights.nc')
                    ds.to_netcdf(wpath,engine='h5netcdf')
                    xr.open_dataset(wpath,engine='h5netcdf').close()
                    logger.info(f'      Saved to {wpath}')
                del model,trainer
            del trainloader,validloader,trainfields,validfields
