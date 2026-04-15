#!/usr/bin/env python

import os
import json
import torch
import logging
import argparse
import numpy as np
import xarray as xr
from scripts.utils import Config
from scripts.data.classes import PredictionWriter
from scripts.models.nn.classes.factory import build_model
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

def parse():
    '''
    Purpose: Parse command-line arguments for running the training script.
    Returns:
    - set[str] | None: run names to train, or None if all runs should be trained
    '''
    parser = argparse.ArgumentParser(description='Train and validate NN models')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated list of run names to train, or `all`')
    args = parser.parse_args()
    return None if args.runs=='all' else {name.strip() for name in args.runs.split(',')}

if __name__=='__main__':
    config    = Config()
    nn        = config.nn
    runs      = nn['runs']
    seeds     = nn['seeds']
    targetvar = config.targetvar
    logger.info('Spinning up...')
    selectedruns = parse()
    cachedkey    = None
    cacheddata   = None
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        fieldvars  = runconfig['fieldvars']
        localvars  = runconfig.get('localvars',[])
        subset     = runconfig.get('subset')
        subsetkey  = tuple(sorted(subset.items())) if subset else None
        cachekey   = (tuple(fieldvars),tuple(localvars),subsetkey)
        if cachekey!=cachedkey:
            logger.info(f'Loading normalized splits for fieldvars={fieldvars}, localvars={localvars}, targetvar={targetvar}...')
            trainfields,trainlocal,trainpr,dsig,nlevs,_,_  = load_split('train',fieldvars,localvars,config.splitsdir,targetvar=targetvar,subset=subset)
            validfields,validlocal,validpr,_,_,_,_          = load_split('valid',fieldvars,localvars,config.splitsdir,targetvar=targetvar,subset=subset)
            cachedkey  = cachekey
            cacheddata = (trainfields,trainlocal,trainpr,validfields,validlocal,validpr,dsig,nlevs)
        else:
            trainfields,trainlocal,trainpr,validfields,validlocal,validpr,dsig,nlevs = cacheddata
        traindataset = FieldDataset(trainfields,trainlocal,trainpr,dsig)
        validdataset = FieldDataset(validfields,validlocal,validpr,dsig)
        trainloader = torch.utils.data.DataLoader(traindataset,batch_size=nn['batchsize'],shuffle=True,num_workers=nn['workers'],pin_memory=True)
        validloader = torch.utils.data.DataLoader(validdataset,batch_size=nn['batchsize'],shuffle=False,num_workers=nn['workers'],pin_memory=True)
        for seed in seeds:
            runid = f'{name}_{seed}'
            if os.path.exists(os.path.join(config.modelsdir,'nn',f'{runid}.pth')):
                logger.info(f'Skipping `{runid}`, checkpoint already exists')
                continue
            logger.info(f'Training `{runid}`...')
            device = setup(seed)
            model  = build_model(name,runconfig,nlevs).to(device)
            criterion        = runconfig.get('criterion',nn['criterion'])
            criterionkwargs  = runconfig.get('criterionkwargs',nn.get('criterionkwargs',{}))
            if criterion=='TweedieLoss':
                from scripts.models.nn.architectures import TARGETSTATS
                stats = TARGETSTATS[targetvar]
                criterionkwargs = {**criterionkwargs,'mean':stats['mean'],'std':stats['std']}
            trainer = Trainer(
                model=model,
                trainloader=trainloader,
                validloader=validloader,
                device=device,
                modeldir=os.path.join(config.modelsdir,'nn'),
                project=nn['projectname'],
                seed=seed,
                lr=nn['learningrate'],
                patience=nn['patience'],
                criterion=criterion,
                criterionkwargs=criterionkwargs,
                epochs=nn['epochs'],
                useamp=True,
                accumsteps=1,
                compile=False)
            trainer.fit(name)
            metapath = os.path.join(config.modelsdir,'nn',f'{name}_meta.json')
            if not os.path.exists(metapath):
                os.makedirs(os.path.join(config.modelsdir,'nn'),exist_ok=True)
                with open(metapath,'w') as f:
                    json.dump({'nparams':model.nparams},f)
                logger.info(f'   Saved nparams metadata for `{name}`')
            haskernel = hasattr(model,'kernel')
            if haskernel:
                logger.info(f'   Saving kernel weights for `{runid}`...')
                model.eval()
                with torch.no_grad():
                    model.kernel.get_weights(dsig.to(device),device)
                weights = model.kernel.norm.detach().cpu().numpy().astype(np.float32)
                refds = xr.open_dataset(os.path.join(config.splitsdir,'norm_train.h5'),engine='h5netcdf')
                ds = PredictionWriter.weights_to_dataset(weights[...,np.newaxis],fieldvars,refds)
                refds.close()
                os.makedirs(config.weightsdir,exist_ok=True)
                wpath = os.path.join(config.weightsdir,f'{runid}_weights.nc')
                ds.to_netcdf(wpath,engine='h5netcdf')
                logger.info(f'      Saved to {wpath}')
            del model,trainer