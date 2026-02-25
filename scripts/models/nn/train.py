#!/usr/bin/env python

import os
import json
import torch
import logging
import argparse
import numpy as np
from scripts.utils import Config
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
    config = Config()
    nn     = config.nn
    runs   = nn['runs']
    seeds  = nn['seeds']
    logger.info('Spinning up...')
    selectedruns = parse()
    cachedkey    = None
    cacheddata   = None
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        fieldvars  = runconfig['fieldvars']
        localvars  = runconfig.get('localvars',[])
        cachekey   = (tuple(fieldvars),tuple(localvars))
        if cachekey!=cachedkey:
            logger.info(f'Loading normalized splits for fieldvars={fieldvars}, localvars={localvars}...')
            trainfields,trainlocal,trainpr,dlev,nlevs,trainmask,_,_  = load_split('train',fieldvars,localvars,config.splitsdir)
            validfields,validlocal,validpr,_,_,validmask,_,_         = load_split('valid',fieldvars,localvars,config.splitsdir)
            cachedkey  = cachekey
            cacheddata = (trainfields,trainlocal,trainpr,validfields,validlocal,validpr,dlev,nlevs,trainmask,validmask)
        else:
            trainfields,trainlocal,trainpr,validfields,validlocal,validpr,dlev,nlevs,trainmask,validmask = cacheddata
        traindataset = FieldDataset(trainfields,trainlocal,trainpr,dlev,mask=trainmask)
        validdataset = FieldDataset(validfields,validlocal,validpr,dlev,mask=validmask)
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
            del model,trainer