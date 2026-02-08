#!/usr/bin/env python

import os
import torch
import logging
import argparse
import numpy as np
from scripts.utils import Config
from scripts.models.nn.classes.factory import build_model
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
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        for seed in seeds:
            runid = f'{name}_{seed}'
            if os.path.exists(os.path.join(config.modelsdir,'nn',f'{runid}.pth')):
                logger.info(f'Skipping `{runid}`, checkpoint already exists')
                continue
            logger.info(f'Training `{runid}`...')
            device = setup(seed)
            # TODO: load normalized split data for runconfig['fieldvars'] + lf + pr
            # trainloader, validloader = ...
            # nlevs = number of vertical levels (1 for scalar inputs)
            nlevs = 1  # placeholder
            model = build_model(name,runconfig,nlevs).to(device)
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
                criterion=nn['criterion'],
                epochs=nn['epochs'],
                useamp=True,
                accumsteps=1,
                compile=False)
            trainer.fit(name)
            del model,trainer
