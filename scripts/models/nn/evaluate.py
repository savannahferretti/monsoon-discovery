#!/usr/bin/env python

import os
import torch
import logging
import argparse
import numpy as np
import xarray as xr
from scripts.utils import Config
from scripts.data.classes import PredictionWriter
from scripts.models.nn.classes.factory import build_model
from scripts.models.nn.classes.dataset import FieldDataset,load_split
from scripts.models.nn.classes.inferencer import Inferencer

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
    Purpose: Parse command-line arguments for running the evaluation script.
    Returns:
    - tuple[set[str] | None, str]: run names to evaluate or None if all, and split name
    '''
    parser = argparse.ArgumentParser(description='Evaluate NN models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated list of run names to evaluate, or `all`.')
    parser.add_argument('--split',type=str,default='test',help='Split to evaluate: train|valid|test (default: test)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {name.strip() for name in args.runs.split(',') if name.strip()}
    return selectedruns,args.split

def load(name,runconfig,nlevs,modeldir,seed,device):
    '''
    Purpose: Build a model and load weights from a saved checkpoint.
    Args:
    - name (str): run name
    - runconfig (dict): run configuration
    - nlevs (int): number of vertical levels
    - modeldir (str): directory containing checkpoints
    - seed (int): random seed used during training
    - device (str): device to use
    Returns:
    - torch.nn.Module: model with loaded state_dict on device, or None if checkpoint not found
    '''
    filepath = os.path.join(modeldir,f'{name}_{seed}.pth')
    if not os.path.exists(filepath):
        logger.error(f'   Checkpoint not found: {filepath}')
        return None
    model = build_model(name,runconfig,nlevs)
    state = torch.load(filepath,map_location='cpu')
    model.load_state_dict(state)
    return model.to(device)

if __name__=='__main__':
    config = Config()
    nn     = config.nn
    runs   = nn['runs']
    seeds  = nn['seeds']
    logger.info('Spinning up...')
    device = setup(seeds[0])
    selectedruns,split = parse()
    writer = PredictionWriter(config.splitsdir)
    cachedvars = None
    cacheddata = None
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        if os.path.exists(os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')):
            logger.info(f'Skipping `{name}`, predictions already exist')
            continue
        haskernel = runconfig['kind']!='baseline'
        fieldvars = runconfig['fieldvars']
        localvars = runconfig.get('localvars',[])
        fieldkey  = (tuple(fieldvars),tuple(localvars))
        if fieldkey!=cachedvars:
            logger.info(f'Loading normalized {split} split for {fieldvars}...')
            fields,local,pr,dlev,nlevs,mask,valid,refda = load_split(split,fieldvars,localvars,config.splitsdir)
            cachedvars = fieldkey
            cacheddata = (fields,local,pr,dlev,nlevs,mask,valid,refda)
        else:
            fields,local,pr,dlev,nlevs,mask,valid,refda = cacheddata
        dataset    = FieldDataset(fields,local,pr,dlev,mask=mask)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=nn['batchsize'],shuffle=False,num_workers=0,pin_memory=True)
        allpreds   = []
        allweights = []
        allfeats   = []
        for seedidx,seed in enumerate(seeds):
            logger.info(f'   Evaluating `{name}` seed {seedidx+1}/{len(seeds)} ({seed})...')
            model = load(name,runconfig,nlevs,os.path.join(config.modelsdir,'nn'),seed,device)
            if model is None:
                logger.error(f'   Failed to load model for seed {seed}, skipping...')
                break
            inferencer = Inferencer(model,dataloader,device)
            preds,feats = inferencer.predict(haskernel)
            allpreds.append(writer.predictions_to_array(preds,valid,refda))
            if haskernel:
                allweights.append(inferencer.extract_weights())
                allfeats.append(writer.features_to_array(feats,valid,refda))
            del model,inferencer
        else:
            logger.info(f'   Saving predictions for `{name}`...')
            predstack = np.stack(allpreds,axis=-1)
            ds = writer.predictions_to_dataset(predstack,refda)
            writer.save(name,ds,'predictions',split,config.predsdir)
            del predstack,ds
            if haskernel:
                logger.info(f'   Saving kernel weights for `{name}`...')
                stacked = np.stack(allweights,axis=-1)
                refds = xr.open_dataset(os.path.join(config.splitsdir,f'norm_{split}.h5'),engine='h5netcdf')
                ds = writer.weights_to_dataset([stacked],fieldvars,refds)
                refds.close()
                writer.save(name,ds,'weights',split,config.weightsdir)
                del stacked,ds
                logger.info(f'   Saving kernel-integrated features for `{name}`...')
                featstack = np.stack(allfeats,axis=-1)
                ds = writer.features_to_dataset(featstack,fieldvars,refda)
                writer.save(name,ds,'features',split,config.featsdir)
                del featstack,ds