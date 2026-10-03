#!/usr/bin/env python

import os
import torch
import logging
import argparse
import numpy as np
from timingutils import TimingConfig,parse_names,load_stats,z_to_precip
from data import load_nn_arrays,unflatten,save_predictions
from nn_train import setup,build_model,to_tensors
from scripts.models.nn.classes.dataset import FieldDataset
from scripts.models.nn.classes.inferencer import Inferencer

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate NN predictions for the timing variants.')
    parser.add_argument('--variants',type=str,default='all',help='Comma-separated variant names, or `all`')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated NN run names, or `all`')
    parser.add_argument('--split',type=str,default='test',help='Split to evaluate: train|valid|test (default: test)')
    parser.add_argument('--seeds',type=str,default=None,help='Comma-separated seeds overriding the config')
    args = parser.parse_args()
    baseconfig = TimingConfig()
    variants   = parse_names(args.variants,list(baseconfig.timing['variants']))
    for variant in variants:
        config    = TimingConfig(variant)
        nn        = config.nn
        targetvar = config.targetvar
        stats     = load_stats(config)
        seeds     = [int(seed) for seed in args.seeds.split(',')] if args.seeds else nn['seeds']
        device    = setup(seeds[0])
        runs      = config.nnruns
        for name in parse_names(args.runs,list(runs)):
            if os.path.exists(os.path.join(config.predsdir,f'{name}_{args.split}_predictions.nc')):
                logger.info(f'[{variant}] Skipping `{name}`, predictions already exist')
                continue
            runconfig = runs[name]
            haskernel = runconfig['kind']!='baseline'
            fields,local,target,dsig,nlevs,valid,truth = load_nn_arrays(config,args.split,runconfig)
            fields,local,target,dsig = to_tensors(fields,local,target,dsig)
            dataloader = torch.utils.data.DataLoader(FieldDataset(fields,local,target,dsig),batch_size=nn['batchsize'],shuffle=False,num_workers=0,pin_memory=True)
            grids = []
            for seed in seeds:
                filepath = os.path.join(config.modelsdir,'nn',f'{name}_{seed}.pth')
                if not os.path.exists(filepath):
                    logger.error(f'[{variant}] Checkpoint not found: {filepath}')
                    break
                logger.info(f'[{variant}] Evaluating `{name}` seed {seed}...')
                model = build_model(runconfig,nlevs,stats,targetvar)
                model.load_state_dict(torch.load(filepath,map_location='cpu'))
                z,_ = Inferencer(model.to(device),dataloader,device).predict(haskernel)
                grids.append(unflatten(z_to_precip(z.astype(np.float64),stats),valid,truth))
                del model
            else:
                save_predictions(config,name,args.split,grids,truth,seeds=seeds)
