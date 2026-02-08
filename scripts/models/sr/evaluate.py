#!/usr/bin/env python

import os
import logging
import argparse
import numpy as np
from scripts.utils import Config

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse():
    '''
    Purpose: Parse command-line arguments for running the SR evaluation script.
    Returns:
    - tuple[set[str] | None, str]: run names to evaluate or None if all, and split name
    '''
    parser = argparse.ArgumentParser(description='Evaluate symbolic regression models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated list of run names to evaluate, or `all`.')
    parser.add_argument('--split',type=str,default='test',help='Split to evaluate: train|valid|test (default: test)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {name.strip() for name in args.runs.split(',') if name.strip()}
    return selectedruns,args.split

if __name__=='__main__':
    config = Config()
    sr     = config.sr
    runs   = sr['runs']
    selectedruns,split = parse()
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        logger.info(f'Evaluating SR model `{name}` on {split} split...')
        features_from = runconfig.get('features_from',None)
        if features_from is not None:
            # TODO: load kernel-integrated features from the NN run
            logger.info(f'   Loading kernel-integrated features from NN run `{features_from}`...')
        else:
            # TODO: load normalized split data directly for scalar inputs
            logger.info(f'   Loading normalized split data for {runconfig["fieldvars"]}...')
        # TODO: load trained SR model from config.modelsdir/sr/
        # TODO: run predictions and save to config.predsdir
