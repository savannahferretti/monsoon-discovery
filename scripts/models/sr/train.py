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
    Purpose: Parse command-line arguments for running the SR training script.
    Returns:
    - set[str] | None: run names to train, or None if all runs should be trained
    '''
    parser = argparse.ArgumentParser(description='Train symbolic regression models')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated list of run names to train, or `all`')
    args = parser.parse_args()
    return None if args.runs=='all' else {name.strip() for name in args.runs.split(',')}

if __name__=='__main__':
    config = Config()
    sr     = config.sr
    runs   = sr['runs']
    selectedruns = parse()
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        logger.info(f'Training SR model `{name}`...')
        features_from = runconfig.get('features_from',None)
        if features_from is not None:
            # TODO: load kernel-integrated features saved by the NN run specified in features_from
            logger.info(f'   Loading kernel-integrated features from NN run `{features_from}`...')
        else:
            # TODO: load normalized split data directly for scalar inputs (bl, cape/subsat)
            logger.info(f'   Loading normalized split data for {runconfig["fieldvars"]}...')
        # TODO: run PySR on features + lf to predict pr
        # TODO: save best equation to config.modelsdir/sr/
