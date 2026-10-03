#!/usr/bin/env python

import os
import json
import glob
import logging
import argparse
import warnings
import numpy as np
import xarray as xr
from timingutils import TimingConfig,parse_names,load_dataset,save_dataset

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

SPLITFILES = ['train.h5','valid.h5','test.h5','stats.json']
NOSTATS    = ('dsig','lf')

def calc_stats(trainds):
    '''
    Purpose: Training-set mean and standard deviation of each variable (log1p for precipitation), in float64.
        Same keys as scripts/data/classes/splitter.py.
    Args:
    - trainds (xr.Dataset): float64 training Dataset
    Returns:
    - dict[str, float]: statistics
    '''
    stats = {}
    for name,da in trainds.data_vars.items():
        if name in NOSTATS:
            continue
        values = np.log1p(da.values) if name in ('pr','tp') else da.values
        stats[f'{name}_mean'] = float(np.nanmean(values))
        stats[f'{name}_std']  = float(np.nanstd(values))
    return stats

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create splits and training statistics for the timing variants.')
    parser.add_argument('--variants',type=str,default='all',help='Comma-separated variant names, or `all`')
    args     = parser.parse_args()
    variants = parse_names(args.variants,list(TimingConfig().timing['variants']))
    for variant in variants:
        config = TimingConfig(variant)
        if all(os.path.exists(os.path.join(config.splitsdir,filename)) for filename in SPLITFILES):
            logger.info(f'Skipping `{variant}`, splits already exist')
            continue
        logger.info(f'Creating splits for `{variant}`...')
        datavars = {}
        for filepath in sorted(glob.glob(os.path.join(config.interimdir,'*.nc'))):
            ds = load_dataset(filepath)
            for name,da in ds.data_vars.items():
                datavars[name] = da.transpose(*[dim for dim in ('lat','lon','sig','time') if dim in da.dims])
        full = xr.Dataset(datavars)
        for split,(start,end) in [('train',config.trainrange),('valid',config.validrange),('test',config.testrange)]:
            splitds = full.sel(time=(full.time.dt.year>=start)&(full.time.dt.year<=end))
            if split=='train':
                stats = calc_stats(splitds)
                os.makedirs(config.splitsdir,exist_ok=True)
                with open(os.path.join(config.splitsdir,'stats.json'),'w',encoding='utf-8') as f:
                    json.dump(stats,f)
                with open(os.path.join(config.splitsdir,'stats.json'),'r',encoding='utf-8') as f:
                    json.load(f)
                logger.info('   Wrote stats.json')
            save_dataset(splitds,os.path.join(config.splitsdir,f'{split}.h5'))
