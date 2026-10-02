#!/usr/bin/env python

import os
import logging
import argparse
import warnings
from timingutils import TimingConfig,parse_names
from scripts.data.classes import DataSplitter

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

SPLITFILES = ['train.h5','valid.h5','test.h5','norm_train.h5','norm_valid.h5','norm_test.h5','stats.json']

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create raw and normalized splits for the timing variants.')
    parser.add_argument('--variants',type=str,default='all',help='Comma-separated variant names, or `all`')
    args     = parser.parse_args()
    variants = parse_names(args.variants,list(TimingConfig().timing['variants']))
    for variant in variants:
        config = TimingConfig(variant)
        if all(os.path.exists(os.path.join(config.splitsdir,filename)) for filename in SPLITFILES):
            logger.info(f'Skipping `{variant}`, splits already exist')
            continue
        logger.info(f'Creating splits for `{variant}`...')
        splitter = DataSplitter(
            filedir=config.interimdir,
            savedir=config.splitsdir,
            trainrange=config.trainrange,
            validrange=config.validrange,
            testrange=config.testrange)
        trainstats = None
        for splitname,splitrange in [('train',splitter.trainrange),('valid',splitter.validrange),('test',splitter.testrange)]:
            splitds = splitter.split(splitrange)
            splitter.save(splitds,splitname)
            if splitname=='train':
                trainstats = splitter.calc_stats(splitds)
            normds = splitter.normalize(splitds,trainstats)
            splitter.save(normds,f'norm_{splitname}')
            del splitds,normds
