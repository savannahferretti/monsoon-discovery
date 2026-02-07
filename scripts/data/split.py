#!/usr/bin/env python

import logging
import warnings
from scripts.utils import Config
from scripts.data.classes import DataSplitter

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

if __name__=='__main__':
    config   = Config()
    splitter = DataSplitter(
        filedir=config.interimdir,
        savedir=config.splitsdir,
        trainrange=config.trainrange,
        validrange=config.validrange,
        testrange=config.testrange)
    logger.info('Setting up splits...')
    splits = [
        ('train',splitter.trainrange),
        ('valid',splitter.validrange),
        ('test', splitter.testrange)]
    logger.info('Creating and saving regular and normalized data splits...')
    trainstats = None
    for splitname, splitrange in splits:
        splitds = splitter.split(splitrange)
        splitter.save(splitds,splitname)
        if splitname == 'train':
            trainstats = splitter.calc_stats(splitds)
        normds = splitter.normalize(splitds,trainstats)
        splitter.save(normds,f'norm_{splitname}')
        del normds