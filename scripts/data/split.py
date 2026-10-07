#!/usr/bin/env python

import os
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
    splits = [
        ('train',splitter.trainrange),
        ('valid',splitter.validrange),
        ('test',splitter.testrange)]
    filenames = [f'{splitname}.h5' for splitname,_ in splits]+['stats.json']
    if all(os.path.exists(os.path.join(config.splitsdir,filename)) for filename in filenames):
        logger.info('Skipping, all split files already exist')
    else:
        logger.info('Loading interim data...')
        ds = splitter.combine()
        logger.info('Creating and saving data splits...')
        for splitname,splitrange in splits:
            splitds = splitter.split(ds,splitrange)
            if splitname=='train':
                splitter.calc_stats(splitds)
            splitter.save(splitds,splitname)
            del splitds
