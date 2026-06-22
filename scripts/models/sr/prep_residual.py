#!/usr/bin/env python

import os
import json
import logging
import numpy as np
from scripts.utils import Config
from scripts.models.sr.train import load_data

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def compute_u_target(target,zmin):
    '''
    Purpose: Convert the floored z-scored target into the pre-cube quantity that SR-HI's
        outer cube(...) actually predicts (y_z = zmin + relu(cube(u))), so a thermo-term
        contribution can be subtracted additively before the cube nonlinearity is applied.
    Args:
    - target (np.ndarray): z-scored log1p(tp) target values
    - zmin (float): z-scored value corresponding to 0 mm precipitation
    Returns:
    - np.ndarray: pre-cube target u, with u = cbrt(max(target - zmin, 0))
    '''
    return np.cbrt(np.maximum(target-zmin,0.0))

if __name__=='__main__':
    config    = Config()
    sr        = config.sr
    targetvar = config.targetvar
    thermorun = sr['runs']['sr_hi_thermo']
    thermoeq  = sr['optimizedeqs']['sr_hi']
    a,b,c     = thermoeq['init']['a'],thermoeq['init']['b'],thermoeq['init']['c']
    statsfile = os.path.join(config.splitsdir,'stats.json')
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    zmin = (0.0-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
    logger.info('Loading train and validation splits for thermo predictors...')
    xtrain,ytrain,reftrain,trainmask = load_data('train',thermorun,config,time_offset=0)
    xvalid,yvalid,_,validmask        = load_data('valid',thermorun,config,time_offset=int(reftrain.sizes['time']))
    outdir = os.path.join(config.splitsdir,'residuals')
    os.makedirs(outdir,exist_ok=True)
    for splitname,x,y,mask in [('train',xtrain,ytrain,trainmask),('valid',xvalid,yvalid,validmask)]:
        xmasked    = x[mask]
        u          = compute_u_target(y[mask],zmin)
        thermopred = a*np.maximum(xmasked['rh'].values,xmasked['thetae'].values+b*xmasked['thetaestar'].values+c)
        residual_u = u-thermopred
        position   = np.where(mask)[0]
        outpath    = os.path.join(outdir,f'sr_hi_surface_{splitname}.npz')
        np.savez(outpath,residual=residual_u,position=position)
        logger.info(f'   Saved {len(residual_u):,} residual samples to {outpath}')
