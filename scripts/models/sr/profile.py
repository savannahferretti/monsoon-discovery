#!/usr/bin/env python

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from scripts.utils import Config
from scripts.models.sr.optimize import (
    extract_constants,eval_form,pysr_init,load_data)

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def parse():
    parser = argparse.ArgumentParser(
        description='Profile loss surface over one or two constants while re-optimizing the rest.')
    parser.add_argument('--equation',type=str,required=True,
                        help='Name of equation in configs.json optimizedeqs')
    parser.add_argument('--constants',type=str,required=True,
                        help='Comma-separated constant names to profile (1 or 2)')
    parser.add_argument('--ranges',type=str,default=None,
                        help='Comma-separated lo:hi ranges per constant (default: 0.2:3.0)')
    parser.add_argument('--gridsize',type=int,default=15,
                        help='Number of grid points per constant (default: 15)')
    parser.add_argument('--nrestarts',type=int,default=20,
                        help='Multistart restarts for free constants at each grid point (default: 20)')
    parser.add_argument('--outdir',type=str,default=None,
                        help='Output directory for profile CSV (default: models/sr/)')
    args = parser.parse_args()
    profileconstants = [c.strip() for c in args.constants.split(',')]
    if len(profileconstants) > 2:
        parser.error('At most 2 constants can be profiled at once')
    if args.ranges is not None:
        rangestrs = args.ranges.split(',')
        if len(rangestrs) != len(profileconstants):
            parser.error('Number of ranges must match number of constants')
        ranges = []
        for r in rangestrs:
            lo,hi = r.split(':')
            ranges.append((float(lo),float(hi)))
    else:
        ranges = [(0.2,3.0)] * len(profileconstants)
    return args.equation,profileconstants,ranges,args.gridsize,args.nrestarts,args.outdir


def optimize_free(form,predictornames,constantnames,x,y,zmin,zmax,init,fixed):
    freenames = [c for c in constantnames if c not in fixed]
    freeinit = np.array([init.get(c,1.0) for c in freenames])
    def objective(params):
        allconstants = dict(fixed)
        allconstants.update(dict(zip(freenames,params)))
        raw = eval_form(form,x,predictornames,allconstants)
        softplus = np.where(raw>20.0,raw,np.log1p(np.exp(np.minimum(raw,20.0))))
        pred = np.clip(zmin+softplus,None,zmax)
        return float(np.mean((pred-y)**2))
    def relu_objective(params):
        allconstants = dict(fixed)
        allconstants.update(dict(zip(freenames,params)))
        raw = eval_form(form,x,predictornames,allconstants)
        pred = np.clip(zmin+np.maximum(raw,0.0),None,zmax)
        return float(np.mean((pred-y)**2))
    res1 = minimize(objective,freeinit,method='L-BFGS-B',
                    options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
    res2 = minimize(relu_objective,res1.x,method='L-BFGS-B',
                    options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
    allconstants = dict(fixed)
    allconstants.update(dict(zip(freenames,res2.x)))
    return allconstants,res2.fun


def multistart_profile(form,predictornames,constantnames,x,y,zmin,zmax,init,fixed,nrestarts,seed=0):
    freenames = [c for c in constantnames if c not in fixed]
    nconstants = len(freenames)
    if nconstants == 0:
        allconstants = dict(fixed)
        raw = eval_form(form,x,predictornames,allconstants)
        pred = np.clip(zmin+np.maximum(raw,0.0),None,zmax)
        return allconstants,float(np.mean((pred-y)**2))
    sampler = LatinHypercube(d=nconstants,seed=seed)
    samples = sampler.random(n=max(0,nrestarts-1))
    inits = [init]
    for i in range(len(samples)):
        restart = dict(init)
        for j,c in enumerate(freenames):
            if c in init:
                restart[c] = init[c] + (samples[i,j]*6.0-3.0)
            else:
                restart[c] = samples[i,j]*10.0-5.0
        inits.append(restart)
    bestconstants,bestloss = None,np.inf
    for restartinit in inits:
        constants,loss = optimize_free(
            form,predictornames,constantnames,x,y,zmin,zmax,restartinit,fixed)
        if loss < bestloss:
            bestconstants,bestloss = constants,loss
    return bestconstants,bestloss


if __name__=='__main__':
    import time
    config = Config()
    sr = config.sr
    targetvar = config.targetvar
    optimizedeqs = sr.get('optimizedeqs',{})
    eqname,profileconstants,ranges,gridsize,nrestarts,outdir = parse()
    if eqname not in optimizedeqs:
        raise ValueError(f'Equation {eqname} not found in configs.json optimizedeqs')
    eqspec = optimizedeqs[eqname]
    form = eqspec['form']
    runname = eqspec['runfrom']
    runconfig = sr['runs'][runname]
    refcomplexity = eqspec.get('refcomplexity')
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    from scripts.data.classes.writer import PMAX
    zmin = (0.0-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
    zmax = (np.log1p(PMAX)-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
    xtrain,ytrain,reftrain,trainmask = load_data('train',runconfig,config,time_offset=0)
    xvalid,yvalid,_,validmask = load_data('valid',runconfig,config,time_offset=int(reftrain.sizes['time']))
    xfit = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
    yfit = np.concatenate([ytrain[trainmask],yvalid[validmask]])
    del xtrain,ytrain,reftrain
    predictornames = [c for c in xfit.columns if c != 'timeidx']
    xfit = xfit[predictornames]
    constantnames = extract_constants(form,predictornames)
    for pc in profileconstants:
        if pc not in constantnames:
            raise ValueError(f'Constant {pc} not found in form. Available: {constantnames}')
    eq_seeds = eqspec.get('seeds',sr['seeds'])
    init = pysr_init(form,predictornames,refcomplexity,runname,eq_seeds,config.modelsdir)
    grids = [np.linspace(lo,hi,gridsize) for (lo,hi) in ranges]
    if len(profileconstants) == 1:
        gridpoints = [(v,) for v in grids[0]]
    else:
        gridpoints = list(product(grids[0],grids[1]))
    logger.info(f'Profiling {eqname}: {profileconstants} over {len(gridpoints)} grid points '
                f'({nrestarts} restarts each)')
    rows = []
    t0 = time.time()
    for i,pt in enumerate(gridpoints):
        fixed = dict(zip(profileconstants,pt))
        allconstants,loss = multistart_profile(
            form,predictornames,constantnames,xfit,yfit,zmin,zmax,init,fixed,nrestarts)
        row = dict(fixed)
        row['loss'] = loss
        for c in constantnames:
            if c not in profileconstants:
                row[f'{c}_opt'] = allconstants.get(c,np.nan)
        rows.append(row)
        logger.info(f'   [{i+1}/{len(gridpoints)}] '
                    f'{", ".join(f"{k}={v:.3f}" for k,v in fixed.items())} → '
                    f'loss={loss:.6f}')
    elapsed = time.time()-t0
    logger.info(f'Profile complete in {elapsed:.0f} s')
    if outdir is None:
        outdir = os.path.join(config.modelsdir,'sr')
    os.makedirs(outdir,exist_ok=True)
    suffix = '_'.join(profileconstants)
    outpath = os.path.join(outdir,f'{eqname}_profile_{suffix}.csv')
    pd.DataFrame(rows).to_csv(outpath,index=False)
    logger.info(f'Profile saved → {outpath}')
