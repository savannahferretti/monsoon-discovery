#!/usr/bin/env python

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from scipy.optimize import minimize
from timingutils import TimingConfig,parse_names,load_stats,restrict_kernel_seeds,calc_zmin,load_dataset
from data import load_features,load_physical,unflatten,save_predictions
from equations import extract_constants,evaluate,raw_to_precip,round_constants,load_registry,save_registry,calc_physical_constants,calc_physical_precip
from scripts.models.sr.optimize import pysr_init

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

NRESTARTS = 50
INITSCALE = 5.0

def calc_loss(form,columns,y,zmin,constants):
    '''
    Purpose: MSE of zmin + max(raw, 0) against the standardized target (same objective as the NNs and PySR).
    Args:
    - form (str): equation form
    - columns (dict[str, np.ndarray]): standardized predictors
    - y (np.ndarray): standardized target
    - zmin (float): standardized value of zero precipitation
    - constants (dict[str, float]): constants
    Returns:
    - float: loss
    '''
    return float(np.mean((zmin+np.maximum(evaluate(form,columns,constants),0.0)-y)**2))

def multistart_optimize(form,columns,y,zmin,inits,nworkers):
    '''
    Purpose: L-BFGS-B from each initialization; return the best result.
    Args:
    - form (str): equation form
    - columns (dict[str, np.ndarray]): standardized predictors
    - y (np.ndarray): standardized target
    - zmin (float): standardized value of zero precipitation
    - inits (list[dict[str, float]]): initial constants
    - nworkers (int): parallel threads
    Returns:
    - tuple[dict[str, float], OptimizeResult]: best constants and result
    '''
    names = list(inits[0])
    def run(init):
        res = minimize(lambda params:calc_loss(form,columns,y,zmin,dict(zip(names,params))),np.array([init[c] for c in names]),
                       method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
        return dict(zip(names,res.x)),res
    results = Parallel(n_jobs=nworkers,prefer='threads')(delayed(run)(init) for init in inits)
    return min(results,key=lambda result:result[1].fun)

def get_inits(name,eqspec,constantnames,predictornames,config,registry,mainregistry):
    '''
    Purpose: Initial constants: PySR constants of the variant's structure at the reference complexity (if it
        matches), the variant's optimized SR-ALL constants (for SR-ALL-PC), the manuscript constants, earlier
        equations from the same run, and uniform random draws in [-INITSCALE, INITSCALE] up to NRESTARTS.
    Args:
    - name (str): equation name
    - eqspec (dict): equation specification
    - constantnames (list[str]): constant names
    - predictornames (list[str]): predictor names
    - config (TimingConfig): variant configuration
    - registry (dict): variant registry
    - mainregistry (dict): current (manuscript) registry
    Returns:
    - list[dict[str, float]]: initializations
    '''
    inits = []
    pysr  = pysr_init(eqspec['form'],predictornames,eqspec.get('refcomplexity'),eqspec['runfrom'],eqspec.get('seeds',config.sr['seeds']),config.modelsdir)
    if pysr:
        inits.append(pysr)
    if name=='sr_all_pc_eq' and 'sr_all_eq' in registry:
        inits.append({'c12':registry['sr_all_eq']['constants']['c9'],'c13':registry['sr_all_eq']['constants']['c10']})
    manuscript = mainregistry.get(name,{}).get('constants',{})
    if set(constantnames)<=set(manuscript):
        inits.append({c:manuscript[c] for c in constantnames})
    for prevname,preventry in registry.items():
        if config.sr['optimizedeqs'].get(prevname,{}).get('runfrom')==eqspec['runfrom'] and set(preventry['constants'])<set(constantnames):
            inits.append({c:preventry['constants'].get(c,1.0) for c in constantnames})
    rng = np.random.default_rng(0)
    while len(inits)<NRESTARTS:
        inits.append(dict(zip(constantnames,rng.uniform(-INITSCALE,INITSCALE,len(constantnames)))))
    inits = [{c:float(init[c]) for c in constantnames} for init in inits]
    logger.info(f'   {len(inits)} starts; first: {inits[0]}')
    return inits

def check_consistency(config,name,split,registry,stats):
    '''
    Purpose: Recompute predictions from the physical-space form and the physical constants, and compare with the
        standardized-space predictions (both float64) and with the saved float32 file.
    Args:
    - config (TimingConfig): configuration object
    - name (str): equation name
    - split (str): split name
    - registry (dict): variant registry
    - stats (dict): training statistics
    Returns:
    - dict[str, float]: maximum absolute differences (mm)
    '''
    runconfig = config.sr['runs'][config.sr['optimizedeqs'][name]['runfrom']]
    fieldvars = ['bl'] if name=='sr_bl_eq' else ['rh','thetae','thetaestar']
    localvars = [] if name=='sr_bl_eq' else ['lf','shf','lhf']
    weightsfrom = None if name=='sr_bl_eq' else config.sr['runs'][config.sr['optimizedeqs']['sr_atm_eq']['runfrom']]['weightsfrom']
    inputs,_,_  = load_physical(config,split,fieldvars,localvars,weightsfrom)
    physical  = calc_physical_constants(name,registry,stats)
    physprecip = calc_physical_precip(name,physical,inputs,stats)
    x,_,_,valid = load_features(config,split,runconfig)
    stdprecip = raw_to_precip(evaluate(registry[name]['form'],{c:x[c].values for c in x.columns if c!='timeidx'},registry[name]['constants']),stats)
    saved     = load_dataset(os.path.join(config.predsdir,f'{name}_{split}_predictions.nc'))['tp'].transpose('time','lat','lon').values.ravel()
    return {'physical_vs_standardized':float(np.nanmax(np.abs(physprecip[valid]-stdprecip[valid]))),
            'physical_vs_saved':float(np.nanmax(np.abs(physprecip[valid]-saved[valid]))),
            'float32_step_at_max':float(np.spacing(np.float32(np.nanmax(saved[valid])))),
            'physical':physical}

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Optimize constants of the manuscript equation forms for the timing variants.')
    parser.add_argument('--variants',type=str,default='all',help='Comma-separated variant names, or `all`')
    parser.add_argument('--equations',type=str,default='all',help='Comma-separated equation names, or `all`')
    parser.add_argument('--splits',type=str,default='test',help='Comma-separated splits to predict (default: test)')
    args = parser.parse_args()
    nworkers     = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
    baseconfig   = TimingConfig()
    sigfigs      = baseconfig.timing['constantsigfigs']
    variants     = parse_names(args.variants,list(baseconfig.timing['variants']))
    mainregistry = load_registry(baseconfig)
    splits       = [s.strip() for s in args.splits.split(',')]
    for variant in variants:
        config = TimingConfig(variant)
        restrict_kernel_seeds(config)
        stats  = load_stats(config)
        zmin   = calc_zmin(stats)
        registry = load_registry(config)
        for name in parse_names(args.equations,list(config.srequations)):
            eqspec    = config.srequations[name]
            form      = eqspec['form']
            runconfig = config.sr['runs'][eqspec['runfrom']]
            if name not in registry:
                residualfrom = runconfig.get('residualfrom')
                if residualfrom and residualfrom not in registry:
                    logger.error(f'[{variant}] `{name}` needs `{residualfrom}` optimized first, skipping')
                    continue
                logger.info(f'[{variant}] Optimizing `{name}`: {form}')
                xtrain,ytrain,reftrain,trainmask = load_features(config,'train',runconfig)
                xvalid,yvalid,_,validmask        = load_features(config,'valid',runconfig,timeoffset=int(reftrain.sizes['time']))
                predictornames = [c for c in xtrain.columns if c!='timeidx']
                fitcols   = {c:np.concatenate([xtrain[c].values[trainmask],xvalid[c].values[validmask]]) for c in predictornames}
                validcols = {c:xvalid[c].values[validmask] for c in predictornames}
                yfit,yval = np.concatenate([ytrain[trainmask],yvalid[validmask]]),yvalid[validmask]
                del xtrain,xvalid
                constantnames = extract_constants(form,predictornames)
                inits = get_inits(name,eqspec,constantnames,predictornames,config,registry,mainregistry)
                logger.info(f'   L-BFGS-B with {len(yfit):,} samples, {nworkers} worker(s)...')
                constants,res = multistart_optimize(form,fitcols,yfit,zmin,inits,nworkers)
                logger.info(f'   Optimized constants: {constants} | converged = {res.success}')
                constants = round_constants(constants,sigfigs)
                trainloss = calc_loss(form,fitcols,yfit,zmin,constants)
                validloss = calc_loss(form,validcols,yval,zmin,constants)
                logger.info(f'   Rounded ({sigfigs} significant figures): {constants} | train+valid loss = {trainloss:.6f} | valid loss = {validloss:.6f}')
                registry[name] = dict(form=form,constants=constants,train_loss=trainloss,valid_loss=validloss)
                save_registry(registry,config)
                registry = load_registry(config)
            for split in splits:
                if not os.path.exists(os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')):
                    logger.info(f'   Generating {split} predictions for `{name}`...')
                    x,_,truth,valid = load_features(config,split,runconfig)
                    raw = evaluate(form,{c:x[c].values[valid] for c in x.columns if c!='timeidx'},registry[name]['constants'])
                    save_predictions(config,name,split,unflatten(raw_to_precip(raw,stats),valid,truth),truth)
                check = check_consistency(config,name,split,registry,stats)
                logger.info(f'   {split} consistency (mm): physical vs standardized = {check["physical_vs_standardized"]:.2e}, physical vs saved = {check["physical_vs_saved"]:.2e} (float32 step at max = {check["float32_step_at_max"]:.2e})')
                os.makedirs(config.resultsdir,exist_ok=True)
                with open(os.path.join(config.resultsdir,f'{variant}_{name}_{split}_constants.json'),'w',encoding='utf-8') as f:
                    json.dump({'standardized':registry[name]['constants'],**check},f,indent=2)
