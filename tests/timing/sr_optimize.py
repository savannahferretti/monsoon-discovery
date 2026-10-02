#!/usr/bin/env python

import os
import logging
import argparse
import numpy as np
import pandas as pd
from timingutils import TimingConfig,parse_names,load_stats,restrict_kernel_seeds
from scripts.data.classes import PredictionWriter
from scripts.models.sr.train import load_data
from scripts.models.sr.optimize import extract_constants,eval_form,multistart_optimize,pysr_init,load_registry,save_registry,predict_split

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

NRESTARTS = 50

def get_init(name,eqspec,form,predictornames,config,registry,mainregistry):
    '''
    Purpose: Choose the primary initialization and extra anchor initializations for one equation.
        Primary: constants matched from the variant's PySR equations at the manuscript reference complexity;
        else (SR-ALL-PC) the variant's optimized SR-ALL constants; else the manuscript constants.
        The manuscript constants are always added as an extra start.
    Args:
    - name (str): equation name
    - eqspec (dict): equation specification from configs.json
    - form (str): equation form
    - predictornames (list[str]): predictor names
    - config (TimingConfig): variant configuration
    - registry (dict): variant registry
    - mainregistry (dict): registry of the current (manuscript) setup
    Returns:
    - tuple[dict, list[dict]]: primary init and extra inits
    '''
    constantnames = extract_constants(form,predictornames)
    manuscript    = {c:mainregistry[name]['constants'][c] for c in constantnames if c in mainregistry.get(name,{}).get('constants',{})}
    init = pysr_init(form,predictornames,eqspec.get('refcomplexity'),eqspec['runfrom'],eqspec.get('seeds',config.sr['seeds']),config.modelsdir)
    if init:
        logger.info(f'   PySR init: {", ".join(f"{k}={v:.4f}" for k,v in init.items())}')
    elif name=='sr_all_pc_eq' and 'sr_all_eq' in registry:
        srall = registry['sr_all_eq']['constants']
        init  = {'c12':srall['c9'],'c13':srall['c10']}
        logger.info(f'   Init from optimized SR-ALL: {init}')
    else:
        init = dict(manuscript)
        logger.info(f'   No PySR match; init from manuscript constants: {init}')
    extra = [manuscript] if manuscript and manuscript!=init else []
    for prevname,preventry in registry.items():
        if config.sr['optimizedeqs'].get(prevname,{}).get('runfrom')!=eqspec['runfrom']:
            continue
        prevconsts = preventry['constants']
        if set(prevconsts.keys())<set(constantnames):
            extra.append({c:(prevconsts[c] if c in prevconsts else 1.0) for c in constantnames})
    return init,extra

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Optimize constants of the manuscript equation forms for the timing variants.')
    parser.add_argument('--variants',type=str,default='all',help='Comma-separated variant names, or `all`')
    parser.add_argument('--equations',type=str,default='all',help='Comma-separated equation names, or `all`')
    parser.add_argument('--splits',type=str,default='test',help='Comma-separated splits to predict (default: test)')
    args = parser.parse_args()
    nworkers   = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
    baseconfig = TimingConfig()
    variants   = parse_names(args.variants,list(baseconfig.timing['variants']))
    mainregistry = load_registry(baseconfig)
    splits     = [s.strip() for s in args.splits.split(',')]
    for variant in variants:
        config = TimingConfig(variant)
        restrict_kernel_seeds(config)
        stats  = load_stats(config)
        zmin   = (0.0-stats[f'{config.targetvar}_mean'])/stats[f'{config.targetvar}_std']
        writer = PredictionWriter(config.splitsdir,targetvar=config.targetvar)
        registry  = load_registry(config)
        datacache = {}
        for name in parse_names(args.equations,list(config.srequations)):
            eqspec    = config.srequations[name]
            form      = eqspec['form']
            runname   = eqspec['runfrom']
            runconfig = config.sr['runs'][runname]
            if name in registry:
                logger.info(f'[{variant}] `{name}` already optimized')
            else:
                residualfrom = runconfig.get('residualfrom')
                if residualfrom and residualfrom not in registry:
                    logger.error(f'[{variant}] `{name}` needs `{residualfrom}` optimized first, skipping')
                    continue
                logger.info(f'[{variant}] Optimizing `{name}`: {form}')
                if runname not in datacache:
                    xtrain,ytrain,reftrain,trainmask = load_data('train',runconfig,config,time_offset=0)
                    xvalid,yvalid,_,validmask        = load_data('valid',runconfig,config,time_offset=int(reftrain.sizes['time']))
                    xfit = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
                    yfit = np.concatenate([ytrain[trainmask],yvalid[validmask]])
                    datacache[runname] = (xfit,yfit,xvalid,yvalid,validmask)
                    del xtrain,ytrain,reftrain
                xfitfull,yfit,xvalid,yvalid,validmask = datacache[runname]
                predictornames = [c for c in xfitfull.columns if c!='timeidx']
                xfit       = xfitfull[predictornames].astype(np.float64)
                init,extra = get_init(name,eqspec,form,predictornames,config,registry,mainregistry)
                logger.info(f'   L-BFGS-B with {len(xfit):,} samples, {NRESTARTS} starts ({len(extra)} extra), {nworkers} worker(s)...')
                constants,res = multistart_optimize(form,predictornames,xfit,yfit,zmin,init,NRESTARTS,eqspec.get('initscale',5.0),nworkers=nworkers,extra_inits=extra)
                constants = {k:round(float(v),2) for k,v in constants.items()}
                xvalidsub = xvalid[validmask][predictornames].reset_index(drop=True)
                trainloss = float(np.mean((zmin+np.maximum(eval_form(form,xfit,predictornames,constants),0.0)-yfit)**2))
                validloss = float(np.mean((zmin+np.maximum(eval_form(form,xvalidsub,predictornames,constants),0.0)-yvalid[validmask])**2))
                logger.info(f'   Rounded constants: {constants} | train+valid loss = {trainloss:.6f} | valid loss = {validloss:.6f} | converged = {res.success}')
                registry[name] = dict(form=form,constants=constants,train_loss=trainloss,valid_loss=validloss)
                save_registry(registry,config)
                registry = load_registry(config)
            for split in splits:
                if os.path.exists(os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')):
                    continue
                logger.info(f'   Generating {split} predictions for `{name}`...')
                predds = predict_split(form,registry[name]['constants'],runconfig,config,writer,split,zmin)
                writer.save(predds,name,'predictions',split,config.predsdir)
                del predds
