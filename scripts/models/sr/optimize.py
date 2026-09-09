#!/usr/bin/env python

import os
import ast
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import sympy as sp
import xarray as xr
from joblib import Parallel,delayed
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
from scripts.utils import Config
from scripts.data.classes import PredictionWriter
from scripts.models.sr.train import load_data

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SRFUNCTIONS = {
    'cube':lambda x:x**3,
    'square': lambda x:x**2,
    'neg':lambda x:-x,
    'sqrt':np.sqrt,
    'exp':np.exp,
    'log':np.log,
    'abs':np.abs,
    'sin':np.sin,
    'cos':np.cos,
    'max':np.maximum,
    'min':np.minimum,
    '_safepow':lambda a,b: np.abs(a)**b}

def _prepare_form(form):
    import re
    return re.sub(r'(\w+)\^(\w+)',r'_safepow(\1,\2)',form)

def parse():
    parser = argparse.ArgumentParser(description='Optimize SR equation constants on full train+valid data.')
    parser.add_argument('--equations',type=str,default='all',help='Comma-separated equation names to optimize, or `all`')
    parser.add_argument('--splits',type=str,default='train,valid,test',help='Comma-separated splits to generate predictions for (default: train,valid,test)')
    parser.add_argument('--force',action='store_true',help='Re-optimize equations even if already in the registry')
    args        = parser.parse_args()
    selectedeqs = None if args.equations=='all' else {n.strip() for n in args.equations.split(',')}
    splits      = [s.strip() for s in args.splits.split(',')]
    nworkers    = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
    return selectedeqs,splits,nworkers,args.force

def extract_constants(form,predictornames):
    '''
    Purpose: Return sorted list of named constants in the form string — identifiers that are
        neither predictor names nor SR function names.
    Args:
    - form (str): Python expression string (e.g., 'a * (thetae + b * thetaestar + c)')
    - predictornames (list[str]): predictor variable names that appear in the form
    Returns:
    - list[str]: sorted constant names
    '''
    names = {node.id for node in ast.walk(ast.parse(form,mode='eval'))
             if isinstance(node,ast.Name)}
    return sorted(names - set(predictornames) - set(SRFUNCTIONS) - {'True','False','None'})

def eval_form(form,x,predictornames,constants):
    '''
    Purpose: Evaluate a form string given predictor values and constant values.
    Args:
    - form (str): Python expression string
    - x (pd.DataFrame): predictor feature matrix; must contain columns for all predictornames
    - predictornames (list[str]): predictor column names to extract from x
    - constants (dict): mapping from constant name to float value
    Returns:
    - np.ndarray: evaluated predictions with shape (nsamples,)
    '''
    ns = dict(SRFUNCTIONS,__builtins__={})
    for pname in predictornames:
        ns[pname] = x[pname].values
    ns.update(constants)
    out = eval(_prepare_form(form),ns)
    if np.ndim(out)==0:
        out = np.full(len(x),float(out))
    return np.asarray(out,dtype=float)

def optimize_constants(form,predictornames,x,y,zmin,zmax,init):
    constantnames = extract_constants(form,predictornames)
    initialparams = np.array([init[c] for c in constantnames])
    bounds        = [(-10.0,10.0) for _ in constantnames]
    def objective(params):
        constants = dict(zip(constantnames,params))
        raw       = eval_form(form,x,predictornames,constants)
        pred      = np.clip(zmin+np.maximum(raw,0.0),None,zmax)
        return float(np.mean((pred-y)**2))
    res = minimize(objective,initialparams,method='L-BFGS-B',bounds=bounds,
                   options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
    return dict(zip(constantnames,res.x)),res

def multistart_optimize(form,predictornames,x,y,zmin,zmax,nrestarts,seed=0,nworkers=1):
    '''
    Purpose: Optimize constants via L-BFGS-B from many starting points sampled
        uniformly across [-10, 10] using Latin Hypercube Sampling for thorough
        coverage of the search space.
    '''
    constantnames = extract_constants(form,predictornames)
    nconstants    = len(constantnames)
    sampler       = LatinHypercube(d=nconstants,seed=seed)
    samples       = sampler.random(n=nrestarts)
    samples       = samples*20.0-10.0
    inits = [{c:float(samples[i,j]) for j,c in enumerate(constantnames)}
             for i in range(nrestarts)]
    resultslist = Parallel(n_jobs=nworkers,prefer='threads')(
        delayed(optimize_constants)(form,predictornames,x,y,zmin,zmax,restartinit)
        for restartinit in inits)
    bestconstants,bestresult = None,None
    nconverged = 0
    for i,(constants,res) in enumerate(resultslist):
        if res.success:
            nconverged += 1
        if bestresult is None or res.fun < bestresult.fun:
            bestconstants,bestresult = constants,res
        logger.debug(f'     restart {i+1}/{nrestarts}: loss={res.fun:.6f} converged={res.success}')
    logger.info(f'   {nconverged}/{nrestarts} restarts converged; best loss={bestresult.fun:.6f}')
    return bestconstants,bestresult

def save_registry(registry,config):
    '''
    Purpose: Save the full optimized-equations registry as a single PKL and CSV.
    Args:
    - registry (dict): mapping name → {form, constants, train_loss, valid_loss}
    - config (Config): project configuration object
    '''
    outdir      = os.path.join(config.modelsdir,'sr')
    os.makedirs(outdir,exist_ok=True)
    registrypath    = os.path.join(outdir,'optimized_equations.pkl')
    registrycsvpath = os.path.join(outdir,'optimized_equations.csv')
    with open(registrypath,'wb') as f:
        pickle.dump(registry,f)
    rows = [dict(name=name,form=entry['form'],train_loss=entry['train_loss'],valid_loss=entry['valid_loss'],
                 constants=json.dumps(entry['constants'])) for name,entry in registry.items()]
    pd.DataFrame(rows).to_csv(registrycsvpath,index=False)
    logger.info(f'   Registry saved ({len(registry)} equation(s)) → {registrypath}')

def predict_split(form,predictornames,constants,runconfig,config,writer,split,zmin,zmax):
    x,y,refda,validmask = load_data(split,runconfig,config)
    xvalid = x[validmask][predictornames].reset_index(drop=True)
    raw    = eval_form(form,xvalid,predictornames,constants)
    pred   = np.clip(zmin+np.maximum(raw,0.0),None,zmax)
    from scripts.data.classes.writer import PMAX
    grid   = np.clip(np.expm1(writer.unflatten(pred,validmask,refda)*writer.std+writer.mean),0.0,PMAX).astype(np.float32)
    da     = xr.DataArray(grid,dims=refda.dims,coords=refda.coords)
    da.attrs = dict(long_name=writer.longname,units=writer.units)
    return da.to_dataset(name=writer.targetvar)

if __name__=='__main__':
    config       = Config()
    sr           = config.sr
    targetvar    = config.targetvar
    optimizedeqs = sr.get('optimizedeqs',{})
    logger.info('Spinning up...')
    selectedeqs,splits,nworkers,force = parse()
    logger.info(f'Using {nworkers} parallel worker(s) for multi-start optimization...')
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    from scripts.data.classes.writer import PMAX
    zmin = (0.0-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
    zmax = (np.log1p(PMAX)-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
    writer = PredictionWriter(config.splitsdir,targetvar=targetvar)
    registrypath = os.path.join(config.modelsdir,'sr','optimized_equations.pkl')
    registry = {}
    if os.path.exists(registrypath):
        with open(registrypath,'rb') as f:
            registry = pickle.load(f)
        logger.info(f'Loaded existing registry with {len(registry)} equation(s)')
    datacache = {}
    for name,eqspec in optimizedeqs.items():
        if selectedeqs is not None and name not in selectedeqs:
            continue
        if eqspec.get('form') is None:
            logger.info(f'Skipping `{name}`, form not yet specified')
            continue
        if name in registry and not force:
            logger.info(f'Skipping `{name}`, already optimized (use --force to re-optimize)')
            continue
        if name in registry and force:
            logger.info(f'Re-optimizing `{name}` (--force)')
            del registry[name]
        runname        = eqspec['runfrom']
        runconfig      = sr['runs'][runname]
        form           = eqspec['form']
        logger.info(f'Optimizing `{name}`...')
        if runname not in datacache:
            logger.info(f'   Loading training + validation sets...')
            xtrain,ytrain,reftrain,trainmask = load_data('train',runconfig,config,time_offset=0)
            xvalid,yvalid,_,validmask        = load_data('valid',runconfig,config,time_offset=int(reftrain.sizes['time']))
            xfit  = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
            yfit  = np.concatenate([ytrain[trainmask],yvalid[validmask]])
            datacache[runname] = (xfit,yfit,xvalid,yvalid,validmask)
            del xtrain,ytrain,reftrain
        xfitfull,yfit,xvalid,yvalid,validmask = datacache[runname]
        predictornames = [c for c in xfitfull.columns if c != 'timeidx']
        xfit           = xfitfull[predictornames]
        nrestarts      = eqspec.get('nrestarts',50)
        constantnames  = extract_constants(form,predictornames)
        logger.info(f'   Form: {form}')
        logger.info(f'   Constants to optimize: {constantnames}')
        logger.info(f'   Running L-BFGS-B with {len(xfit):,} samples, {nrestarts} restart(s) '
                    f'(LHS over [-10,10]), {nworkers} worker(s)...')
        constants,res = multistart_optimize(form,predictornames,xfit,yfit,zmin,zmax,
                                            nrestarts,nworkers=nworkers)
        trainloss  = float(res.fun)
        xvalidsub  = xvalid[validmask][predictornames].reset_index(drop=True)
        validtgt   = yvalid[validmask]
        validpred  = np.clip(zmin+np.maximum(eval_form(form,xvalidsub,predictornames,constants),0.0),None,zmax)
        validloss  = float(np.mean((validpred-validtgt)**2))
        logger.info(f'   Constants: {", ".join(f"{k}={v:.6f}" for k,v in constants.items())}')
        logger.info(f'   Training Loss: {trainloss:.6f} | Validation Loss: {validloss:.6f} | Converged: {res.success}')
        constants  = {k:round(float(v),2) for k,v in constants.items()}
        trainpred  = np.clip(zmin+np.maximum(eval_form(form,xfit,predictornames,constants),0.0),None,zmax)
        validpred  = np.clip(zmin+np.maximum(eval_form(form,xvalidsub,predictornames,constants),0.0),None,zmax)
        trainloss  = float(np.mean((trainpred-yfit)**2))
        validloss  = float(np.mean((validpred-validtgt)**2))
        logger.info(f'   Rounded constants: {", ".join(f"{k}={v:.2f}" for k,v in constants.items())}')
        logger.info(f'   Rounded Training Loss: {trainloss:.6f} | Rounded Validation Loss: {validloss:.6f}')
        registry[name] = dict(form=form,constants=constants,
                              train_loss=trainloss,valid_loss=validloss)
        save_registry(registry,config)
        for split in splits:
            predpath = os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')
            if os.path.exists(predpath) and not force:
                logger.info(f'   Skipping {split} predictions, already exist (use --force to regenerate)')
                continue
            logger.info(f'   Generating {split} predictions...')
            predds = predict_split(form,predictornames,constants,runconfig,config,writer,split,zmin,zmax)
            writer.save(predds,name,'predictions',split,config.predsdir)
            del predds
