#!/usr/bin/env python

import os
import ast
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scripts.utils import Config
from scripts.data.classes import PredictionWriter
from scripts.models.sr.train import load_data

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

SRFUNCTIONS = {
    'cube':   lambda x: x**3,
    'square': lambda x: x**2,
    'neg':    lambda x: -x,
    'sqrt':   np.sqrt,
    'exp':    np.exp,
    'log':    np.log,
    'abs':    np.abs,
    'sin':    np.sin,
    'cos':    np.cos,
}

def parse():
    '''
    Purpose: Parse command-line arguments for running the optimization script.
    Returns:
    - tuple[set[str]|None, list[str], int]: selected equation names (or None for all),
        list of splits for which to save predictions, and number of parallel workers
    '''
    parser = argparse.ArgumentParser(description='Optimize SR equation constants on full train+valid data.')
    parser.add_argument('--equations',type=str,default='all',help='Comma-separated equation names to optimize, or `all`')
    parser.add_argument('--splits',type=str,default='train,valid,test',help='Comma-separated splits to generate predictions for (default: train,valid,test)')
    args        = parser.parse_args()
    selectedeqs = None if args.equations=='all' else {n.strip() for n in args.equations.split(',')}
    splits      = [s.strip() for s in args.splits.split(',')]
    nworkers    = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
    return selectedeqs,splits,nworkers

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
    out = eval(form,ns)
    if np.ndim(out)==0:
        out = np.full(len(x),float(out))
    return np.asarray(out,dtype=float)

def optimize_constants(form,predictornames,x,y,zfloor,init):
    '''
    Purpose: Optimize named constants in an SR equation form via scipy L-BFGS-B.
        The objective is the clipped MSE in z-scored logz space, consistent with
        neural network training. Predictions are clipped at zfloor before computing
        the loss, matching the floor applied during NN training and in predict_split.
    Args:
    - form (str): Python expression string using predictor names and constant names
    - predictornames (list[str]): predictor column names in x
    - x (pd.DataFrame): full train+valid feature matrix (valid samples only)
    - y (np.ndarray): z-scored log1p target for the same samples
    - zfloor (float): z-scored floor corresponding to 0 mm precipitation
    - init (dict): initial constant values; constants absent from dict default to 1.0
    Returns:
    - tuple[dict, OptimizeResult]: optimized constants and scipy optimization result
    '''
    constantnames = extract_constants(form,predictornames)
    initialparams = np.array([init.get(c,1.0) for c in constantnames])
    def objective(params):
        constants = dict(zip(constantnames,params))
        pred      = np.maximum(eval_form(form,x,predictornames,constants),zfloor)
        return float(np.nanmean((pred-y)**2))
    res = minimize(objective,initialparams,method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
    return dict(zip(constantnames,res.x)),res

def multistart_optimize(form,predictornames,x,y,zfloor,init,nrestarts=1,initscale=5.0,seed=0,nworkers=1):
    '''
    Purpose: Optimize named constants via L-BFGS-B with optional random restarts.
        The first restart uses `init`; subsequent restarts draw starting points uniformly
        from [-initscale, initscale], allowing the optimizer to discover the correct
        signs and magnitudes of each constant without manual initialization.
        Restarts are run in parallel across `nworkers` threads.
    Args:
    - form (str): Python expression string using predictor names and constant names
    - predictornames (list[str]): predictor column names in x
    - x (pd.DataFrame): full train+valid feature matrix (valid samples only)
    - y (np.ndarray): z-scored log1p target for the same samples
    - zfloor (float): z-scored floor corresponding to 0 mm precipitation
    - init (dict): initial constant values for the first restart
    - nrestarts (int): total number of optimization runs (default 1 = single run from init)
    - initscale (float): half-range for uniform random initialization (default 5.0)
    - seed (int): RNG seed for reproducible random restarts
    - nworkers (int): number of parallel threads for restarts (default 1)
    Returns:
    - tuple[dict, OptimizeResult]: best constants and corresponding scipy result
    '''
    constantnames = extract_constants(form,predictornames)
    rng           = np.random.default_rng(seed)
    inits         = [init] + [
        {c:float(v) for c,v in zip(constantnames,rng.uniform(-initscale,initscale,len(constantnames)))}
        for _ in range(nrestarts-1)]
    resultslist = Parallel(n_jobs=nworkers,prefer='threads')(
        delayed(optimize_constants)(form,predictornames,x,y,zfloor,restartinit)
        for restartinit in inits)
    bestconstants,bestresult = None,None
    for i,(constants,res) in enumerate(resultslist):
        if bestresult is None or res.fun < bestresult.fun:
            bestconstants,bestresult = constants,res
        logger.debug(f'     restart {i+1}/{len(inits)}: loss={res.fun:.6f} converged={res.success}')
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
    rows = [
        dict(name=name,form=entry['form'],
             train_loss=entry['train_loss'],valid_loss=entry['valid_loss'],
             constants=json.dumps(entry['constants']))
        for name,entry in registry.items()
    ]
    pd.DataFrame(rows).to_csv(registrycsvpath,index=False)
    logger.info(f'   Registry saved ({len(registry)} equation(s)) → {registrypath}')

def predict_split(form,predictornames,constants,runconfig,config,writer,split,zfloor):
    '''
    Purpose: Generate native-unit gridded predictions for a data split using an optimized equation.
    Args:
    - form (str): optimized equation form string
    - predictornames (list[str]): predictor column names
    - constants (dict): optimized constant values
    - runconfig (dict): SR run config specifying which features to load
    - config (Config): project configuration object
    - writer (PredictionWriter): used for denormalization and saving
    - split (str): 'train' | 'valid' | 'test'
    - zfloor (float): z-scored floor corresponding to 0 mm precipitation
    Returns:
    - xr.Dataset: predictions in native units with dims (time, lat, lon)
    '''
    x,y,refda,validmask = load_data(split,runconfig,config)
    xvalid = x[validmask][predictornames].reset_index(drop=True)
    pred   = np.maximum(eval_form(form,xvalid,predictornames,constants),zfloor)
    grid   = np.maximum(np.expm1(writer.unflatten(pred,validmask,refda)*writer.std+writer.mean),0.0).astype(np.float32)
    da     = xr.DataArray(grid,dims=refda.dims,coords=refda.coords)
    da.attrs = dict(long_name=writer.longname,units=writer.units)
    return da.to_dataset(name=writer.targetvar)

if __name__=='__main__':
    config       = Config()
    sr           = config.sr
    targetvar    = config.targetvar
    optimizedeqs = sr.get('optimizedeqs',{})
    logger.info('Spinning up...')
    selectedeqs,splits,nworkers = parse()
    logger.info(f'Using {nworkers} parallel worker(s) for multi-start optimization')
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    zfloor = (0.0-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
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
        if name in registry:
            logger.info(f'Skipping `{name}`, already optimized')
            continue
        runname        = eqspec['runfrom']
        runconfig      = sr['runs'][runname]
        predictornames = runconfig['fieldvars']+runconfig.get('localvars',[])
        form           = eqspec['form']
        refcomplexity  = eqspec.get('refcomplexity')
        logger.info(f'Optimizing `{name}` (form: {form}, refcomplexity: {refcomplexity})...')
        if runname not in datacache:
            logger.info(f'   Loading full train+valid features for `{runname}`...')
            xtrain,ytrain,reftrain,trainmask = load_data('train',runconfig,config,time_offset=0)
            xvalid,yvalid,_,validmask        = load_data('valid',runconfig,config,time_offset=int(reftrain.sizes['time']))
            xfit  = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
            yfit  = np.concatenate([ytrain[trainmask],yvalid[validmask]])
            datacache[runname] = (xfit,yfit,xvalid,yvalid,validmask)
            del xtrain,ytrain,reftrain
        xfitfull,yfit,xvalid,yvalid,validmask = datacache[runname]
        xfit       = xfitfull[predictornames]
        nrestarts  = eqspec.get('nrestarts',1)
        initscale  = eqspec.get('initscale',5.0)
        logger.info(f'   Running L-BFGS-B ({len(xfit):,} samples, logz loss, {nrestarts} restart(s), {nworkers} worker(s))...')
        constants,res = multistart_optimize(form,predictornames,xfit,yfit,zfloor,{},nrestarts,initscale,nworkers=nworkers)
        trainloss  = float(res.fun)
        validpred  = np.maximum(eval_form(form,xvalid[validmask][predictornames].reset_index(drop=True),predictornames,constants),zfloor)
        validloss  = float(np.nanmean((validpred-yvalid[validmask])**2))
        logger.info(f'   Constants: {", ".join(f"{k}={v:.6f}" for k,v in constants.items())}')
        logger.info(f'   Train loss: {trainloss:.6f}   Valid loss: {validloss:.6f}   Converged: {res.success}')
        registry[name] = dict(form=form,constants={k:float(v) for k,v in constants.items()},
                              train_loss=trainloss,valid_loss=validloss)
        save_registry(registry,config)
        for split in splits:
            predpath = os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')
            if os.path.exists(predpath):
                logger.info(f'   Skipping {split} predictions, already exist')
                continue
            logger.info(f'   Generating {split} predictions...')
            predds = predict_split(form,predictornames,constants,runconfig,config,writer,split,zfloor)
            writer.save(predds,name,'predictions',split,config.predsdir)
            del predds
