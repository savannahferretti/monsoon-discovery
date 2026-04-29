#!/usr/bin/env python

import os
import ast
import json
import logging
import warnings
warnings.filterwarnings('ignore')
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
    - tuple[set[str]|None, list[str]]: selected equation names (or None for all), and
        list of splits for which to save predictions
    '''
    parser = argparse.ArgumentParser(description='Optimize SR equation constants on full train+valid data.')
    parser.add_argument('--equations',type=str,default='all',help='Comma-separated equation names to optimize, or `all`')
    parser.add_argument('--splits',type=str,default='train,valid,test',help='Comma-separated splits to generate predictions for (default: train,valid,test)')
    args  = parser.parse_args()
    selectedeqs = None if args.equations=='all' else {n.strip() for n in args.equations.split(',')}
    splits      = [s.strip() for s in args.splits.split(',')]
    return selectedeqs,splits

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

def eval_form(form,X,predictornames,constants):
    '''
    Purpose: Evaluate a form string given predictor values and constant values.
    Args:
    - form (str): Python expression string
    - X (pd.DataFrame): predictor feature matrix; must contain columns for all predictornames
    - predictornames (list[str]): predictor column names to extract from X
    - constants (dict): mapping from constant name to float value
    Returns:
    - np.ndarray: evaluated predictions with shape (nsamples,)
    '''
    ns = dict(SRFUNCTIONS,__builtins__={})
    for pname in predictornames:
        ns[pname] = X[pname].values
    ns.update(constants)
    out = eval(form,ns)
    if np.ndim(out)==0:
        out = np.full(len(X),float(out))
    return np.asarray(out,dtype=float)

def optimize_constants(form,predictornames,X,y,zmin,init):
    '''
    Purpose: Optimize named constants in an SR equation form via scipy L-BFGS-B.
    Args:
    - form (str): Python expression string using predictor names and constant names
    - predictornames (list[str]): predictor column names in X
    - X (pd.DataFrame): full train+valid feature matrix (valid samples only)
    - y (np.ndarray): z-scored log1p target for the same samples
    - zmin (float): z-scored floor corresponding to 0 mm precipitation
    - init (dict): initial constant values; constants absent from dict default to 1.0
    Returns:
    - tuple[dict, OptimizeResult]: optimized constants and scipy optimization result
    '''
    cnames = extract_constants(form,predictornames)
    x0     = np.array([init.get(c,1.0) for c in cnames])
    def objective(params):
        const = dict(zip(cnames,params))
        pred  = np.maximum(eval_form(form,X,predictornames,const),zmin)
        return float(np.nanmean((pred-y)**2))
    res = minimize(objective,x0,method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
    return dict(zip(cnames,res.x)),res

def multistart_optimize(form,predictornames,X,y,zmin,init,nrestarts=1,init_scale=5.0,seed=0,njobs=1):
    '''
    Purpose: Optimize named constants via L-BFGS-B with optional random restarts.
        The first restart uses `init`; subsequent restarts draw starting points uniformly
        from [-init_scale, init_scale], allowing the optimizer to discover the correct
        signs and magnitudes of each constant without manual initialization.
        Restarts are run in parallel across `njobs` threads.
    Args:
    - nrestarts (int): total number of optimization runs (default 1 = single run from init)
    - init_scale (float): half-range for uniform random initialization (default 5.0)
    - seed (int): RNG seed for reproducible random restarts
    - njobs (int): number of parallel threads for restarts (default 1)
    Returns:
    - tuple[dict, OptimizeResult]: best constants and corresponding scipy result
    '''
    cnames = extract_constants(form,predictornames)
    rng    = np.random.default_rng(seed)
    inits  = [init] + [
        {c:float(v) for c,v in zip(cnames,rng.uniform(-init_scale,init_scale,len(cnames)))}
        for _ in range(nrestarts-1)]
    results_list = Parallel(n_jobs=njobs,prefer='threads')(
        delayed(optimize_constants)(form,predictornames,X,y,zmin,init_i)
        for init_i in inits)
    best_const,best_res = None,None
    for i,(const,res) in enumerate(results_list):
        if best_res is None or res.fun < best_res.fun:
            best_const,best_res = const,res
        logger.debug(f'     restart {i+1}/{len(inits)}: loss={res.fun:.6f} converged={res.success}')
    return best_const,best_res

def log_seed_reference(runname,refcomplexity,config):
    '''
    Purpose: Log equations from each seed near refcomplexity to help the user choose initial constant values.
    Args:
    - runname (str): SR run name (e.g., 'sr_gauss_all')
    - refcomplexity (int | None): target complexity; equations within ±2 are printed
    - config (Config): project configuration object
    '''
    if refcomplexity is None:
        return
    seeds  = config.sr['seeds']
    outdir = os.path.join(config.modelsdir,'sr')
    logger.info(f'   Seed reference equations near complexity {refcomplexity}:')
    for seed in seeds:
        csvpath = os.path.join(outdir,f'{runname}_{seed}_equations.csv')
        if not os.path.exists(csvpath):
            logger.info(f'     seed {seed}: no CSV found')
            continue
        eqdf   = pd.read_csv(csvpath)
        nearby = eqdf[(eqdf['complexity']>=refcomplexity-2)&(eqdf['complexity']<=refcomplexity+2)]
        if nearby.empty:
            logger.info(f'     seed {seed}: no equation within ±2 of complexity {refcomplexity}')
            continue
        for _,row in nearby.sort_values('complexity').iterrows():
            logger.info(f'     seed {seed} | complexity {int(row["complexity"])} | loss {float(row["loss"]):.6f} | {row["equation"]}')

def save_constants(name,constants,trainloss,validloss,form,config):
    '''
    Purpose: Save optimized equation constants and training diagnostics to a JSON file.
    Args:
    - name (str): equation identifier used for the output filename
    - constants (dict): optimized constant name → float value
    - trainloss (float): MSE on the train+valid optimization set
    - validloss (float): MSE on the held-out validation set
    - form (str): the equation form string
    - config (Config): project configuration object
    '''
    outdir  = os.path.join(config.modelsdir,'sr')
    os.makedirs(outdir,exist_ok=True)
    outpath = os.path.join(outdir,f'{name}_optimized.json')
    payload = dict(name=name,form=form,constants=constants,train_loss=trainloss,valid_loss=validloss)
    with open(outpath,'w',encoding='utf-8') as f:
        json.dump(payload,f,indent=2)
    logger.info(f'   Saved optimized constants to {outpath}')

def predict_split(form,predictornames,constants,runconfig,config,writer,split,zmin):
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
    - zmin (float): z-scored floor corresponding to 0 mm precipitation
    Returns:
    - xr.Dataset: predictions in native units with dims (time, lat, lon)
    '''
    X,y,refda,validmask = load_data(split,runconfig,config)
    Xvalid = X[validmask][predictornames].reset_index(drop=True)
    pred   = np.maximum(eval_form(form,Xvalid,predictornames,constants),zmin)
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
    selectedeqs,splits = parse()

    njobs = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
    logger.info(f'Using {njobs} parallel worker(s) for multi-start optimization')

    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    zmin   = (0.0-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
    writer = PredictionWriter(config.splitsdir,targetvar=targetvar)

    # Cache loaded data keyed by runfrom to avoid redundant I/O when multiple
    # equations share the same source run (common when all use sr_gauss_all).
    data_cache = {}

    for name,eqspec in optimizedeqs.items():
        if selectedeqs is not None and name not in selectedeqs:
            continue
        jsonpath = os.path.join(config.modelsdir,'sr',f'{name}_optimized.json')
        if os.path.exists(jsonpath):
            logger.info(f'Skipping `{name}`, already optimized')
            continue
        runname        = eqspec['runfrom']
        runconfig      = sr['runs'][runname]
        predictornames = runconfig['fieldvars']+runconfig.get('localvars',[])
        form           = eqspec['form']
        init           = eqspec.get('constants',{})
        refcomplexity  = eqspec.get('refcomplexity')

        logger.info(f'Optimizing `{name}` (form: {form})...')
        log_seed_reference(runname,refcomplexity,config)

        if runname not in data_cache:
            logger.info(f'   Loading full train+valid features for `{runname}`...')
            Xtrain,ytrain,refdatrain,vmtrain = load_data('train',runconfig,config,time_offset=0)
            Xvalid,yvalid,_,vmvalid          = load_data('valid',runconfig,config,time_offset=int(refdatrain.sizes['time']))
            _Xfit = pd.concat([Xtrain[vmtrain],Xvalid[vmvalid]]).reset_index(drop=True)
            _yfit = np.concatenate([ytrain[vmtrain],yvalid[vmvalid]])
            data_cache[runname] = (_Xfit,_yfit,Xvalid,yvalid,vmvalid)
            del Xtrain,ytrain,refdatrain

        Xfit_full,yfit,Xvalid,yvalid,vmvalid = data_cache[runname]
        Xfit = Xfit_full[predictornames]

        nrestarts  = eqspec.get('nrestarts',1)
        init_scale = eqspec.get('init_scale',5.0)
        logger.info(f'   Running L-BFGS-B ({len(Xfit):,} samples, logz loss, {nrestarts} restart(s), {njobs} worker(s))...')
        optconstants,res = multistart_optimize(form,predictornames,Xfit,yfit,zmin,init,nrestarts,init_scale,njobs=njobs)

        trainloss  = float(res.fun)
        vpred      = np.maximum(eval_form(form,Xvalid[vmvalid][predictornames].reset_index(drop=True),predictornames,optconstants),zmin)
        validloss  = float(np.nanmean((vpred-yvalid[vmvalid])**2))

        logger.info(f'   Constants: {", ".join(f"{k}={v:.6f}" for k,v in optconstants.items())}')
        logger.info(f'   Train loss: {trainloss:.6f}   Valid loss: {validloss:.6f}   Converged: {res.success}')
        save_constants(name,{k:float(v) for k,v in optconstants.items()},trainloss,validloss,form,config)

        for split in splits:
            predpath = os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')
            if os.path.exists(predpath):
                logger.info(f'   Skipping {split} predictions, already exist')
                continue
            logger.info(f'   Generating {split} predictions...')
            predds = predict_split(form,predictornames,optconstants,runconfig,config,writer,split,zmin)
            writer.save(predds,name,'predictions',split,config.predsdir)
            del predds
