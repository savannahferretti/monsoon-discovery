#!/usr/bin/env python

import os
import logging
import argparse
import numpy as np
import pandas as pd
import sympy as sp
import xarray as xr
from joblib import Parallel,delayed
from scipy.optimize import minimize
from scripts.utils import Config,load_stats
from scripts.data.classes import PredictionWriter
from scripts.models.sr.train import load_features
from scripts.models.sr.equations import extract_constants,evaluate,raw_to_precip,round_constants,load_registry,save_registry,calc_physical_constants,calc_physical_precip

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse():
    '''
    Purpose: Parse command-line arguments.
    Returns:
    - tuple[set[str] | None, list[str], int, bool]: equations to optimize (None for all), splits to predict, number of
        parallel workers, and whether to only predict from existing constants
    '''
    parser = argparse.ArgumentParser(description='Optimize SR equation constants on full train+valid data.')
    parser.add_argument('--equations',type=str,default='all',help='Comma-separated equation names to optimize, or `all`')
    parser.add_argument('--splits',type=str,default='train,valid,test',help='Comma-separated splits to generate predictions for (default: train,valid,test)')
    parser.add_argument('--predict-only',action='store_true',help='Skip optimization; generate predictions from existing constants in optimized_equations.csv')
    args        = parser.parse_args()
    selectedeqs = None if args.equations=='all' else {n.strip() for n in args.equations.split(',')}
    splits      = [s.strip() for s in args.splits.split(',')]
    nworkers    = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
    return selectedeqs,splits,nworkers,args.predict_only

def calc_loss(form,columns,y,zmin,constants):
    '''
    Purpose: MSE of zmin + max(z, 0) against the standardized target, the same objective as the NNs and PySR.
    Args:
    - form (str): equation form
    - columns (dict[str,np.ndarray]): standardized predictors
    - y (np.ndarray): standardized target
    - zmin (float): standardized value of zero precipitation
    - constants (dict[str,float]): constants
    Returns:
    - float: loss
    '''
    return float(np.mean((zmin+np.maximum(evaluate(form,columns,constants),0.0)-y)**2))

def multistart_optimize(form,columns,y,zmin,inits,nworkers):
    '''
    Purpose: Run L-BFGS-B from each set of initial constants in parallel and keep the best fit.
    Args:
    - form (str): equation form
    - columns (dict[str,np.ndarray]): standardized predictors
    - y (np.ndarray): standardized target
    - zmin (float): standardized value of zero precipitation
    - inits (list[dict[str,float]]): initial constants
    - nworkers (int): parallel threads
    Returns:
    - tuple[dict[str,float], OptimizeResult]: best constants and its optimizer result
    '''
    names = list(inits[0])
    def run(init):
        res = minimize(lambda params:calc_loss(form,columns,y,zmin,dict(zip(names,params))),np.array([init[c] for c in names]),
                       method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
        return dict(zip(names,res.x)),res
    results = Parallel(n_jobs=nworkers,prefer='threads')(delayed(run)(init) for init in inits)
    for i,(_,res) in enumerate(results):
        logger.debug(f'     restart {i+1}/{len(inits)}: loss={res.fun:.6f} converged={res.success}')
    return min(results,key=lambda result:result[1].fun)

def get_pysr_constants(form,predictornames,refcomplexity,runname,seeds,modelsdir,base=None):
    '''
    Purpose: Read the constants of a form from the PySR equations at refcomplexity, by matching the form's structure
    to each seed's equation, and average them across the seeds that match.
    Args:
    - form (str): equation form
    - predictornames (list[str]): predictor names
    - refcomplexity (int | None): complexity of the PySR equations to match
    - runname (str): SR run whose equation tables are searched
    - seeds (list[int]): seeds to search
    - modelsdir (str): models directory
    - base (str | None): base equation added to each table equation, for additive searches (defaults to None)
    Returns:
    - dict[str,float]: averaged constants (empty if no seed matches)
    '''
    if refcomplexity is None:
        return {}
    constantnames = extract_constants(form,predictornames)
    if not constantnames:
        return {}
    sympyfunctions = {'cube':lambda x:x**3,'square':lambda x:x**2,'neg':lambda x:-x,'sqrt':sp.sqrt,'exp':sp.exp,'log':sp.log,
                      'abs':sp.Abs,'sin':sp.sin,'cos':sp.cos,'max':sp.Max,'min':sp.Min}
    predictorsyms = {p:sp.Symbol(p) for p in predictornames}
    wildsyms      = {c:sp.Wild(c,exclude=list(predictorsyms.values())) for c in constantnames}
    try:
        formexpr = sp.sympify(form,locals=dict(sympyfunctions,**predictorsyms,**wildsyms))
    except Exception as e:
        logger.warning(f'   Could not parse form `{form}`: {e}')
        return {}
    seedconsts = []
    for seed in seeds:
        filepath = os.path.join(modelsdir,'sr',f'{runname}_{seed}_equations.csv')
        if not os.path.exists(filepath):
            continue
        df  = pd.read_csv(filepath)
        row = df[df['complexity']==refcomplexity]
        if row.empty:
            continue
        pysreq = str(row.iloc[0]['equation']).replace('^','**')
        if base:
            pysreq = f'{base}+({pysreq})'
        try:
            match = sp.sympify(pysreq,locals=dict(sympyfunctions,**predictorsyms)).match(formexpr)
        except Exception:
            match = None
        if match is None:
            logger.info(f'   Seed {seed}: no structural match at complexity {refcomplexity}, skipping')
            continue
        vals = {}
        for c in constantnames:
            v = match.get(wildsyms[c])
            if v is None or not v.is_Number:
                vals = None
                break
            vals[c] = float(v)
        if vals is None:
            logger.info(f'   Seed {seed}: match found but constants are non-numeric, skipping')
            continue
        logger.info(f'   Seed {seed}: {", ".join(f"{k}={v:.4f}" for k,v in vals.items())}')
        seedconsts.append(vals)
    if not seedconsts:
        return {}
    return {c:float(np.mean([sc[c] for sc in seedconsts])) for c in constantnames}

def get_initial_constants(eqspec,constantnames,predictornames,config,registry):
    '''
    Purpose: Build the starting constants for the multistart fit: one set from the run's own PySR results, and the rest
    drawn uniformly from [-initscale, initscale], for `nrestarts` sets in total. The PySR set is, in order of
    preference, the optimized equation named in `initfrom`, a structural match of the form to the PySR equations at
    `refcomplexity`, or constants copied by hand from the run's PySR table into `init`. Without one, all sets are
    random and a warning is logged.
    Args:
    - eqspec (dict): equation specification from configs.json
    - constantnames (list[str]): constant names
    - predictornames (list[str]): predictor names
    - config (Config): project configuration object
    - registry (dict[str,dict]): optimized equations
    Returns:
    - list[dict[str,float]]: sets of initial constants
    '''
    sr = config.sr
    initfrom = eqspec.get('initfrom')
    if initfrom:
        first,source = registry.get(initfrom,{}).get('constants',{}),f'optimized `{initfrom}`'
    else:
        first,source = get_pysr_constants(eqspec['form'],predictornames,eqspec.get('refcomplexity'),eqspec['runfrom'],eqspec.get('seeds',sr['seeds']),config.modelsdir,
            sr['runs'][eqspec['runfrom']]['residualfrom'] if sr['runs'][eqspec['runfrom']].get('additive') else None),'PySR match (averaged across seeds)'
        if not first:
            first,source = eqspec.get('init',{}),'configured PySR constants (`init`)'
    if first and set(constantnames)<=set(first):
        logger.info(f'   Start from {source}: {first}')
        inits = [{c:float(first[c]) for c in constantnames}]
    else:
        logger.warning('   No PySR start found; all starts are random (set `refcomplexity`/`seeds`, or copy the PySR constants into `init`)')
        inits = []
    rng = np.random.default_rng(0)
    while len(inits)<sr['nrestarts']:
        inits.append({c:float(v) for c,v in zip(constantnames,rng.uniform(-sr['initscale'],sr['initscale'],len(constantnames)))})
    return inits

def check_physical_form(name,registry,stats,config,split,x,validmask):
    '''
    Purpose: Check that the physical-space form with physical constants reproduces the standardized-form predictions,
    and warn if they differ by more than float32 rounding.
    Args:
    - name (str): equation name
    - registry (dict[str,dict]): optimized equations
    - stats (dict[str,float]): training statistics
    - config (Config): project configuration object
    - split (str): split name
    - x (pd.DataFrame): standardized predictors for the split
    - validmask (np.ndarray): valid-sample mask
    '''
    try:
        physical = calc_physical_constants(name,registry,stats)
    except (ValueError,KeyError) as e:
        logger.warning(f'   No physical-space check for `{name}`: {e}')
        return
    columns = {c:x[c].values for c in x.columns if c!='timeidx'}
    if name!='sr_bl_eq' and 'rh' not in columns:
        atmrun = config.sr['runs'][config.sr['optimizedeqs']['sr_atm_eq']['runfrom']]
        atmx,_,_,atmmask = load_features(split,atmrun,config)
        columns.update({c:atmx[c].values for c in ('rh','thetae','thetaestar')})
        validmask = validmask&atmmask
    inputs     = {var:(values if var=='lf' else values*stats[f'{var}_std']+stats[f'{var}_mean']) for var,values in columns.items() if var=='lf' or f'{var}_mean' in stats}
    physprecip = calc_physical_precip(name,physical,inputs,stats)[validmask]
    stdprecip  = raw_to_precip(evaluate(registry[name]['form'],columns,registry[name]['constants']),stats[f'{config.targetvar}_std'])[validmask]
    maxdiff    = float(np.max(np.abs(physprecip-stdprecip)))
    float32step = float(np.spacing(np.float32(np.max(stdprecip))))
    if maxdiff>float32step:
        logger.warning(f'   Physical-space form of `{name}` differs from the standardized form by more than float32 rounding')

def save_predictions(name,form,constants,runconfig,config,writer,split,stats):
    '''
    Purpose: Save gridded precipitation predictions of an optimized equation for one split, then check its
    physical-space form.
    Args:
    - name (str): equation name
    - form (str): equation form
    - constants (dict[str,float]): constants
    - runconfig (dict): SR run configuration
    - config (Config): project configuration object
    - writer (PredictionWriter): prediction writer
    - split (str): split name
    - stats (dict[str,float]): training statistics
    '''
    x,_,refda,validmask = load_features(split,runconfig,config)
    predpath = os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')
    if os.path.exists(predpath):
        logger.info(f'   Skipping {split} predictions, already exist')
    else:
        logger.info(f'   Generating {split} predictions...')
        raw  = evaluate(form,{c:x[c].values[validmask] for c in x.columns if c!='timeidx'},constants)
        grid = writer.unflatten(raw_to_precip(raw,writer.std),validmask,refda)
        da   = xr.DataArray(grid,dims=refda.dims,coords=refda.coords)
        da.attrs = dict(long_name=writer.longname,units=writer.units)
        writer.save(da.to_dataset(name=writer.targetvar),name,'predictions',split,config.predsdir)
    registry = load_registry(config.modelsdir)
    check_physical_form(name,registry,stats,config,split,x,validmask)

if __name__=='__main__':
    config       = Config()
    sr           = config.sr
    targetvar    = config.targetvar
    optimizedeqs = sr.get('optimizedeqs',{})
    logger.info('Spinning up...')
    selectedeqs,splits,nworkers,predictonly = parse()
    logger.info(f'Using {nworkers} parallel worker(s) for multi-start optimization...')
    stats  = load_stats(config.splitsdir)
    zmin   = (0.0-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
    writer = PredictionWriter(config.splitsdir,targetvar=targetvar)
    registry = load_registry(config.modelsdir)
    if registry:
        logger.info(f'Loaded existing registry with {len(registry)} equation(s)')
    datacache = {}
    for name,eqspec in optimizedeqs.items():
        if selectedeqs is not None and name not in selectedeqs:
            continue
        if eqspec.get('form') is None:
            logger.info(f'Skipping `{name}`, form not yet specified')
            continue
        runname   = eqspec['runfrom']
        runconfig = sr['runs'][runname]
        form      = eqspec['form']
        if predictonly:
            if name not in registry:
                logger.info(f'Skipping `{name}`, not in registry (run without --predict-only to optimize)')
                continue
            constants = registry[name]['constants']
            logger.info(f'Predicting `{name}` from existing constants: {", ".join(f"{k}={v}" for k,v in constants.items())}')
            for split in splits:
                save_predictions(name,form,constants,runconfig,config,writer,split,stats)
            continue
        if name in registry:
            logger.info(f'Skipping `{name}`, already optimized')
            continue
        residualfrom = runconfig.get('residualfrom')
        if residualfrom and residualfrom not in registry:
            logger.error(f'Skipping `{name}`, `{residualfrom}` must be optimized first')
            continue
        logger.info(f'Optimizing `{name}`...')
        if runname not in datacache:
            logger.info(f'   Loading training + validation sets...')
            xtrain,ytrain,reftrain,trainmask = load_features('train',runconfig,config,timeoffset=0)
            xvalid,yvalid,_,validmask        = load_features('valid',runconfig,config,timeoffset=int(reftrain.sizes['time']))
            predictornames = [c for c in xtrain.columns if c!='timeidx']
            fitcols   = {c:np.concatenate([xtrain[c].values[trainmask],xvalid[c].values[validmask]]) for c in predictornames}
            validcols = {c:xvalid[c].values[validmask] for c in predictornames}
            yfit      = np.concatenate([ytrain[trainmask],yvalid[validmask]])
            datacache[runname] = (predictornames,fitcols,validcols,yfit,yvalid[validmask])
            del xtrain,xvalid,ytrain,yvalid,reftrain
        predictornames,fitcols,validcols,yfit,yval = datacache[runname]
        constantnames = extract_constants(form,predictornames)
        inits = get_initial_constants(eqspec,constantnames,predictornames,config,registry)
        logger.info(f'   Running L-BFGS-B with {len(yfit):,} samples, {len(inits)} start(s), {nworkers} worker(s)...')
        constants,res = multistart_optimize(form,fitcols,yfit,zmin,inits,nworkers)
        logger.info(f'   Constants: {", ".join(f"{k}={v:.6f}" for k,v in constants.items())}')
        logger.info(f'   Training Loss: {res.fun:.6f} | Converged: {res.success}')
        constants = round_constants(constants,sr['constantsigfigs'])
        trainloss = calc_loss(form,fitcols,yfit,zmin,constants)
        validloss = calc_loss(form,validcols,yval,zmin,constants)
        logger.info(f'   Rounded constants ({sr["constantsigfigs"]} significant figures): {", ".join(f"{k}={v}" for k,v in constants.items())}')
        logger.info(f'   Rounded Training Loss: {trainloss:.6f} | Rounded Validation Loss: {validloss:.6f}')
        registry[name] = dict(form=form,constants=constants,train_loss=trainloss,valid_loss=validloss)
        save_registry(registry,config.modelsdir)
        logger.info(f'   Registry saved ({len(registry)} equation(s))')
        for split in splits:
            save_predictions(name,form,constants,runconfig,config,writer,split,stats)
