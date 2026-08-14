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
from joblib import Parallel, delayed
from scipy.optimize import minimize
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
    'min':np.minimum}

SRSYMPY = {
    'cube':lambda x:x**3,
    'square':lambda x:x**2,
    'neg':lambda x:-x,
    'sqrt':sp.sqrt,
    'exp':sp.exp,
    'log':sp.log,
    'abs':sp.Abs,
    'sin':sp.sin,
    'cos':sp.cos,
    'max':sp.Max,
    'min':sp.Min}

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

def optimize_constants(form,predictornames,x,y,zmin,init,baseline=None):
    '''
    Purpose: Optimize named constants in an SR equation form via scipy L-BFGS-B.
        The objective is MSE in z-scored log1p space, consistent with neural network
        training. For non-residual models, predictions are zmin + ReLU(f(x)); for
        residual models (baseline provided), predictions are max(baseline + f(x), zmin)
        compared against the full (non-residual) target y.
    Args:
    - form (str): Python expression string using predictor names and constant names
    - predictornames (list[str]): predictor column names in x
    - x (pd.DataFrame): full train+valid feature matrix (valid samples only)
    - y (np.ndarray): z-scored log1p target for the same samples (full target, not residual)
    - zmin (float): z-scored value corresponding to 0 mm precipitation (-mu/sigma)
    - init (dict): initial constant values; constants absent from dict default to 1.0
    - baseline (np.ndarray|None): baseline prediction for valid samples (residual models only)
    Returns:
    - tuple[dict, OptimizeResult]: optimized constants and scipy optimization result
    '''
    constantnames = extract_constants(form,predictornames)
    initialparams = np.array([init.get(c,1.0) for c in constantnames])
    def objective(params):
        constants = dict(zip(constantnames,params))
        raw       = eval_form(form,x,predictornames,constants)
        if baseline is not None:
            pred = np.maximum(baseline+raw,zmin)
        else:
            pred = zmin+np.maximum(raw,0.0)
        return float(np.nanmean((pred-y)**2))
    res = minimize(objective,initialparams,method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
    return dict(zip(constantnames,res.x)),res

def multistart_optimize(form,predictornames,x,y,zmin,init,nrestarts=1,initscale=5.0,seed=0,nworkers=1,extra_inits=None,baseline=None):
    '''
    Purpose: Optimize named constants via L-BFGS-B with optional random restarts.
        The first restart uses `init`; any extra_inits follow; remaining slots are
        filled with uniform random draws from [-initscale, initscale].
        Restarts are run in parallel across `nworkers` threads.
    Args:
    - form (str): Python expression string using predictor names and constant names
    - predictornames (list[str]): predictor column names in x
    - x (pd.DataFrame): full train+valid feature matrix (valid samples only)
    - y (np.ndarray): z-scored log1p target for the same samples
    - zmin (float): z-scored value corresponding to 0 mm precipitation (-mu/sigma)
    - init (dict): initial constant values for the first restart
    - nrestarts (int): total number of optimization runs (default 1 = single run from init)
    - initscale (float): half-range for uniform random initialization (default 5.0)
    - seed (int): RNG seed for reproducible random restarts
    - nworkers (int): number of parallel threads for restarts (default 1)
    - extra_inits (list[dict]|None): additional anchor starting points inserted after init
        and before random draws; counted against nrestarts
    - baseline (np.ndarray|None): baseline prediction for valid samples (residual models only)
    Returns:
    - tuple[dict, OptimizeResult]: best constants and corresponding scipy result
    '''
    constantnames = extract_constants(form,predictornames)
    rng           = np.random.default_rng(seed)
    fixed_inits   = [init] + (extra_inits or [])
    nrandom       = max(0, nrestarts - len(fixed_inits))
    inits         = fixed_inits + [
        {c:float(v) for c,v in zip(constantnames,rng.uniform(-initscale,initscale,len(constantnames)))}
        for _ in range(nrandom)]
    resultslist = Parallel(n_jobs=nworkers,prefer='threads')(
        delayed(optimize_constants)(form,predictornames,x,y,zmin,restartinit,baseline=baseline)
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
    rows = [dict(name=name,form=entry['form'],train_loss=entry['train_loss'],valid_loss=entry['valid_loss'],
                 constants=json.dumps(entry['constants'])) for name,entry in registry.items()]
    pd.DataFrame(rows).to_csv(registrycsvpath,index=False)
    logger.info(f'   Registry saved ({len(registry)} equation(s)) → {registrypath}')

def pysr_init(form,predictornames,refcomplexity,runname,seeds,modelsdir):
    '''
    Purpose: Initialize constants by structurally unifying the parametric form with
        each seed's PySR equation at refcomplexity, then averaging matched constants
        across seeds. Uses SymPy's Wild + match so trivial algebraic rearrangements
        (e.g. `- -b` vs `+ b`, `a + x` vs `x + a`) don't count as structural mismatches.
        Seeds whose PySR equation cannot be unified with the form — or whose matched
        constants are not purely numeric — are skipped.
    Args:
    - form (str): Python expression string with named constants
    - predictornames (list[str]): predictor column names
    - refcomplexity (int|None): target complexity level to read from per-seed CSVs
    - runname (str): SR run name (used to locate per-seed equation CSVs)
    - seeds (list[int]): list of random seeds
    - modelsdir (str): path to models directory
    Returns:
    - dict: constant name → averaged float value, or {} if no seeds unify
    '''
    if refcomplexity is None:
        return {}
    constantnames = extract_constants(form,predictornames)
    if not constantnames:
        return {}
    predictorsyms = {p:sp.Symbol(p) for p in predictornames}
    wildsyms      = {c:sp.Wild(c,exclude=list(predictorsyms.values())) for c in constantnames}
    formns        = dict(SRSYMPY,**predictorsyms,**wildsyms)
    parsens       = dict(SRSYMPY,**predictorsyms)
    try:
        formexpr = sp.sympify(form,locals=formns)
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
        try:
            pysrexpr = sp.sympify(pysreq,locals=parsens)
            match    = pysrexpr.match(formexpr)
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

def predict_split(form,predictornames,constants,runconfig,config,writer,split,zmin,combined=False):
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
    - zmin (float): z-scored value corresponding to 0 mm precipitation (-mu/sigma)
    - combined (bool): if True, form predicts full target (ignore baseline from load_data)
    Returns:
    - xr.Dataset: predictions in native units with dims (time, lat, lon)
    '''
    x,y,refda,validmask,baseline = load_data(split,runconfig,config)
    if combined:
        baseline = None
    xvalid = x[validmask][predictornames].reset_index(drop=True)
    raw    = eval_form(form,xvalid,predictornames,constants)
    if baseline is not None:
        pred = np.maximum(baseline[validmask]+raw,zmin)
    else:
        pred = zmin+np.maximum(raw,0.0)
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
    logger.info(f'Using {nworkers} parallel worker(s) for multi-start optimization...')
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    zmin = (0.0-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
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
        if name in registry:
            logger.info(f'Skipping `{name}`, already optimized')
            continue
        runname        = eqspec['runfrom']
        runconfig      = sr['runs'][runname]
        form           = eqspec['form']
        refcomplexity  = eqspec.get('refcomplexity')
        logger.info(f'Optimizing `{name}`...')
        if runname not in datacache:
            logger.info(f'   Loading training + validation sets...')
            xtrain,ytrain,reftrain,trainmask,btrain = load_data('train',runconfig,config,time_offset=0)
            xvalid,yvalid,_,validmask,bvalid        = load_data('valid',runconfig,config,time_offset=int(reftrain.sizes['time']))
            xfit  = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
            yfit_resid = np.concatenate([ytrain[trainmask],yvalid[validmask]])
            if btrain is not None:
                bfit = np.concatenate([btrain[trainmask],bvalid[validmask]])
                yfit = yfit_resid + bfit
                yvalid_full = yvalid + bvalid
            else:
                bfit = None
                yfit = yfit_resid
                yvalid_full = yvalid
            datacache[runname] = (xfit,yfit,xvalid,yvalid_full,validmask,bfit,bvalid)
            del xtrain,ytrain,reftrain,btrain
        xfitfull,yfit,xvalid,yvalid,validmask,bfit,bvalid = datacache[runname]
        combined = eqspec.get('combined',False)
        if combined:
            bfit,bvalid = None,None
        predictornames = [c for c in xfitfull.columns if c != 'timeidx']
        xfit       = xfitfull[predictornames]
        nrestarts     = eqspec.get('nrestarts',1)
        initscale     = eqspec.get('initscale',5.0)
        constantnames = extract_constants(form,predictornames)
        refcomplexity = eqspec.get('refcomplexity')
        eq_seeds = eqspec.get('seeds', sr['seeds'])
        explicit_init = eqspec.get('init')
        if explicit_init is not None:
            init = explicit_init
            logger.info(f'   Configured init: {", ".join(f"{k}={v:.4f}" for k,v in init.items())}')
        else:
            init = pysr_init(form,predictornames,refcomplexity,runname,eq_seeds,config.modelsdir)
            if init:
                logger.info(f'   PySR init (averaged across seeds): {", ".join(f"{k}={v:.4f}" for k,v in init.items())}')
            else:
                logger.info(f'   No PySR init found; defaulting all constants to 1.0')
        # Anchor starts: for each already-optimized equation whose constant set is a strict
        # subset of this one's, inject its constants (new constants default to 1.0) as an
        # additional guaranteed starting point — without making it the primary init.
        anchor_inits = []
        for prevname,preventry in registry.items():
            if optimizedeqs.get(prevname,{}).get('runfrom') != runname:
                continue
            prevconsts = preventry['constants']
            if set(prevconsts.keys()) < set(constantnames):
                anchor = {c:(prevconsts[c] if c in prevconsts else 1.0) for c in constantnames}
                anchor_inits.append(anchor)
                logger.info(f'   Anchor start from {prevname}: {", ".join(f"{k}={v:.4f}" for k,v in anchor.items())}')
        logger.info(f'   Running L-BFGS-B with {len(xfit):,} samples, {nrestarts} restart(s) '
                    f'({len(anchor_inits)} anchor(s)), {nworkers} worker(s)...')
        constants,res = multistart_optimize(form,predictornames,xfit,yfit,zmin,init,nrestarts,initscale,
                                            nworkers=nworkers,extra_inits=anchor_inits,baseline=bfit)
        trainloss  = float(res.fun)
        xvalidsub  = xvalid[validmask][predictornames].reset_index(drop=True)
        validtgt   = yvalid[validmask]
        raw        = eval_form(form,xvalidsub,predictornames,constants)
        if bvalid is not None:
            bvalidsub = bvalid[validmask]
            validpred = np.maximum(bvalidsub+raw,zmin)
        else:
            validpred = zmin+np.maximum(raw,0.0)
        validloss  = float(np.nanmean((validpred-validtgt)**2))
        logger.info(f'   Constants: {", ".join(f"{k}={v:.6f}" for k,v in constants.items())}')
        logger.info(f'   Training Loss: {trainloss:.6f} | Validation Loss: {validloss:.6f} | Converged: {res.success}')
        constants = {k:round(float(v),2) for k,v in constants.items()}
        raw        = eval_form(form,xfit,predictornames,constants)
        if bfit is not None:
            trainpred = np.maximum(bfit+raw,zmin)
        else:
            trainpred = zmin+np.maximum(raw,0.0)
        trainloss  = float(np.nanmean((trainpred-yfit)**2))
        raw        = eval_form(form,xvalidsub,predictornames,constants)
        if bvalid is not None:
            bvalidsub = bvalid[validmask]
            validpred = np.maximum(bvalidsub+raw,zmin)
        else:
            validpred = zmin+np.maximum(raw,0.0)
        validloss  = float(np.nanmean((validpred-validtgt)**2))
        logger.info(f'   Rounded constants: {", ".join(f"{k}={v:.2f}" for k,v in constants.items())}')
        logger.info(f'   Rounded Training Loss: {trainloss:.6f} | Rounded Validation Loss: {validloss:.6f}')
        registry[name] = dict(form=form,constants=constants,
                              train_loss=trainloss,valid_loss=validloss)
        save_registry(registry,config)
        for split in splits:
            predpath = os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')
            if os.path.exists(predpath):
                logger.info(f'   Skipping {split} predictions, already exist')
                continue
            logger.info(f'   Generating {split} predictions...')
            predds = predict_split(form,predictornames,constants,runconfig,config,writer,split,zmin,combined=combined)
            writer.save(predds,name,'predictions',split,config.predsdir)
            del predds