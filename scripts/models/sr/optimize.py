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
    - tuple[set[str]|None, list[str], int, bool]: selected equation names (or None for all),
        list of splits for which to save predictions, number of parallel workers, and force flag
    '''
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

def optimize_constants(form,predictornames,x,y,zmin,init,plainmse=False):
    constantnames = extract_constants(form,predictornames)
    initialparams = np.array([init.get(c,1.0) for c in constantnames])
    def objective(params):
        constants = dict(zip(constantnames,params))
        raw       = eval_form(form,x,predictornames,constants)
        if plainmse:
            return float(np.mean((raw-y)**2))
        pred      = zmin+np.maximum(raw,0.0)
        return float(np.mean((pred-y)**2))
    res = minimize(objective,initialparams,method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-14,'gtol':1e-10})
    return dict(zip(constantnames,res.x)),res

def multistart_optimize(form,predictornames,x,y,zmin,init,nrestarts=50,seed=0,nworkers=1,extra_inits=None,plainmse=False):
    '''
    Purpose: Optimize constants via L-BFGS-B from multiple starting points. The primary
        start uses PySR-derived constants (init); remaining restarts perturb those values
        via LHS over [-5, 5] for constants not in init, and within +/-3 of init values
        for known ones.
    '''
    constantnames = extract_constants(form,predictornames)
    nconstants    = len(constantnames)
    fixed_inits   = [init] + (extra_inits or [])
    nrandom       = max(0,nrestarts - len(fixed_inits))
    sampler       = LatinHypercube(d=nconstants,seed=seed)
    samples       = sampler.random(n=nrandom)
    inits = list(fixed_inits)
    for i in range(nrandom):
        restart = {}
        for j,c in enumerate(constantnames):
            if c in init:
                restart[c] = init[c] + (samples[i,j]*6.0-3.0)
            else:
                restart[c] = samples[i,j]*10.0-5.0
        inits.append(restart)
    resultslist = Parallel(n_jobs=nworkers,prefer='threads')(
        delayed(optimize_constants)(form,predictornames,x,y,zmin,restartinit,plainmse=plainmse)
        for restartinit in inits)
    bestconstants,bestresult = None,None
    nconverged = 0
    for i,(constants,res) in enumerate(resultslist):
        if res.success:
            nconverged += 1
        if bestresult is None or res.fun < bestresult.fun:
            bestconstants,bestresult = constants,res
        logger.debug(f'     restart {i+1}/{len(inits)}: loss={res.fun:.6f} converged={res.success}')
    logger.debug(f'   {nconverged}/{len(inits)} restarts converged; best loss={bestresult.fun:.6f}')
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
    logger.info(f'   Registry saved → {registrypath}')

def pysr_init(form,predictornames,refcomplexity,runname,seeds,modelsdir,matchform=None,reparameterize=None):
    '''
    Purpose: Initialize constants by structurally unifying the parametric form with
        each seed's PySR equation at refcomplexity, then averaging matched constants
        across seeds. Uses SymPy's Wild + match so trivial algebraic rearrangements
        (e.g. `- -b` vs `+ b`, `a + x` vs `x + a`) don't count as structural mismatches.
        Seeds whose PySR equation cannot be unified with the form — or whose matched
        constants are not purely numeric — are skipped. When a matchform is provided
        (a more general pattern with extra constants), matching uses that form and then
        reparameterizes the matched constants to the optimization form.
    Args:
    - form (str): Python expression string with named constants (optimization form)
    - predictornames (list[str]): predictor column names
    - refcomplexity (int|None): target complexity level to read from per-seed CSVs
    - runname (str): SR run name (used to locate per-seed equation CSVs)
    - seeds (list[int]): list of random seeds
    - modelsdir (str): path to models directory
    - matchform (str|None): optional generalized form for PySR matching (has extra constants
        not in the optimization form, e.g. a coefficient on a bare predictor)
    - reparameterize (dict|None): when matchform is used, maps each optimization constant name
        to a Python expression in terms of the matchform's constant names
    Returns:
    - dict: constant name → averaged float value, or {} if no seeds unify
    '''
    if refcomplexity is None:
        return {}
    constantnames = extract_constants(form,predictornames)
    if not constantnames:
        return {}
    useform        = matchform if matchform is not None else form
    matchconstants = extract_constants(useform,predictornames)
    predictorsyms  = {p:sp.Symbol(p) for p in predictornames}
    wildsyms       = {c:sp.Wild(c,exclude=list(predictorsyms.values())) for c in matchconstants}
    formns         = dict(SRSYMPY,**predictorsyms,**wildsyms)
    parsens        = dict(SRSYMPY,**predictorsyms)
    try:
        formexpr = sp.sympify(useform,locals=formns)
    except Exception as e:
        logger.warning(f'   Could not parse form `{useform}`: {e}')
        return {}
    seedconsts = []
    for seed in seeds:
        filepath = os.path.join(modelsdir,'sr',f'{runname}_{seed}_equations.csv')
        if not os.path.exists(filepath):
            logger.info(f'   Seed {seed}: equation CSV not found, skipping')
            continue
        df  = pd.read_csv(filepath)
        row = df[df['complexity']==refcomplexity]
        if row.empty:
            logger.info(f'   Seed {seed}: no equation at complexity {refcomplexity}, skipping')
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
        rawvals = {}
        for c in matchconstants:
            v = match.get(wildsyms[c])
            if v is None or not v.is_Number:
                rawvals = None
                break
            rawvals[c] = float(v)
        if rawvals is None:
            logger.info(f'   Seed {seed}: match found but constants are non-numeric, skipping')
            continue
        if reparameterize is not None:
            vals = {c:float(eval(reparameterize[c],{'__builtins__':{}},rawvals)) for c in constantnames}
        else:
            vals = rawvals
        logger.info(f'   Seed {seed}: {", ".join(f"{k}={v:.4f}" for k,v in vals.items())}')
        seedconsts.append(vals)
    if not seedconsts:
        return {}
    return {c:float(np.mean([sc[c] for sc in seedconsts])) for c in constantnames}

def predict_split(form,predictornames,constants,runconfig,config,writer,split,zmin):
    x,y,refda,validmask = load_data(split,runconfig,config)
    xvalid = x[validmask][predictornames].reset_index(drop=True)
    raw    = eval_form(form,xvalid,predictornames,constants)
    residualfrom = runconfig.get('residualfrom')
    if residualfrom:
        from scripts.models.sr.train import eval_baseline
        registrypath = os.path.join(config.modelsdir,'sr','optimized_equations.pkl')
        with open(registrypath,'rb') as f:
            reg = pickle.load(f)
        entry = reg[residualfrom]
        eqspec = config.sr['optimizedeqs'][residualfrom]
        baserunconfig = config.sr['runs'][eqspec['runfrom']]
        basex,_,_,bvmask = load_data(split,baserunconfig,config)
        basecols = {c:basex[bvmask][c].values for c in basex.columns if c != 'timeidx'}
        baseline = eval_baseline(entry['form'],basecols,entry['constants'])
        raw = baseline + raw
    pred   = zmin+np.maximum(raw,0.0)
    grid   = np.maximum(np.expm1(writer.unflatten(pred,validmask,refda)*writer.std+writer.mean),0.0).astype(np.float32)
    da     = xr.DataArray(grid,dims=refda.dims,coords=refda.coords)
    da.attrs = dict(long_name=writer.longname,units=writer.units)
    return da.to_dataset(name=writer.targetvar)

if __name__=='__main__':
    import time
    config       = Config()
    sr           = config.sr
    targetvar    = config.targetvar
    optimizedeqs = sr.get('optimizedeqs',{})
    selectedeqs,splits,nworkers,force = parse()
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
    datacache = {}
    for name,eqspec in optimizedeqs.items():
        if selectedeqs is not None and name not in selectedeqs:
            continue
        if eqspec.get('form') is None:
            continue
        if name in registry and not force:
            continue
        if name in registry and force:
            del registry[name]
        runname        = eqspec['runfrom']
        runconfig      = sr['runs'][runname]
        form           = eqspec['form']
        refcomplexity  = eqspec.get('refcomplexity')
        nrestarts      = eqspec.get('nrestarts',50)
        useplainmse    = runconfig.get('residualfrom') is not None
        logger.info(f'Optimizing {name} with form {form}...')
        logger.info('Spinning up...')
        if runname not in datacache:
            xtrain,ytrain,reftrain,trainmask = load_data('train',runconfig,config,time_offset=0)
            xvalid,yvalid,_,validmask        = load_data('valid',runconfig,config,time_offset=int(reftrain.sizes['time']))
            xfit  = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
            yfit  = np.concatenate([ytrain[trainmask],yvalid[validmask]])
            datacache[runname] = (xfit,yfit,xvalid,yvalid,validmask)
            del xtrain,ytrain,reftrain
        xfitfull,yfit,xvalid,yvalid,validmask = datacache[runname]
        predictornames = [c for c in xfitfull.columns if c != 'timeidx']
        xfit           = xfitfull[predictornames]
        logger.info(f'   Loading training + validation sets ({len(yfit):,} samples)...')
        constantnames  = extract_constants(form,predictornames)
        eq_seeds       = eqspec.get('seeds',sr['seeds'])
        matchform      = eqspec.get('matchform')
        reparameterize = eqspec.get('reparameterize')
        explicit_init  = eqspec.get('init')
        if explicit_init is not None:
            init = explicit_init
        else:
            init = pysr_init(form,predictornames,refcomplexity,runname,eq_seeds,config.modelsdir,
                             matchform=matchform,reparameterize=reparameterize)
        initdisplay = {c:init.get(c,1.0) for c in constantnames}
        initsource = 'configured' if explicit_init is not None else 'averaged across seeds'
        logger.info(f'   Initial constants ({initsource}): {", ".join(f"{k}={v:.4f}" for k,v in initdisplay.items())}')
        anchor_inits = []
        for prevname,preventry in registry.items():
            if optimizedeqs.get(prevname,{}).get('runfrom') != runname:
                continue
            prevconsts = preventry['constants']
            if set(prevconsts.keys()) < set(constantnames):
                anchor = {c:(prevconsts[c] if c in prevconsts else 1.0) for c in constantnames}
                anchor_inits.append(anchor)
                logger.info(f'   Anchor start from {prevname}: {", ".join(f"{k}={v:.4f}" for k,v in anchor.items())}')
        logger.info(f'Running L-BFGS-B with {nrestarts} restarts and {nworkers} workers...')
        t0 = time.time()
        constants,res = multistart_optimize(form,predictornames,xfit,yfit,zmin,init,nrestarts,
                                            nworkers=nworkers,extra_inits=anchor_inits,plainmse=useplainmse)
        elapsed = time.time()-t0
        logger.info(f'   Time to completion: {elapsed:.0f} s')
        xvalidsub  = xvalid[validmask][predictornames].reset_index(drop=True)
        validtgt   = yvalid[validmask]
        constants  = {k:round(float(v),2) for k,v in constants.items()}
        if useplainmse:
            trainpred = eval_form(form,xfit,predictornames,constants)
            validpred = eval_form(form,xvalidsub,predictornames,constants)
        else:
            trainpred = zmin+np.maximum(eval_form(form,xfit,predictornames,constants),0.0)
            validpred = zmin+np.maximum(eval_form(form,xvalidsub,predictornames,constants),0.0)
        trainloss  = float(np.mean((trainpred-yfit)**2))
        validloss  = float(np.mean((validpred-validtgt)**2))
        logger.info(f'   Optimized constants: {", ".join(f"{k}={v}" for k,v in constants.items())}')
        logger.info(f'   Training Loss: {trainloss:.6f} | Validation Loss: {validloss:.6f}')
        registry[name] = dict(form=form,constants=constants,
                              train_loss=trainloss,valid_loss=validloss)
        save_registry(registry,config)
        for split in splits:
            predpath = os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')
            if os.path.exists(predpath) and not force:
                continue
            logger.info(f'   Generating {split} predictions...')
            predds = predict_split(form,predictornames,constants,runconfig,config,writer,split,zmin)
            writer.save(predds,name,'predictions',split,config.predsdir)
            del predds