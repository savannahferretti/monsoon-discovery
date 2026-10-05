#!/usr/bin/env python

import os
import shutil
import logging
import argparse
import tempfile
import warnings
import numpy as np
import pandas as pd
from timingutils import TimingConfig,parse_names,load_stats,restrict_kernel_seeds,z_to_precip,calc_zmin
from data import load_features
from equations import load_registry
from scripts.models.sr.train import save,TIMEOUT

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)

def subsample_timestep(features,target,subsetfrac,seed,stats,logmin=-4,logmax=2):
    '''
    Purpose: Copy of scripts/models/sr/train.py:subsample_timestep that takes the variant's statistics instead of
        reading data/splits/stats.json.
    Args:
    - features (pd.DataFrame): predictor features including a 'timeidx' column
    - target (np.ndarray): z-scored log1p(tp) target values
    - subsetfrac (float): target fraction of total available samples
    - seed (int): random seed
    - stats (dict): variant training statistics
    - logmin (float): log10 lower bound of wet bins in mm
    - logmax (float): log10 upper bound of wet bins in mm
    Returns:
    - tuple[pd.DataFrame, np.ndarray]: subsampled features (without 'timeidx') and target
    '''
    precip        = z_to_precip(np.asarray(target),stats)
    rng           = np.random.default_rng(seed)
    timeidx       = features['timeidx'].values
    uniquetimes,startindices = np.unique(timeidx,return_index=True)
    sortorder     = np.argsort(timeidx,kind='stable')
    peakprecip    = np.maximum.reduceat(precip[sortorder],startindices)
    nbins         = int(logmax-logmin)
    ntimesteps    = max(1,int(round(subsetfrac*len(uniquetimes))))
    logbins       = np.linspace(logmin,logmax,nbins+1)
    logpeakprecip = np.log10(peakprecip.clip(min=10**(logmin-1)))
    drymask       = peakprecip<=10**logmin
    binpools = []
    if drymask.any():
        binpools.append(uniquetimes[drymask])
    for i in range(nbins):
        pool = uniquetimes[(logpeakprecip>logbins[i])&(logpeakprecip<=logbins[i+1])]
        if len(pool)>0:
            binpools.append(pool)
    totalavailable = sum(len(pool) for pool in binpools)
    selected = []
    for pool in binpools:
        ndraw = max(1,round(len(pool)/totalavailable*ntimesteps))
        selected.append(rng.choice(pool,ndraw,replace=len(pool)<ndraw))
    selectedtimes  = np.unique(np.concatenate(selected))
    subsetindices  = np.where(np.isin(timeidx,selectedtimes))[0]
    rng.shuffle(subsetindices)
    return features.iloc[subsetindices].drop(columns=['timeidx']).reset_index(drop=True),np.asarray(target)[subsetindices]

def fit(xsub,ysub,predictors,srconfig,seed,procs,tmpdir,zmin):
    '''
    Purpose: Copy of scripts/models/sr/train.py:fit that takes the variant's zmin instead of reading
        data/splits/stats.json.
    Args:
    - xsub (pd.DataFrame): subsampled predictors
    - ysub (np.ndarray): subsampled target
    - predictors (list[str]): predictor names
    - srconfig (dict): SR configuration with run-level searchparams merged in
    - seed (int): random seed
    - procs (int): number of Julia workers
    - tmpdir (str): temporary directory for PySR
    - zmin (float): standardized value of zero precipitation
    Returns:
    - PySRRegressor: fitted model
    '''
    searchparams      = srconfig['searchparams']
    operators         = srconfig['operators']
    complexityparams  = srconfig['complexity']
    populations       = searchparams.get('populations',3*procs)
    niterations       = searchparams.get('targettotal',searchparams['iterations']*populations)//populations
    loss = 'loss(x, y) = (x - y)^2' if searchparams.get('loss')=='plainmse' else f'loss(x, y) = (({zmin:.8f}) + max(x, 0.0) - y)^2'
    os.environ.setdefault('JULIA_NUM_THREADS',str(os.cpu_count() or 1))
    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        population_size=searchparams['populationsize'],
        ncycles_per_iteration=searchparams['cyclesperiteration'],
        weight_optimize=searchparams['weightoptimize'],
        parsimony=searchparams['parsimony'],
        binary_operators=operators['binary'],
        unary_operators=operators['unary'],
        complexity_of_operators=operators['complexity'],
        complexity_of_variables=[complexityparams['ofvariables'].get(p,2) for p in predictors]
            if isinstance(complexityparams['ofvariables'],dict) else complexityparams['ofvariables'],
        complexity_of_constants=complexityparams['ofconstants'],
        maxsize=searchparams['maxsize'],
        maxdepth=searchparams['maxdepth'],
        constraints={k:tuple(v) for k,v in srconfig.get('constraints',{}).items()},
        nested_constraints=srconfig.get('nestedconstraints',{}),
        extra_sympy_mappings={'square':lambda x:x**2},
        elementwise_loss=loss,
        model_selection='best',
        batch_size=searchparams['batchsize'],
        random_state=seed,
        parallelism='multithreading',
        procs=procs,
        tempdir=tmpdir,
        temp_equation_file=True,
        delete_tempfiles=True,
        timeout_in_seconds=TIMEOUT,
        progress=False)
    model.fit(xsub.values,ysub,variable_names=predictors)
    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run PySR searches for the timing variants.')
    parser.add_argument('--variants',type=str,default='all',help='Comma-separated variant names, or `all`')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated SR run names, or `all`')
    parser.add_argument('--procs',type=int,default=50,help='Number of Julia worker processes (default: 50)')
    parser.add_argument('--seeds',type=str,default=None,help='Comma-separated seeds overriding the config')
    parser.add_argument('--iterations',type=int,default=None,help='Override iterations from config (quick tests)')
    parser.add_argument('--subsetfrac',type=float,default=None,help='Override subsetfrac from config (quick tests)')
    args = parser.parse_args()
    baseconfig = TimingConfig()
    variants   = parse_names(args.variants,list(baseconfig.timing['variants']))
    for variant in variants:
        config = TimingConfig(variant)
        sr     = config.sr
        stats  = load_stats(config)
        zmin   = calc_zmin(stats)
        seeds  = [int(seed) for seed in args.seeds.split(',')] if args.seeds else sr['seeds']
        kernelseeds = restrict_kernel_seeds(config)
        logger.info(f'[{variant}] Kernel-integrated features average NN-GAUSS seeds {kernelseeds}' if kernelseeds else f'[{variant}] No NN-GAUSS kernel weights found; only runs without `weightsfrom` can run')
        if args.iterations is not None:
            sr['searchparams']['iterations'] = args.iterations
            sr['searchparams'].pop('targettotal',None)
        subsetfrac = args.subsetfrac if args.subsetfrac is not None else sr['subsetfrac']
        runs       = config.srruns
        for name in parse_names(args.runs,list(runs)):
            runconfig = runs[name]
            todo = [seed for seed in seeds if not os.path.exists(os.path.join(config.modelsdir,'sr',f'{name}_{seed}_equations.csv'))]
            if not todo:
                logger.info(f'[{variant}] Skipping `{name}`, all equation files already exist')
                continue
            residualfrom = runconfig.get('residualfrom')
            if residualfrom and residualfrom not in load_registry(config):
                logger.error(f'[{variant}] `{name}` needs `{residualfrom}` in {config.modelsdir}/sr/optimized_equations.csv; run sr_optimize.py --equations {residualfrom} first')
                continue
            searchparams = {**sr['searchparams'],**runconfig.get('searchparams',{})}
            complexity   = {**sr['complexity'],'ofvariables':{**sr['complexity']['ofvariables'],**runconfig.get('complexityofvariables',{})}}
            srrun        = {**sr,'searchparams':searchparams,'complexity':complexity}
            logger.info(f'[{variant}] Variable complexities for `{name}`: {complexity["ofvariables"]}')
            logger.info(f'[{variant}] Loading training and validation splits for `{name}`...')
            xtrain,ytrain,reftrain,trainmask = load_features(config,'train',runconfig)
            xvalid,yvalid,_,validmask        = load_features(config,'valid',runconfig,timeoffset=int(reftrain.sizes['time']))
            predictors = [c for c in xtrain.columns if c!='timeidx']
            xfit = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
            yfit = np.concatenate([ytrain[trainmask],yvalid[validmask]])
            del xtrain,xvalid,ytrain,yvalid,reftrain
            for seed in todo:
                logger.info(f'[{variant}] Running `{name}` seed {seed}, subsampling ~{subsetfrac:.1%} of samples by timestep...')
                xsub,ysub   = subsample_timestep(xfit,yfit,subsetfrac,seed,stats)
                tempdirpath = tempfile.mkdtemp(prefix='pysr_')
                try:
                    model = fit(xsub,ysub,predictors,srrun,seed,args.procs,tempdirpath,zmin)
                finally:
                    shutil.rmtree(tempdirpath,ignore_errors=True)
                save(model,name,seed,config)
                del model
            del xfit,yfit
