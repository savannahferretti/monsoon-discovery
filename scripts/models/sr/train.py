#!/usr/bin/env python

import os
import shutil
import logging
import argparse
import tempfile
import warnings
import numpy as np
import pandas as pd
from scripts.utils import Config,load,load_stats,standardize,flatten
from scripts.models.sr.equations import evaluate,load_registry

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)

def parse():
    '''
    Purpose: Parse command-line arguments.
    Returns:
    - tuple[set[str] | None, int, int | None, float | None]: run names to train (None for all), number of Julia
        workers, and optional overrides of iterations and subsetfrac for quick tests
    '''
    parser = argparse.ArgumentParser(description='Train PySR symbolic regression models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated run names to train, or `all`')
    parser.add_argument('--procs',type=int,default=50,help='Number of Julia worker processes (default: 50)')
    parser.add_argument('--iterations',type=int,default=None,help='Override iterations from config (useful for quick tests)')
    parser.add_argument('--subsetfrac',type=float,default=None,help='Override subsetfrac from config (useful for quick tests)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {n.strip() for n in args.runs.split(',')}
    return selectedruns,args.procs,args.iterations,args.subsetfrac

def load_kernels(config,weightsfrom,dsig):
    '''
    Purpose: Load a kernel NN's weights for every seed, rescale each so that sum(k·Δσ) = 1 in float64, and average
    across seeds.
    Args:
    - config (Config): project configuration object
    - weightsfrom (str): NN run whose kernels are used
    - dsig (np.ndarray): sigma thickness weights with shape (nlevs,)
    Returns:
    - dict[str,np.ndarray]: profile variable → kernel weights with shape (nlevs,)
    '''
    kernels = []
    for seed in config.nn['seeds']:
        k = load(os.path.join(config.weightsdir,f'{weightsfrom}_{seed}_weights.nc'))['k']
        kernels.append(k.values/(k.values*dsig[None,:]).sum(axis=1,keepdims=True))
        fieldnames = [str(field) for field in k['field'].values]
    return dict(zip(fieldnames,np.mean(kernels,axis=0)))

def load_features(splitname,runconfig,config,timeoffset=0):
    '''
    Purpose: Load the SR predictors and target for one split. Profiles are kernel-integrated in physical units (if
    `weightsfrom`) and every predictor except land fraction is then standardized. An optimized SR equation named in
    `residualfrom` is added as a predictor.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - runconfig (dict): SR run configuration
    - config (Config): project configuration object
    - timeoffset (int): offset added to the time index, so that train and valid timesteps stay distinct
    Returns:
    - tuple[pd.DataFrame, np.ndarray, xr.DataArray, np.ndarray]: predictors (plus 'timeidx'), standardized target,
        target DataArray with (time, lat, lon) coordinates, and boolean mask of valid samples
    '''
    fieldvars   = runconfig['fieldvars']
    localvars   = runconfig.get('localvars',[])
    weightsfrom = runconfig.get('weightsfrom')
    stats = load_stats(config.splitsdir)
    ds    = load(os.path.join(config.splitsdir,f'{splitname}.h5'))
    ntime = ds.sizes['time']
    refda = ds[config.targetvar].transpose('time','lat','lon')
    columns = {}
    if weightsfrom and fieldvars:
        dsig    = ds['dsig'].values
        kernels = load_kernels(config,weightsfrom,dsig)
        for var in fieldvars:
            profiles = ds[var].transpose('time','lat','lon','sig').values.reshape(-1,ds.sizes['sig'])
            columns[var] = standardize(profiles@(kernels[var]*dsig),var,stats)
    else:
        for var in fieldvars:
            columns[var] = standardize(flatten(ds[var],ntime),var,stats)
    for var in localvars:
        columns[var] = standardize(flatten(ds[var],ntime),var,stats)
    residualfrom = runconfig.get('residualfrom')
    if residualfrom:
        entry = load_registry(config.modelsdir)[residualfrom]
        baserunconfig = config.sr['runs'][config.sr['optimizedeqs'][residualfrom]['runfrom']]
        basefeatures,_,_,_ = load_features(splitname,baserunconfig,config,timeoffset)
        columns[residualfrom] = evaluate(entry['form'],{c:basefeatures[c].values for c in basefeatures.columns},entry['constants'])
        logger.info(f'   Added `{residualfrom}` as input feature (form: {entry["form"]})')
    columns['timeidx'] = np.repeat(np.arange(ntime),ds.sizes['lat']*ds.sizes['lon'])+timeoffset
    features  = pd.DataFrame(columns)
    target    = standardize(flatten(ds[config.targetvar],ntime),config.targetvar,stats)
    validmask = np.isfinite(features.drop(columns=['timeidx'])).all(axis=1).values&np.isfinite(target)
    return features,target,refda,validmask

def subsample_timesteps(features,target,subsetfrac,seed,stats,logmin=-4,logmax=2):
    '''
    Purpose: Draw whole timesteps so that the subset covers the precipitation distribution in proportion. Timesteps are
    binned by their domain-maximum precipitation (a dry bin plus log10 bins from 10^logmin to 10^logmax mm), and each
    bin is sampled in proportion to its size.
    Args:
    - features (pd.DataFrame): predictors including 'timeidx'
    - target (np.ndarray): standardized log1p(precipitation)
    - subsetfrac (float): fraction of timesteps to keep
    - seed (int): random seed
    - stats (dict[str,float]): training statistics
    - logmin (float): log10 lower edge of the wet bins in mm (defaults to -4)
    - logmax (float): log10 upper edge of the wet bins in mm (defaults to 2)
    Returns:
    - tuple[pd.DataFrame, np.ndarray]: subsampled predictors (without 'timeidx') and target
    '''
    precip        = np.expm1(np.asarray(target)*stats['tp_std']+stats['tp_mean'])
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
    selectedtimes = np.unique(np.concatenate(selected))
    subsetindices = np.where(np.isin(timeidx,selectedtimes))[0]
    rng.shuffle(subsetindices)
    return features.iloc[subsetindices].drop(columns=['timeidx']).reset_index(drop=True),np.asarray(target)[subsetindices]

def run_search(xsub,ysub,predictors,srconfig,seed,procs,tmpdir,zmin):
    '''
    Purpose: Run one PySR search.
    Args:
    - xsub (pd.DataFrame): subsampled predictors
    - ysub (np.ndarray): subsampled standardized target
    - predictors (list[str]): predictor names
    - srconfig (dict): SR configuration, with the run's searchparams and complexity overrides merged in
    - seed (int): random seed
    - procs (int): number of Julia workers
    - tmpdir (str): temporary directory for PySR files
    - zmin (float): standardized value of zero precipitation
    Returns:
    - PySRRegressor: fitted model
    '''
    searchparams     = srconfig['searchparams']
    operators        = srconfig['operators']
    complexityparams = srconfig['complexity']
    populations      = searchparams.get('populations',3*procs)
    niterations      = searchparams.get('targettotal',searchparams['iterations']*populations)//populations
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
        timeout_in_seconds=searchparams['timeout'],
        progress=False)
    model.fit(xsub.values,ysub,variable_names=predictors)
    return model

def select_pareto_elbow(equations,mincomplexity=3):
    '''
    Purpose: Select the equation at the elbow of the Pareto frontier.
    Args:
    - equations (pd.DataFrame): Pareto frontier with 'complexity' and 'loss' columns
    - mincomplexity (int): ignore equations simpler than this (defaults to 3)
    Returns:
    - pd.Series: selected row
    '''
    front = equations[equations['complexity']>=mincomplexity].sort_values('complexity').reset_index(drop=True)
    if len(front)==1:
        return front.iloc[0]
    complexitynorm = (front['complexity'].values-front['complexity'].min())/(front['complexity'].max()-front['complexity'].min()+1e-12)
    lossnorm       = (front['loss'].values-front['loss'].min())/(front['loss'].max()-front['loss'].min()+1e-12)
    startpoint     = np.array([complexitynorm[0],lossnorm[0]])
    linerange      = np.array([complexitynorm[-1],lossnorm[-1]])-startpoint
    distances      = [np.abs(np.cross(linerange,startpoint-np.array([complexitynorm[i],lossnorm[i]])))/(np.linalg.norm(linerange)+1e-12) for i in range(len(front))]
    return front.iloc[int(np.argmax(distances))]

def save_equations(model,runname,seed,config):
    '''
    Purpose: Save a search's Pareto frontier as {runname}_{seed}_equations.csv and log its elbow equation.
    Args:
    - model (PySRRegressor): fitted model
    - runname (str): run name
    - seed (int): search seed
    - config (Config): project configuration object
    '''
    outdir        = os.path.join(config.modelsdir,'sr')
    os.makedirs(outdir,exist_ok=True)
    equationspath = os.path.join(outdir,f'{runname}_{seed}_equations.csv')
    dropcols = [c for c in ['sympy_format','lambda_format'] if c in model.equations_.columns]
    model.equations_.drop(columns=dropcols).to_csv(equationspath,index=False)
    pd.read_csv(equationspath)
    best = select_pareto_elbow(model.equations_)
    logger.info(f'   Elbow equation (complexity {int(best["complexity"])}): {best["equation"]}  loss={best["loss"]:.6f}')
    logger.info(f'   Saved to {equationspath}')

if __name__=='__main__':
    config = Config()
    sr     = config.sr
    runs   = sr['runs']
    seeds  = sr['seeds']
    stats  = load_stats(config.splitsdir)
    zmin   = (0.0-stats[f'{config.targetvar}_mean'])/stats[f'{config.targetvar}_std']
    logger.info('Spinning up...')
    selectedruns,procs,iterationsoverride,subsetfracoverride = parse()
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        subsetfrac = subsetfracoverride if subsetfracoverride is not None else sr['subsetfrac']
        if iterationsoverride is not None:
            sr['searchparams']['iterations'] = iterationsoverride
            sr['searchparams'].pop('targettotal',None)
        searchparams = {**sr['searchparams'],**runconfig.get('searchparams',{})}
        complexity   = {**sr['complexity'],'ofvariables':{**sr['complexity']['ofvariables'],**runconfig.get('complexityofvariables',{})}}
        if runconfig.get('residualfrom'):
            complexity['ofvariables'][runconfig['residualfrom']] = 2
        srrun        = {**sr,'searchparams':searchparams,'complexity':complexity}
        populations  = searchparams.get('populations',3*procs)
        niterations  = searchparams.get('targettotal',searchparams['iterations']*populations)//populations
        logger.info(f'Loading normalized training and validation splits for `{name}`...')
        xtrain,ytrain,reftrain,trainmask = load_features('train',runconfig,config,timeoffset=0)
        xvalid,yvalid,_,validmask        = load_features('valid',runconfig,config,timeoffset=int(reftrain.sizes['time']))
        predictors = [c for c in xtrain.columns if c!='timeidx']
        varcomplexities = {p:complexity['ofvariables'].get(p,2) for p in predictors}
        logger.info(f'   Variable complexities: {varcomplexities}')
        xfit = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
        yfit = np.concatenate([ytrain[trainmask],yvalid[validmask]])
        del xtrain,xvalid,ytrain,yvalid,reftrain
        for seedidx,seed in enumerate(seeds):
            equationspath = os.path.join(config.modelsdir,'sr',f'{name}_{seed}_equations.csv')
            if os.path.exists(equationspath):
                logger.info(f'Skipping `{name}` seed {seed}, model already exists')
                continue
            logger.info(f'Running `{name}` seed {seedidx+1}/{len(seeds)} ({seed})...')
            logger.info(f'   Subsampling ~{subsetfrac:.1%} of samples by timestep...')
            xsub,ysub = subsample_timesteps(xfit,yfit,subsetfrac,seed,stats)
            logger.info(f'   Starting PySR search with {niterations} iterations, {populations} populations, and {procs} workers...')
            tempdirpath = tempfile.mkdtemp(prefix='pysr_')
            try:
                model = run_search(xsub,ysub,predictors,srrun,seed,procs,tempdirpath,zmin)
            finally:
                shutil.rmtree(tempdirpath,ignore_errors=True)
            save_equations(model,name,seed,config)
            del model
        del xfit,yfit