#!/usr/bin/env python

import os
import json
import shutil
import logging
import argparse
import tempfile
import warnings
import numpy as np
import pandas as pd
from scripts.utils import Config,load
from scripts.models.sr.equations import evaluate,load_registry

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)

def select_pareto_elbow(equations,mincomplexity=3):
    '''
    Purpose: Select the equation at the elbow of the Pareto frontier.
    Args:
    - equations (pd.DataFrame): model.equations_ with 'complexity' and 'loss' columns
    - mincomplexity (int): ignore equations simpler than this (avoids trivial picks)
    Returns:
    - pd.Series: the selected row from the equations DataFrame
    '''
    front = equations[equations['complexity']>=mincomplexity].copy()
    front = front.sort_values('complexity').reset_index(drop=True)
    if len(front)==1:
        return front.iloc[0]
    complexityvals = front['complexity'].values.astype(float)
    lossvals       = front['loss'].values.astype(float)
    complexitynorm = (complexityvals-complexityvals.min())/(complexityvals.max()-complexityvals.min()+1e-12)
    lossnorm       = (lossvals-lossvals.min())/(lossvals.max()-lossvals.min()+1e-12)
    startpoint     = np.array([complexitynorm[0],lossnorm[0]])
    endpoint       = np.array([complexitynorm[-1],lossnorm[-1]])
    linerange      = endpoint-startpoint
    linelength     = np.linalg.norm(linerange)
    distances      = [np.abs(np.cross(linerange,startpoint-np.array([complexitynorm[i],lossnorm[i]])))/(linelength+1e-12) for i in range(len(front))]
    elbowindex     = int(np.argmax(distances))
    return front.iloc[elbowindex]

def parse():
    '''
    Purpose: Parse command-line arguments for running the training script.
    Returns:
    - tuple[set[str]|None, int, int|None, float|None]: selected run names (or None for
        all), number of Julia worker processes, and optional overrides for iterations and
        subsetfrac (None means use the value from configs.json)
    '''
    parser = argparse.ArgumentParser(description='Train PySR symbolic regression models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated run names to train, or `all`')
    parser.add_argument('--procs',type=int,default=50,help='Number of Julia worker processes (default: 50)')
    parser.add_argument('--iterations',type=int,default=None,help='Override iterations from config (useful for quick tests)')
    parser.add_argument('--subsetfrac',type=float,default=None,help='Override subsetfrac from config (useful for quick tests)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {n.strip() for n in args.runs.split(',')}
    return selectedruns,args.procs,args.iterations,args.subsetfrac

def load_stats(config):
    '''
    Purpose: Load training statistics from the configured splits directory.
    Args:
    - config (Config): project configuration object
    Returns:
    - dict[str,float]: training statistics
    '''
    with open(os.path.join(config.splitsdir,'stats.json'),'r',encoding='utf-8') as f:
        return json.load(f)

def load_kernels(config,weightsfrom,dsig):
    '''
    Purpose: Load NN kernel weights for every seed, renormalize each so that sum(k·Δσ) = 1 in float64, and average
        across seeds.
    Args:
    - config (Config): project configuration object
    - weightsfrom (str): NN run whose kernel weights are used
    - dsig (np.ndarray): sigma thickness weights with shape (nsig,)
    Returns:
    - dict[str,np.ndarray]: field variable → kernel weights with shape (nsig,)
    '''
    kernels = []
    for seed in config.nn['seeds']:
        k = load(os.path.join(config.weightsdir,f'{weightsfrom}_{seed}_weights.nc'))['k']
        kernels.append(k.values/(k.values*dsig[None,:]).sum(axis=1,keepdims=True))
        fieldnames = [str(field) for field in k['field'].values]
    return dict(zip(fieldnames,np.mean(kernels,axis=0)))

def load_data(splitname,runconfig,config,time_offset=0):
    '''
    Purpose: Load SR predictors and the target for one split. Profile variables are kernel-integrated in physical units
        (if weightsfrom) and then standardized with training statistics, like all other predictors except land
        fraction; the target is standardized log1p(precipitation). A prior SR equation (residualfrom) is added as a
        predictor.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - runconfig (dict): SR run configuration
    - config (Config): project configuration object
    - time_offset (int): offset added to the time index (to keep train and valid timesteps distinct)
    Returns:
    - tuple[pd.DataFrame,np.ndarray,xr.DataArray,np.ndarray]: features (with 'timeidx'), target, reference
        DataArray with (time, lat, lon) coordinates, and valid-sample mask
    '''
    fieldvars   = runconfig['fieldvars']
    localvars   = runconfig.get('localvars',[])
    weightsfrom = runconfig.get('weightsfrom')
    stats   = load_stats(config)
    splitds = load(os.path.join(config.splitsdir,f'{splitname}.h5'))
    refda   = splitds[config.targetvar].transpose('time','lat','lon')
    ntime   = splitds.sizes['time']
    nlat    = splitds.sizes.get('lat',1)
    nlon    = splitds.sizes.get('lon',1)
    def flatten(var):
        da = splitds[var]
        return da.transpose('time','lat','lon').values.ravel() if 'time' in da.dims else np.tile(da.values,(ntime,1,1)).ravel()
    def standardize(values,var):
        return values if var=='lf' else (values-stats[f'{var}_mean'])/stats[f'{var}_std']
    columns = {}
    if weightsfrom and fieldvars:
        dsig    = splitds['dsig'].values
        kernels = load_kernels(config,weightsfrom,dsig)
        for var in fieldvars:
            profiles = splitds[var].transpose('time','lat','lon','sig').values.reshape(-1,splitds.sizes['sig'])
            columns[var] = standardize(profiles@(kernels[var]*dsig),var)
    else:
        for var in fieldvars:
            columns[var] = standardize(flatten(var),var)
    for var in localvars:
        columns[var] = standardize(flatten(var),var)
    residualfrom = runconfig.get('residualfrom')
    if residualfrom:
        entry = load_registry(config.modelsdir)[residualfrom]
        baserunconfig = config.sr['runs'][config.sr['optimizedeqs'][residualfrom]['runfrom']]
        basefeatures,_,_,_ = load_data(splitname,baserunconfig,config,time_offset=time_offset)
        columns[residualfrom] = evaluate(entry['form'],{c:basefeatures[c].values for c in basefeatures.columns},entry['constants'])
        logger.info(f'   Added `{residualfrom}` as input feature (form: {entry["form"]})')
    columns['timeidx'] = np.repeat(np.arange(ntime),nlat*nlon)+time_offset
    features  = pd.DataFrame(columns)
    target    = (np.log1p(refda.values.ravel())-stats[f'{config.targetvar}_mean'])/stats[f'{config.targetvar}_std']
    validmask = np.isfinite(features.drop(columns=['timeidx'])).all(axis=1).values&np.isfinite(target)
    return features,target,refda,validmask

def subsample_timestep(features,target,subsetfrac,seed,stats,logmin=-4,logmax=2):
    '''
    Purpose: Subsample complete timesteps with proportional coverage of the precipitation
        distribution. Timesteps are grouped by their domain-maximum precipitation and drawn
        from each log-decade bin in proportion to its share of the full dataset. All valid
        spatial points within each selected timestep are retained.
    Args:
    - features (pd.DataFrame): predictor features including a 'timeidx' column added by load_data
    - target (np.ndarray): z-scored log1p(tp) target values with shape (nsamples,)
    - subsetfrac (float): target fraction of total available samples
    - seed (int): random seed for reproducibility
    - stats (dict[str,float]): training statistics
    - logmin (float): log10 lower bound of wet bins in mm (default -4)
    - logmax (float): log10 upper bound of wet bins in mm (default 2)
    Returns:
    - tuple[pd.DataFrame, np.ndarray]: subsampled features (without 'timeidx') and target
    '''
    precip        = np.expm1(np.asarray(target)*stats['tp_std']+stats['tp_mean'])
    rng           = np.random.default_rng(seed)
    timeidx       = features['timeidx'].values
    uniquetimes,startindices = np.unique(timeidx,return_index=True)
    sort_order    = np.argsort(timeidx,kind='stable')
    peakprecip    = np.maximum.reduceat(precip[sort_order],startindices)
    nbins         = int(logmax-logmin)
    ntimesteps    = max(1,int(round(subsetfrac*len(uniquetimes))))
    logbins       = np.linspace(logmin,logmax,nbins+1)
    logpeakprecip = np.log10(peakprecip.clip(min=10**(logmin-1)))
    drymask       = peakprecip<=10**logmin
    def drawfrompool(pool,n):
        return rng.choice(pool,n,replace=len(pool)<n)
    binpools = []
    if drymask.any():
        binpools.append(uniquetimes[drymask])
    for i in range(nbins):
        lo,hi  = logbins[i],logbins[i+1]
        pool   = uniquetimes[(logpeakprecip>lo)&(logpeakprecip<=hi)]
        if len(pool)>0:
            binpools.append(pool)
    totalavailable = sum(len(p) for p in binpools)
    selected       = [drawfrompool(pool,max(1,round(len(pool)/totalavailable*ntimesteps))) for pool in binpools]
    selectedtimes  = np.unique(np.concatenate(selected))
    keep           = np.isin(timeidx,selectedtimes)
    subsetindices  = np.where(keep)[0]
    rng.shuffle(subsetindices)
    return features.iloc[subsetindices].drop(columns=['timeidx']).reset_index(drop=True),np.asarray(target)[subsetindices]

TIMEOUT = 19800

def fit(xsub,ysub,predictors,srconfig,seed,procs,tmpdir,zmin):
    '''
    Purpose: Run a PySR search.
    Args:
    - xsub (pd.DataFrame): subsampled predictors
    - ysub (np.ndarray): subsampled target
    - predictors (list[str]): predictor names
    - srconfig (dict): SR configuration with run-level searchparams and complexity overrides merged in
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
    constraints       = {k:tuple(v) for k,v in srconfig.get('constraints',{}).items()}
    nestedconstraints = srconfig.get('nestedconstraints',{})
    populations       = searchparams.get('populations',3*procs)
    niterations       = searchparams.get('targettotal',searchparams['iterations']*populations)//populations
    loss = 'loss(x, y) = (x - y)^2' if searchparams.get('loss') == 'plainmse' else f'loss(x, y) = (({zmin:.8f}) + max(x, 0.0) - y)^2'
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
        constraints=constraints,
        nested_constraints=nestedconstraints,
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

def save_equations(model,runname,seed,config):
    '''
    Purpose: Save a fitted PySRRegressor's equation Pareto frontier to disk as CSV.
    Args:
    - model (PySRRegressor): fitted symbolic regression model
    - runname (str): run identifier used for output filenames
    - seed (int): training seed used for output filenames
    - config (Config): project configuration object
    '''
    outdir        = os.path.join(config.modelsdir,'sr')
    os.makedirs(outdir,exist_ok=True)
    equationspath = os.path.join(outdir,f'{runname}_{seed}_equations.csv')
    dropcols = [c for c in ['sympy_format','lambda_format'] if c in model.equations_.columns]
    model.equations_.drop(columns=dropcols).to_csv(equationspath,index=False)
    best = select_pareto_elbow(model.equations_)
    logger.info(f'   Elbow equation (complexity {int(best["complexity"])}): {best["equation"]}  loss={best["loss"]:.6f}')
    logger.info(f'   Saved to {equationspath}')

if __name__=='__main__':
    config = Config()
    sr     = config.sr
    runs   = sr['runs']
    seeds  = sr['seeds']
    stats  = load_stats(config)
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
        srrun        = {**sr,'searchparams':searchparams,'complexity':complexity}
        populations  = searchparams.get('populations',3*procs)
        niterations  = searchparams.get('targettotal',searchparams['iterations']*populations)//populations
        logger.info(f'Loading normalized training and validation splits for `{name}`...')
        xtrain,ytrain,reftrain,trainmask = load_data('train',runconfig,config,time_offset=0)
        xvalid,yvalid,_,validmask       = load_data('valid',runconfig,config,time_offset=int(reftrain.sizes['time']))
        predictors = [c for c in xtrain.columns if c != 'timeidx']
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
            xsub,ysub = subsample_timestep(xfit,yfit,subsetfrac,seed,stats)
            logger.info(f'   Starting PySR search with {niterations} iterations, {populations} populations, and {procs} workers...')
            tempdirpath = tempfile.mkdtemp(prefix='pysr_')
            try:
                model = fit(xsub,ysub,predictors,srrun,seed,procs,tempdirpath,zmin)
            finally:
                shutil.rmtree(tempdirpath,ignore_errors=True)
            save_equations(model,name,seed,config)
            del model
        del xfit,yfit