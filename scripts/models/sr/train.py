#!/usr/bin/env python

import os
import json
import shutil
import logging
import argparse
import pickle
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from scripts.utils import Config

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def select_pareto_elbow(equations,mincomplexity=3):
    '''
    Purpose: Select the equation at the elbow of the Pareto frontier, where marginal loss
        reduction per unit complexity is largest.
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
    - tuple[set[str]|None, int, int, int|None, float|None]: selected run names (or None for
        all), number of Julia worker processes, search timeout in seconds, and optional
        overrides for iterations and subsetfrac (None means use the value from configs.json)
    '''
    parser = argparse.ArgumentParser(description='Train PySR symbolic regression models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated run names to train, or `all`')
    parser.add_argument('--procs',type=int,default=50,help='Number of Julia worker processes (default: 50)')
    parser.add_argument('--timeout',type=int,default=19800,help='PySR search timeout in seconds (default: 19800)')
    parser.add_argument('--iterations',type=int,default=None,help='Override iterations from config (useful for quick tests)')
    parser.add_argument('--subsetfrac',type=float,default=None,help='Override subsetfrac from config (useful for quick tests)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {n.strip() for n in args.runs.split(',')}
    return selectedruns,args.procs,args.timeout,args.iterations,args.subsetfrac

def kernel_integrate(fields,weights,dsig,mask=None):
    '''
    Purpose: Integrate vertical field profiles using kernel weights and sigma-level thicknesses.
    Args:
    - fields (np.ndarray): profile data with shape (nsamples, nfieldvars, nsig)
    - weights (np.ndarray): kernel weights with shape (nfieldvars, nsig)
    - dsig (np.ndarray): sigma thickness weights with shape (nsig,)
    - mask (np.ndarray | None): surface mask with shape (nsamples, nsig), or None to skip masking
    Returns:
    - np.ndarray: integrated features with shape (nsamples, nfieldvars)
    '''
    weighted = fields*weights[None,:,:]*dsig[None,None,:]
    if mask is not None:
        weighted = weighted*mask[:,None,:]
    return weighted.sum(axis=2)

def load_data(splitname,runconfig,config,time_offset=0):
    '''
    Purpose: Load a normalized data split and construct predictor features for symbolic
        regression. If 'weightsfrom' is set, vertically integrate field variables using kernel
        weights from a previously trained NN model, averaging across seeds; otherwise read
        scalar field variables directly. A 'timeidx' column is appended so that the
        subsampler can select whole timesteps.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - runconfig (dict): run configuration with keys 'fieldvars', 'localvars', and optionally 'weightsfrom'
    - config (Config): project configuration object
    - time_offset (int): added to each time index so that train and valid indices are globally unique
    Returns:
    - tuple[pd.DataFrame, np.ndarray, xr.DataArray, np.ndarray]: predictor features, target
        values, reference DataArray, and boolean mask of finite samples
    '''
    fieldvars   = runconfig['fieldvars']
    localvars   = runconfig.get('localvars',[])
    weightsfrom = runconfig.get('weightsfrom')
    seeds       = config.nn['seeds']
    splitds     = xr.open_dataset(os.path.join(config.splitsdir,f'norm_{splitname}.h5'),engine='h5netcdf')
    refda       = splitds[config.targetvar].transpose('time','lat','lon')
    ntime       = splitds.sizes['time']
    nlat        = splitds.sizes.get('lat',1)
    nlon        = splitds.sizes.get('lon',1)
    columns     = {}
    if weightsfrom:
        nsig         = splitds.sizes['sig']
        dsig         = splitds['dsig'].values
        fieldarrays  = [splitds[var].transpose('time','lat','lon','sig').values.reshape(-1,nsig) for var in fieldvars]
        fieldstack   = np.stack(fieldarrays,axis=1)
        surfmask     = splitds['surfmask'].transpose('time','lat','lon','sig').values.reshape(-1,nsig) if 'surfmask' in splitds else None
        seedfeatures = []
        for seed in seeds:
            weightsds = xr.open_dataset(os.path.join(config.weightsdir,f'{weightsfrom}_{seed}_weights.nc'),engine='h5netcdf')
            seedfeatures.append(kernel_integrate(fieldstack,weightsds['k'].values,dsig,surfmask))
            weightsds.close()
        features = np.mean(seedfeatures,axis=0)
        for i,var in enumerate(fieldvars):
            columns[var] = features[:,i]
    else:
        for var in fieldvars:
            da = splitds[var]
            columns[var] = da.transpose('time','lat','lon').values.ravel() if 'time' in da.dims else np.tile(da.values,(ntime,1,1)).ravel()
    for var in localvars:
        da = splitds[var]
        columns[var] = da.transpose('time','lat','lon').values.ravel() if 'time' in da.dims else np.tile(da.values,(ntime,1,1)).ravel()
    columns['timeidx'] = np.repeat(np.arange(ntime),nlat*nlon)+time_offset
    features  = pd.DataFrame(columns)
    target    = refda.values.ravel()
    validmask = np.isfinite(features.drop(columns=['timeidx'])).all(axis=1).values & np.isfinite(target)
    splitds.close()
    return features,target,refda,validmask

def subsample_timestep(features,target,subsetfrac,seed,logmin=-4,logmax=2):
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
    - logmin (float): log10 lower bound of wet bins in mm (default -4)
    - logmax (float): log10 upper bound of wet bins in mm (default 2)
    Returns:
    - tuple[pd.DataFrame, np.ndarray]: subsampled features (without 'timeidx') and target
    '''
    statsfile = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    precip        = np.expm1(np.asarray(target)*stats['tp_std']+stats['tp_mean'])
    rng           = np.random.default_rng(seed)
    timeidx       = features['timeidx'].values
    uniquetimes,startindices,_ = np.unique(timeidx,return_index=True,return_counts=True)
    sortedprecip  = precip[np.argsort(timeidx,kind='stable')]
    peakprecip    = np.maximum.reduceat(sortedprecip,startindices)
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
        lo,hi = logbins[i],logbins[i+1]
        inbin = uniquetimes[(logpeakprecip>lo)&(logpeakprecip<=hi)]
        if len(inbin)>0:
            binpools.append(inbin)
    totalavailable = sum(len(p) for p in binpools)
    selected       = [drawfrompool(pool,max(1,round(len(pool)/totalavailable*ntimesteps))) for pool in binpools]
    selectedtimes  = np.unique(np.concatenate(selected))
    keep           = np.isin(timeidx,selectedtimes)
    subsetindices  = np.where(keep)[0]
    rng.shuffle(subsetindices)
    return features.iloc[subsetindices].drop(columns=['timeidx']).reset_index(drop=True),np.asarray(target)[subsetindices]

def print_subsample_diagnostics(yfull,ysub,logmin=-4,logmax=2):
    '''
    Purpose: Log a comparative summary of native-unit precipitation distributions between
        the full training pool and the SR subsample, broken down by log-decade bin.
    Args:
    - yfull (np.ndarray): z-scored log1p(tp) for the full pool
    - ysub (np.ndarray): z-scored log1p(tp) for the subsample
    - logmin (float): log10 lower bound of wet bins in mm (default -4)
    - logmax (float): log10 upper bound of wet bins in mm (default 2)
    '''
    statsfile = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    def denorm(y):
        return np.expm1(np.asarray(y)*stats['tp_std']+stats['tp_mean'])
    precipfull = denorm(yfull)
    precipsub  = denorm(ysub)
    nbins      = int(logmax-logmin)
    logbins    = np.linspace(logmin,logmax,nbins+1)
    def bincounts(precip):
        logprecip = np.log10(precip.clip(min=10**(logmin-1)))
        counts    = [(precip<=10**logmin).sum()]
        for i in range(nbins):
            lo,hi = logbins[i],logbins[i+1]
            counts.append(((logprecip>lo)&(logprecip<=hi)).sum())
        counts.append((precip>10**logmax).sum())
        return counts
    labels = [f'dry (<=1e{int(logmin)} mm)']
    for i in range(nbins):
        labels.append(f'1e{int(logbins[i])} to 1e{int(logbins[i+1])} mm')
    labels.append(f'>1e{int(logmax)} mm')
    fullcounts = bincounts(precipfull)
    subcounts  = bincounts(precipsub)
    logger.info('   Subsample diagnostics:')
    logger.info(f'     Pool: {len(yfull):,}   Subset: {len(ysub):,}')
    logger.info(f'     Native tp (mm) pool — mean={precipfull.mean():.4f}  med={np.median(precipfull):.4f}  max={precipfull.max():.4f}')
    logger.info(f'     Native tp (mm) sub  — mean={precipsub.mean():.4f}  med={np.median(precipsub):.4f}  max={precipsub.max():.4f}')
    logger.info(f'     {"Bin":<26} {"Pool":>12} {"Subset":>12}')
    for label,fullcount,subcount in zip(labels,fullcounts,subcounts):
        logger.info(f'     {label:<26} {fullcount:>12,} {subcount:>12,}')

def fit(xsub,ysub,predictors,srconfig,seed,procs,timeout,tmpdir,lossspace='logz'):
    '''
    Purpose: Instantiate and fit a PySRRegressor on the given data subset. Operators,
        complexity penalties, and operator constraints are read from srconfig so they can be
        tuned in configs.json without touching this script. Parallelism is achieved via Julia
        worker processes rather than threads.
    Args:
    - xsub (pd.DataFrame): predictor features with shape (subsetsize, nfeatures)
    - ysub (np.ndarray): target values with shape (subsetsize,)
    - predictors (list[str]): variable names corresponding to columns of xsub
    - srconfig (dict): SR experiment configuration with keys 'searchparams', 'operators',
        'complexity', 'constraints', and 'nestedconstraints'
    - seed (int): random seed for PySR search
    - procs (int): number of Julia worker processes
    - timeout (int): search timeout in seconds; acts as a safety net alongside iterations
    - tmpdir (str): temporary directory for Julia equation files
    - lossspace (str): 'logz' (default) computes MSE in z-scored log1p space;
        'native' denormalizes both prediction and target to mm before computing MSE
    Returns:
    - PySRRegressor: fitted model containing the full Pareto frontier of discovered equations
    '''
    searchparams      = srconfig['searchparams']
    operators         = srconfig['operators']
    complexityparams  = srconfig['complexity']
    constraints       = {k:tuple(v) for k,v in srconfig.get('constraints',{}).items()}
    nestedconstraints = srconfig.get('nestedconstraints',{})
    populations       = searchparams.get('populations',3*procs)
    niterations       = searchparams.get('targettotal',searchparams['iterations']*populations)//populations
    statsfile = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    zfloor = (0.0-stats['tp_mean'])/stats['tp_std']
    if lossspace=='native':
        tpstd  = float(stats['tp_std'])
        tpmean = float(stats['tp_mean'])
        loss = (f'loss(x, y) = (max(expm1(x * {tpstd:.8f} + {tpmean:.8f}), 0.0)'
                f' - expm1(y * {tpstd:.8f} + {tpmean:.8f}))^2')
    else:
        loss = f'loss(x, y) = (max(x, {zfloor:.8f}) - y)^2'
    os.environ.setdefault('JULIA_NUM_THREADS',str(os.cpu_count() or 1))
    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        population_size=searchparams['populationsize'],
        ncycles_per_iteration=searchparams['cyclesperiteration'],
        weight_optimize=searchparams['weightoptimize'],
        parsimony=searchparams.get('parsimony',0.0032),
        binary_operators=operators['binary'],
        unary_operators=operators['unary'],
        complexity_of_operators=operators['complexity'],
        complexity_of_variables=[complexityparams['ofvariables'].get(p,1) for p in predictors]
            if isinstance(complexityparams['ofvariables'],dict) else complexityparams['ofvariables'],
        complexity_of_constants=complexityparams['ofconstants'],
        maxsize=searchparams['maxsize'],
        maxdepth=searchparams['maxdepth'],
        constraints=constraints,
        nested_constraints=nestedconstraints,
        extra_sympy_mappings={'square':lambda x:x**2},
        loss=loss,
        model_selection='best',
        batching=True,
        batch_size=searchparams['batchsize'],
        random_state=seed,
        parallelism='multithreading',
        procs=procs,
        tempdir=tmpdir,
        temp_equation_file=True,
        delete_tempfiles=True,
        timeout_in_seconds=timeout,
        progress=False)
    model.fit(xsub.values,ysub,variable_names=predictors)
    return model

def save(model,runname,seed,config):
    '''
    Purpose: Save a fitted PySRRegressor and its equation Pareto frontier to disk.
    Args:
    - model (PySRRegressor): fitted symbolic regression model
    - runname (str): run identifier used for output filenames
    - seed (int): training seed used for output filenames
    - config (Config): project configuration object
    '''
    outdir       = os.path.join(config.modelsdir,'sr')
    os.makedirs(outdir,exist_ok=True)
    paretopath   = os.path.join(outdir,f'{runname}_{seed}_pareto.pkl')
    equationspath = os.path.join(outdir,f'{runname}_{seed}_equations.csv')
    with open(paretopath,'wb') as f:
        pickle.dump(model,f)
    dropcols = [c for c in ['sympy_format','lambda_format'] if c in model.equations_.columns]
    model.equations_.drop(columns=dropcols).to_csv(equationspath,index=False)
    best = select_pareto_elbow(model.equations_)
    logger.info(f'   Elbow equation (complexity {int(best["complexity"])}): {best["equation"]}  loss={best["loss"]:.6f}')
    logger.info(f'   Saved to {paretopath}')

if __name__=='__main__':
    config = Config()
    sr     = config.sr
    runs   = sr['runs']
    seeds  = sr['seeds']
    logger.info('Spinning up...')
    selectedruns,procs,timeout,iterationsoverride,subsetfracoverride = parse()
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        fieldvars  = runconfig['fieldvars']
        localvars  = runconfig.get('localvars',[])
        predictors = fieldvars+localvars
        subsetfrac = subsetfracoverride if subsetfracoverride is not None else sr['subsetfrac']
        if iterationsoverride is not None:
            sr['searchparams']['iterations'] = iterationsoverride
            sr['searchparams'].pop('targettotal',None)
        searchparams = {**sr['searchparams'], **runconfig.get('searchparams',{})}
        srrun        = {**sr, 'searchparams': searchparams}
        populations  = searchparams.get('populations',3*procs)
        niterations  = searchparams.get('targettotal',searchparams['iterations']*populations)//populations
        logger.info(f'Loading normalized training and validation splits for `{name}`...')
        xtrain,ytrain,reftrain,trainmask = load_data('train',runconfig,config,time_offset=0)
        xvalid,yvalid,_,validmask        = load_data('valid',runconfig,config,time_offset=int(reftrain.sizes['time']))
        xfit = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
        yfit = np.concatenate([ytrain[trainmask],yvalid[validmask]])
        del xtrain,xvalid,ytrain,yvalid,reftrain
        for seedidx,seed in enumerate(seeds):
            paretopath = os.path.join(config.modelsdir,'sr',f'{name}_{seed}_pareto.pkl')
            if os.path.exists(paretopath):
                logger.info(f'Skipping `{name}` seed {seed}, model already exists')
                continue
            logger.info(f'Running `{name}` seed {seedidx+1}/{len(seeds)} ({seed})...')
            logger.info(f'   Subsampling ~{subsetfrac:.1%} of samples by timestep...')
            xsub,ysub = subsample_timestep(xfit,yfit,subsetfrac,seed)
            print_subsample_diagnostics(yfit,ysub)
            logger.info(f'   Starting PySR search ({niterations} iters × {populations} populations, {procs} workers, {timeout}s timeout)...')
            tempdirpath = tempfile.mkdtemp(prefix='pysr_')
            try:
                model = fit(xsub,ysub,predictors,srrun,seed,procs,timeout,tempdirpath,
                            lossspace=runconfig.get('lossspace','logz'))
            finally:
                shutil.rmtree(tempdirpath,ignore_errors=True)
            save(model,name,seed,config)
            del model
        del xfit,yfit
