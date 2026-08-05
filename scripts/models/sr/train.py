#!/usr/bin/env python

import os
import json
import shutil
import pickle
import logging
import argparse
import tempfile
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scripts.utils import Config

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

SRFUNCTIONS = {
    'cube':  lambda x: x**3,
    'square':lambda x: x**2,
    'neg':   lambda x: -x,
    'sqrt':  np.sqrt,
    'exp':   np.exp,
    'log':   np.log,
    'abs':   np.abs,
    'sin':   np.sin,
    'cos':   np.cos,
    'max':   np.maximum,
    'min':   np.minimum}

def eval_baseline(form,columns,constants):
    '''
    Purpose: Evaluate an SR equation form over a dict of named numpy arrays.
    Args:
    - form (str): Python expression string (e.g., 'a * cube(max(rh, thetae - b * thetaestar - c))')
    - columns (dict[str, np.ndarray]): mapping from variable name to flat array; 'timeidx' is skipped
    - constants (dict[str, float]): mapping from constant name to value
    Returns:
    - np.ndarray: evaluated result with the same length as the input arrays
    '''
    ns = dict(SRFUNCTIONS,__builtins__={})
    for col,vals in columns.items():
        if col != 'timeidx':
            ns[col] = np.asarray(vals,dtype=float)
    ns.update(constants)
    out = eval(form,ns)
    if np.ndim(out) == 0:
        n = len(next(v for v in columns.values() if hasattr(v,'__len__')))
        out = np.full(n,float(out))
    return np.asarray(out,dtype=float)

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
    fieldvars    = runconfig['fieldvars']
    localvars    = runconfig.get('localvars',[])
    weightsfrom  = runconfig.get('weightsfrom')
    rotatefields = runconfig.get('rotatefields',{})
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
        for newvar,spec in rotatefields.items():
            w = np.array(spec['weights'])
            columns[newvar] = sum(wi*columns[v] for wi,v in zip(w,spec['vars']))
            for v in spec['vars']:
                del columns[v]
    else:
        for var in fieldvars:
            da = splitds[var]
            columns[var] = da.transpose('time','lat','lon').values.ravel() if 'time' in da.dims else np.tile(da.values,(ntime,1,1)).ravel()
    for var in localvars:
        da = splitds[var]
        columns[var] = da.transpose('time','lat','lon').values.ravel() if 'time' in da.dims else np.tile(da.values,(ntime,1,1)).ravel()
    baselinefrom     = runconfig.get('baselinefrom')
    baselinemode     = runconfig.get('baselinemode','feature')
    gapfrom          = runconfig.get('gapfrom')
    targetfrom       = runconfig.get('targetfrom')
    srmed_for_target = None
    srmed_raw        = None
    if baselinefrom:
        baselinespec      = config.sr['optimizedeqs'][baselinefrom]
        baselineform      = baselinespec['form']
        registrypath      = os.path.join(config.modelsdir,'sr','optimized_equations.pkl')
        if os.path.exists(registrypath):
            with open(registrypath,'rb') as f:
                registry = pickle.load(f)
            baselineconstants = registry.get(baselinefrom,{}).get('constants') or baselinespec['init']
        else:
            baselineconstants = baselinespec['init']
        srmed_raw = eval_baseline(baselineform,columns,baselineconstants)
        if baselinemode == 'multiplicative':
            srmed_for_target = srmed_raw
        else:
            columns['srmed'] = srmed_raw
            for var in fieldvars:
                columns.pop(var,None)
    columns['timeidx'] = np.repeat(np.arange(ntime),nlat*nlon)+time_offset
    features  = pd.DataFrame(columns)
    target    = refda.values.ravel()
    validmask = np.isfinite(features.drop(columns=['timeidx'])).all(axis=1).values & np.isfinite(target)
    if srmed_for_target is not None or gapfrom or targetfrom:
        statsfile = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
        with open(statsfile,'r',encoding='utf-8') as f:
            stats = json.load(f)
        tp_mean = stats[f'{config.targetvar}_mean']
        tp_std  = stats[f'{config.targetvar}_std']
        zmin    = (0.0 - tp_mean) / tp_std
    if srmed_for_target is not None:
        validmask = validmask & (srmed_for_target > 0)
        target    = target.copy()
        target[validmask] = (target[validmask] - zmin) / srmed_for_target[validmask]
    if gapfrom:
        predsfile = os.path.join(config.predsdir,f'{gapfrom}_{splitname}_predictions.nc')
        with xr.open_dataset(predsfile,engine='h5netcdf') as predsds:
            nn_pred_mm = predsds[config.targetvar].transpose('time','lat','lon','seed').mean('seed').values.ravel()
        nn_pred_z  = (np.log1p(nn_pred_mm) - tp_mean) / tp_std
        if srmed_raw is None:
            raise ValueError('`gapfrom` requires `baselinefrom` to also be set in the run config')
        raw_target = refda.values.ravel()
        gap_score  = np.abs(raw_target - srmed_raw) - np.abs(raw_target - nn_pred_z)
        features['__gapscore__'] = np.where(np.isfinite(gap_score),gap_score,0.0)
    if targetfrom:
        predsfile = os.path.join(config.predsdir,f'{targetfrom}_{splitname}_predictions.nc')
        with xr.open_dataset(predsfile,engine='h5netcdf') as predsds:
            nn_pred_mm = predsds[config.targetvar].transpose('time','lat','lon','seed').mean('seed').values.ravel()
        target    = (np.log1p(nn_pred_mm) - tp_mean) / tp_std
        validmask = validmask & np.isfinite(target)
    splitds.close()
    return features,target,refda,validmask

def subsample_timestep(features,target,subsetfrac,seed,logmin=-4,logmax=2,gaptemp=1.0):
    '''
    Purpose: Subsample complete timesteps with proportional coverage of the precipitation
        distribution. Timesteps are grouped by their domain-maximum precipitation and drawn
        from each log-decade bin in proportion to its share of the full dataset. If a
        '__gapscore__' column is present in features, sampling within each bin is biased
        toward timesteps with higher mean learnability gap using softmax weighting. All valid
        spatial points within each selected timestep are retained.
    Args:
    - features (pd.DataFrame): predictor features including a 'timeidx' column added by load_data,
        and optionally a '__gapscore__' column added when gapfrom is configured
    - target (np.ndarray): z-scored log1p(tp) target values with shape (nsamples,)
    - subsetfrac (float): target fraction of total available samples
    - seed (int): random seed for reproducibility
    - logmin (float): log10 lower bound of wet bins in mm (default -4)
    - logmax (float): log10 upper bound of wet bins in mm (default 2)
    - gaptemp (float): softmax temperature for gap-weighted sampling within bins; higher values
        approach uniform sampling, lower values concentrate on highest-gap timesteps (default 1.0)
    Returns:
    - tuple[pd.DataFrame, np.ndarray]: subsampled features (without 'timeidx' or '__gapscore__') and target
    '''
    statsfile = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    precip        = np.expm1(np.asarray(target)*stats['tp_std']+stats['tp_mean'])
    rng           = np.random.default_rng(seed)
    timeidx       = features['timeidx'].values
    hasgap        = '__gapscore__' in features.columns
    uniquetimes,startindices,timecounts = np.unique(timeidx,return_index=True,return_counts=True)
    sort_order    = np.argsort(timeidx,kind='stable')
    sortedprecip  = precip[sort_order]
    peakprecip    = np.maximum.reduceat(sortedprecip,startindices)
    if hasgap:
        sorted_gaps  = np.maximum(features['__gapscore__'].values[sort_order],0.0)
        gap_per_time = np.add.reduceat(sorted_gaps,startindices) / timecounts
        timegapmap   = dict(zip(uniquetimes,gap_per_time))
    nbins         = int(logmax-logmin)
    ntimesteps    = max(1,int(round(subsetfrac*len(uniquetimes))))
    logbins       = np.linspace(logmin,logmax,nbins+1)
    logpeakprecip = np.log10(peakprecip.clip(min=10**(logmin-1)))
    drymask       = peakprecip<=10**logmin
    def drawfrompool(pool,n,gaps=None):
        if gaps is not None and gaps.sum() > 0:
            logits = gaps / gaptemp
            logits = logits - logits.max()
            probs  = np.exp(logits)
            probs  = probs / probs.sum()
            return rng.choice(pool,n,replace=len(pool)<n,p=probs)
        return rng.choice(pool,n,replace=len(pool)<n)
    binpools = []
    if drymask.any():
        pool = uniquetimes[drymask]
        gaps = np.array([timegapmap[t] for t in pool]) if hasgap else None
        binpools.append((pool,gaps))
    for i in range(nbins):
        lo,hi = logbins[i],logbins[i+1]
        pool  = uniquetimes[(logpeakprecip>lo)&(logpeakprecip<=hi)]
        if len(pool)>0:
            gaps = np.array([timegapmap[t] for t in pool]) if hasgap else None
            binpools.append((pool,gaps))
    totalavailable = sum(len(p) for p,_ in binpools)
    selected       = [drawfrompool(pool,max(1,round(len(pool)/totalavailable*ntimesteps)),gaps) for pool,gaps in binpools]
    selectedtimes  = np.unique(np.concatenate(selected))
    keep           = np.isin(timeidx,selectedtimes)
    subsetindices  = np.where(keep)[0]
    rng.shuffle(subsetindices)
    dropcols = ['timeidx']+(['__gapscore__'] if hasgap else [])
    return features.iloc[subsetindices].drop(columns=dropcols).reset_index(drop=True),np.asarray(target)[subsetindices]

def fit(xsub,ysub,predictors,srconfig,seed,procs,timeout,tmpdir):
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
    zmin = (0.0-stats['tp_mean'])/stats['tp_std']
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
        predictors = [c for c in xtrain.columns if c not in ('timeidx','__gapscore__')]
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
            gaptemp = runconfig.get('gaptemp',1.0)
            xsub,ysub = subsample_timestep(xfit,yfit,subsetfrac,seed,gaptemp=gaptemp)
            logger.info(f'   Starting PySR search with {niterations} iterations, {populations} populations, and {procs} workers...')
            tempdirpath = tempfile.mkdtemp(prefix='pysr_')
            try:
                model = fit(xsub,ysub,predictors,srrun,seed,procs,timeout,tempdirpath)
            finally:
                shutil.rmtree(tempdirpath,ignore_errors=True)
            save(model,name,seed,config)
            del model
        del xfit,yfit