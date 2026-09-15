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
    'min':   np.minimum,
    '_safepow':lambda a,b: np.abs(a)**b}

def _prepare_form(form):
    import re
    return re.sub(r'(\w+)\^(\w+)',r'_safepow(\1,\2)',form)

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
    out = eval(_prepare_form(form),ns)
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
    if weightsfrom and fieldvars:
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
    columns['timeidx'] = np.repeat(np.arange(ntime),nlat*nlon)+time_offset
    features  = pd.DataFrame(columns)
    target    = refda.values.ravel()
    targetfrom = runconfig.get('targetfrom')
    if targetfrom:
        statsfile = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
        with open(statsfile,'r',encoding='utf-8') as f:
            stats = json.load(f)
        predpath = os.path.join(config.predsdir,f'{targetfrom}_{splitname}_predictions.nc')
        with xr.open_dataset(predpath) as pds:
            predtp = pds.tp.load()
        if 'seed' in predtp.dims:predtp = predtp.mean('seed')
        predtp = predtp.transpose('time','lat','lon')
        target = (np.log1p(predtp.values.clip(min=0).ravel())-stats['tp_mean'])/stats['tp_std']
    residualfrom = runconfig.get('residualfrom')
    if residualfrom:
        registrypath = os.path.join(config.modelsdir,'sr','optimized_equations.pkl')
        with open(registrypath,'rb') as f:
            registry = pickle.load(f)
        entry = registry[residualfrom]
        eqspec = config.sr['optimizedeqs'][residualfrom]
        baserunconfig = config.sr['runs'][eqspec['runfrom']]
        basefeatures,_,_,_ = load_data(splitname,baserunconfig,config,time_offset=time_offset)
        basecols = {c:basefeatures[c].values for c in basefeatures.columns if c != 'timeidx'}
        baseline = eval_baseline(entry['form'],basecols,entry['constants'])
        features[residualfrom] = baseline
        logger.info(f'   Added `{residualfrom}` as input feature (form: {entry["form"]})')
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

def compute_error_weights(config,runconfig,trainmask,validmask,ntraintimes):
    errorsampling = runconfig.get('errorsampling')
    if not errorsampling:
        return None
    baseline = errorsampling.get('baseline')
    if not baseline:
        logger.warning('   errorsampling.baseline not specified; falling back to uniform sampling')
        return None
    statsfile = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    nnmodel = errorsampling['nn']
    registrypath = os.path.join(config.modelsdir,'sr','optimized_equations.pkl')
    with open(registrypath,'rb') as f:
        registry = pickle.load(f)
    entry = registry[baseline]
    eqspec = config.sr['optimizedeqs'][baseline]
    baserunconfig = config.sr['runs'][eqspec['runfrom']]
    allnn = []
    allsr = []
    for split,offset,mask in [('train',0,trainmask),('valid',ntraintimes,validmask)]:
        nnpath = os.path.join(config.predsdir,f'{nnmodel}_{split}_predictions.nc')
        with xr.open_dataset(nnpath) as pds:
            nnpred = pds.tp.load()
        if 'seed' in nnpred.dims:nnpred = nnpred.mean('seed')
        nnpred = nnpred.transpose('time','lat','lon')
        nnmm = nnpred.values.ravel()
        nnz = (np.log1p(np.maximum(nnmm,0.0))-stats['tp_mean'])/stats['tp_std']
        allnn.append(nnz[mask])
        basex,_,_,_ = load_data(split,baserunconfig,config,time_offset=offset)
        basecols = {c:basex[c].values for c in basex.columns if c != 'timeidx'}
        srz = eval_baseline(entry['form'],basecols,entry['constants'])
        allsr.append(srz[mask])
    nnall = np.concatenate(allnn)
    srall = np.concatenate(allsr)
    zmin = (0.0-stats['tp_mean'])/stats['tp_std']
    nnmm = np.maximum(np.expm1((zmin+np.maximum(nnall,0.0))*stats['tp_std']+stats['tp_mean']),0.0)
    srmm = np.maximum(np.expm1((zmin+np.maximum(srall,0.0))*stats['tp_std']+stats['tp_mean']),0.0)
    error = np.abs(nnmm-srmm)
    error = np.nan_to_num(error,nan=0.0)
    p99 = np.percentile(error,99)
    weights = (error/(p99+1e-12)).clip(max=1.0)
    logger.info(f'   Error weights: mean={error.mean():.4f} mm, p90={np.percentile(error,90):.4f} mm, p99={p99:.4f} mm')
    return weights

def subsample_errorweighted(features,target,subsetfrac,seed,weights,alpha=5.0):
    rng = np.random.default_rng(seed)
    n = len(target)
    nsamp = max(1,int(round(subsetfrac*n)))
    prob = 1.0+alpha*weights
    prob = prob/prob.sum()
    indices = rng.choice(n,nsamp,replace=False,p=prob)
    rng.shuffle(indices)
    return features.iloc[indices].drop(columns=['timeidx']).reset_index(drop=True),np.asarray(target)[indices]

TIMEOUT = 19800

def build_guesses(runconfig,predictors):
    return runconfig.get('guesses',[])

def fit(xsub,ysub,predictors,srconfig,runconfig,seed,procs,tmpdir):
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
    guesses = build_guesses(runconfig,predictors)
    os.environ.setdefault('JULIA_NUM_THREADS',str(os.cpu_count() or 1))
    from pysr import PySRRegressor
    kwargs = dict(
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
    if guesses:
        kwargs['guesses'] = guesses
        kwargs['fraction_replaced_guesses'] = searchparams.get('fractionreplacedguesses',0.01)
        logger.info(f'   Seeding search with {len(guesses)} guess(es)')
    model = PySRRegressor(**kwargs)
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
    selectedruns,procs,iterationsoverride,subsetfracoverride = parse()
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
        xvalid,yvalid,_,validmask       = load_data('valid',runconfig,config,time_offset=int(reftrain.sizes['time']))
        predictors = [c for c in xtrain.columns if c != 'timeidx']
        errorweights = compute_error_weights(config,runconfig,trainmask,validmask,int(reftrain.sizes['time']))
        xfit = pd.concat([xtrain[trainmask],xvalid[validmask]]).reset_index(drop=True)
        yfit = np.concatenate([ytrain[trainmask],yvalid[validmask]])
        del xtrain,xvalid,ytrain,yvalid,reftrain
        for seedidx,seed in enumerate(seeds):
            paretopath = os.path.join(config.modelsdir,'sr',f'{name}_{seed}_pareto.pkl')
            if os.path.exists(paretopath):
                logger.info(f'Skipping `{name}` seed {seed}, model already exists')
                continue
            logger.info(f'Running `{name}` seed {seedidx+1}/{len(seeds)} ({seed})...')
            if errorweights is not None:
                alpha = runconfig.get('errorsampling',{}).get('alpha',5)
                logger.info(f'   Error-weighted subsampling ~{subsetfrac:.1%} of samples (alpha={alpha})...')
                xsub,ysub = subsample_errorweighted(xfit,yfit,subsetfrac,seed,errorweights,alpha=alpha)
            else:
                logger.info(f'   Subsampling ~{subsetfrac:.1%} of samples by timestep...')
                xsub,ysub = subsample_timestep(xfit,yfit,subsetfrac,seed)
            logger.info(f'   Starting PySR search with {niterations} iterations, {populations} populations, and {procs} workers...')
            tempdirpath = tempfile.mkdtemp(prefix='pysr_')
            try:
                model = fit(xsub,ysub,predictors,srrun,runconfig,seed,procs,tempdirpath)
            finally:
                shutil.rmtree(tempdirpath,ignore_errors=True)
            save(model,name,seed,config)
            del model
        del xfit,yfit