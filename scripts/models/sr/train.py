#!/usr/bin/env python

import os
os.environ.setdefault('JULIA_NUM_THREADS','1')

import json
import shutil
import logging
import warnings
warnings.filterwarnings('ignore')
import tempfile
import argparse
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from pysr import PySRRegressor
from scripts.utils import Config

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def select_pareto_elbow(equations,mincomplexity=3):
    '''
    Purpose: Select the equation at the elbow of the Pareto frontier, where marginal loss reduction per unit complexity is largest.
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
    c       = front['complexity'].values.astype(float)
    l       = front['loss'].values.astype(float)
    cnorm   = (c-c.min())/(c.max()-c.min()+1e-12)
    lnorm   = (l-l.min())/(l.max()-l.min()+1e-12)
    p1      = np.array([cnorm[0],lnorm[0]])
    p2      = np.array([cnorm[-1],lnorm[-1]])
    line    = p2-p1
    linelen = np.linalg.norm(line)
    distances = [np.abs(np.cross(line,p1-np.array([cnorm[i],lnorm[i]])))/(linelen+1e-12) for i in range(len(front))]
    elbowix = int(np.argmax(distances))
    return front.iloc[elbowix]

def parse():
    '''
    Purpose: Parse command-line arguments for running the training script.
     - set[str] | None: run names to train, or None if all runs should be trained
    Returns:
    - tuple[set[str]|None,int,int,int|None,int None]: selected run names (or None for
        all), number of Julia worker processes, search timeout in seconds, and optional overrides
        for iterations and subsetsize (None means use the value from configs.json)
    '''
    parser = argparse.ArgumentParser(description='Train PySR symbolic regression models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated run names to train, or `all`')
    parser.add_argument('--procs',type=int,default=50,help='Number of Julia worker processes (default: 50)')
    parser.add_argument('--timeout',type=int,default=19800,help='PySR search timeout in seconds (default: 19800)')
    parser.add_argument('--iterations',type=int,default=None,help='Override iterations from config (useful for quick tests)')
    parser.add_argument('--subsetsize',type=int,default=None,help='Override subsetsize from config (useful for quick tests)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {n.strip() for n in args.runs.split(',')}
    return selectedruns,args.procs,args.timeout,args.iterations,args.subsetsize

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
    Purpose: Load a normalized data split and construct predictor features for symbolic regression. If 'weightsfrom' is set, vertically 
    integrate field variables using kernel weights from a previously trained NN model, averaging across seeds; otherwise read scalar field
    variables directly. A 'timeidx' column is appended to X so that the subsamplers can select whole timesteps or individual samples.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - runconfig (dict): run configuration with keys 'fieldvars', 'localvars', and optionally 'weightsfrom'
    - config (Config): project configuration object
    - time_offset (int): added to each time index so that train and valid indices are globally unique
    Returns:
    - tuple[pd.DataFrame, np.ndarray, xr.DataArray, np.ndarray]: predictor features, tagret values, reference DataArray, and mask of finite samples
    '''
    fieldvars   = runconfig['fieldvars']
    localvars   = runconfig.get('localvars',[])
    weightsfrom = runconfig.get('weightsfrom')
    seeds       = config.nn['seeds']
    filepath    = os.path.join(config.splitsdir,f'norm_{splitname}.h5')
    splitds     = xr.open_dataset(filepath,engine='h5netcdf')
    refda       = splitds[config.targetvar].transpose('time','lat','lon')
    ntime       = splitds.sizes['time']
    nlat        = splitds.sizes.get('lat',1)
    nlon        = splitds.sizes.get('lon',1)
    cols        = {}
    if weightsfrom:
        nsig        = splitds.sizes['sig']
        dsig        = splitds['dsig'].values
        fieldarrays = []
        for var in fieldvars:
            da = splitds[var].transpose('time','lat','lon','sig')
            fieldarrays.append(da.values.reshape(-1,nsig))
        fields3d = np.stack(fieldarrays,axis=1)
        if 'surfmask' in splitds:
            surfmask = splitds['surfmask'].transpose('time','lat','lon','sig').values.reshape(-1,nsig)
        else:
            surfmask = None
        seedfeats = []
        for seed in seeds:
            wpath   = os.path.join(config.weightsdir,f'{weightsfrom}_{seed}_weights.nc')
            wds     = xr.open_dataset(wpath,engine='h5netcdf')
            weights = wds['k'].values
            wds.close()
            seedfeats.append(kernel_integrate(fields3d,weights,dsig,surfmask))
        feats = np.mean(seedfeats,axis=0)
        for i,var in enumerate(fieldvars):
            cols[var] = feats[:,i]
    else:
        for var in fieldvars:
            da      = splitds[var]
            arr     = da.transpose('time','lat','lon').values.ravel() if 'time' in da.dims else np.tile(da.values,(ntime,1,1)).ravel()
            cols[var] = arr
    for var in localvars:
        da      = splitds[var]
        arr     = da.transpose('time','lat','lon').values.ravel() if 'time' in da.dims else np.tile(da.values,(ntime,1,1)).ravel()
        cols[var] = arr
    cols['timeidx'] = np.repeat(np.arange(ntime),nlat*nlon)+time_offset
    X         = pd.DataFrame(cols)
    y         = refda.values.ravel()
    validmask = np.isfinite(X.drop(columns=['timeidx'])).all(axis=1).values & np.isfinite(y)
    splitds.close()
    return X,y,refda,validmask

def subsample_timestep(X,y,subsetsize,seed,loglo=-4,loghi=2):
    '''
    Purpose: Subsample complete timesteps with proportional coverage of the precipitation
        distribution. Timesteps are grouped by their domain-maximum precipitation and drawn
        from each log-decade bin in proportion to its share of the full dataset. ALL valid
        spatial points within each selected timestep are retained.
    Args:
    - X (pd.DataFrame): features including a 'timeidx' column added by load_data()
    - y (np.ndarray): z-scored log1p(tp) target values with shape (nsamples,)
    - subsetsize (int): approximate total samples; actual count is nselectedtimesteps x avgptspertime
    - seed (int): random seed for reproducibility
    - loglo (float): log10 lower bound of wet bins in mm (default -4)
    - loghi (float): log10 upper bound of wet bins in mm (default 2)
    Returns:
    - tuple[pd.DataFrame, np.ndarray]: (xsub, ysub) without the 'timeidx' column
    '''
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        flat = json.load(f)
    tp         = np.expm1(np.asarray(y)*flat['tp_std']+flat['tp_mean'])
    rng        = np.random.default_rng(seed)
    timeidx    = X['timeidx'].values
    uniqtimes,startidx,_ = np.unique(timeidx,return_index=True,return_counts=True)
    sortedtp   = tp[np.argsort(timeidx,kind='stable')]
    tmax       = np.maximum.reduceat(sortedtp,startidx)
    avgpts     = len(timeidx)/len(uniqtimes)
    nbins      = int(loghi-loglo)
    ntotal     = max(1,int(round(subsetsize/avgpts)))
    logbounds  = np.linspace(loglo,loghi,nbins+1)
    logtmax    = np.log10(tmax.clip(min=10**(loglo-1)))
    drymask    = tmax<=10**loglo
    def draw(tidx,n):
        return rng.choice(tidx,n,replace=len(tidx)<n)
    binpools = []
    if drymask.any():
        binpools.append(uniqtimes[drymask])
    for i in range(nbins):
        lo,hi = logbounds[i],logbounds[i+1]
        inbin = uniqtimes[(logtmax>lo)&(logtmax<=hi)]
        if len(inbin)>0:
            binpools.append(inbin)
    totalavail = sum(len(p) for p in binpools)
    selected   = [draw(pool,max(1,round(len(pool)/totalavail*ntotal))) for pool in binpools]
    seltimes   = np.unique(np.concatenate(selected))
    keep       = np.isin(timeidx,seltimes)
    subidx     = np.where(keep)[0]
    rng.shuffle(subidx)
    xsub = X.iloc[subidx].drop(columns=['timeidx']).reset_index(drop=True)
    return xsub,np.asarray(y)[subidx]

def subsample_pointwise(X,y,subsetsize,seed,loglo=-4,loghi=2):
    '''
    Purpose: Pointwise precipitation-stratified subsampling. Each gridpoint sample is
        binned by its own native precipitation value across dry, log-decade wet, and extreme
        bins. An equal number of samples is drawn from each non-empty bin, with replacement
        for rare bins. This ensures heavy-precipitation events are represented at training
        time regardless of how rarely they occur in the full dataset.
    Args:
    - X (pd.DataFrame): features including a 'timeidx' column added by load_data()
    - y (np.ndarray): z-scored log1p(tp) target values with shape (nsamples,)
    - subsetsize (int): approximate total samples; actual = nnonemptybins * (subsetsize // nnonemptybins)
    - seed (int): random seed for reproducibility
    - loglo (float): log10 lower bound of wet bins in mm (default -4)
    - loghi (float): log10 upper bound of wet bins in mm (default 2)
    Returns:
    - tuple[pd.DataFrame, np.ndarray]: (xsub, ysub) without the 'timeidx' column
    '''
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        flat = json.load(f)
    tp        = np.expm1(np.asarray(y)*flat['tp_std']+flat['tp_mean'])
    rng       = np.random.default_rng(seed)
    nbins     = int(loghi-loglo)
    logbounds = np.linspace(loglo,loghi,nbins+1)
    logtpi    = np.log10(tp.clip(min=10**(loglo-1)))
    allidx    = np.arange(len(y))
    binpools  = []
    if (tp<=10**loglo).any():
        binpools.append(allidx[tp<=10**loglo])
    for i in range(nbins):
        lo,hi = logbounds[i],logbounds[i+1]
        inbin = allidx[(logtpi>lo)&(logtpi<=hi)]
        if len(inbin)>0:
            binpools.append(inbin)
    extreme = allidx[tp>10**loghi]
    if len(extreme)>0:
        binpools.append(extreme)
    nnonempty = len(binpools)
    nperbin   = max(1,subsetsize//nnonempty)
    selected  = [rng.choice(pool,nperbin,replace=len(pool)<nperbin) for pool in binpools]
    subidx    = np.concatenate(selected)
    subidx    = subidx[rng.permutation(len(subidx))]
    xsub      = X.iloc[subidx].drop(columns=['timeidx']).reset_index(drop=True)
    return xsub,np.asarray(y)[subidx]

def print_subsample_diagnostics(y_full,y_sub,loglo=-4,loghi=2):
    '''
    Purpose: Log a comparative summary of native-unit precipitation distributions
        between the full training pool and the SR subsample, broken down by bin.
    Args:
    - y_full (np.ndarray): z-scored log1p(tp) for the full pool
    - y_sub (np.ndarray): z-scored log1p(tp) for the subsample
    - loglo (float): log10 lower bound of wet bins in mm (default -4)
    - loghi (float): log10 upper bound of wet bins in mm (default 2)
    '''
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        flat = json.load(f)
    def denorm(y):
        return np.expm1(np.asarray(y)*flat['tp_std']+flat['tp_mean'])
    tpfull    = denorm(y_full)
    tpsub     = denorm(y_sub)
    nbins     = int(loghi-loglo)
    logbounds = np.linspace(loglo,loghi,nbins+1)
    def bin_counts(tp):
        logtpi = np.log10(tp.clip(min=10**(loglo-1)))
        counts = [(tp<=10**loglo).sum()]
        for i in range(nbins):
            lo,hi = logbounds[i],logbounds[i+1]
            counts.append(((logtpi>lo)&(logtpi<=hi)).sum())
        counts.append((tp>10**loghi).sum())
        return counts
    labels = [f'dry (<=1e{int(loglo)} mm)']
    for i in range(nbins):
        labels.append(f'1e{int(logbounds[i])} to 1e{int(logbounds[i+1])} mm')
    labels.append(f'>1e{int(loghi)} mm')
    fc = bin_counts(tpfull)
    sc = bin_counts(tpsub)
    logger.info('   Subsample diagnostics:')
    logger.info(f'     Pool: {len(y_full):,}   Subset: {len(y_sub):,}')
    logger.info(f'     Native tp (mm) pool — mean={tpfull.mean():.4f}  med={np.median(tpfull):.4f}  max={tpfull.max():.4f}')
    logger.info(f'     Native tp (mm) sub  — mean={tpsub.mean():.4f}  med={np.median(tpsub):.4f}  max={tpsub.max():.4f}')
    logger.info(f'     log1p(tp) pool — mean={np.log1p(tpfull).mean():.4f}  std={np.log1p(tpfull).std():.4f}')
    logger.info(f'     log1p(tp) sub  — mean={np.log1p(tpsub).mean():.4f}  std={np.log1p(tpsub).std():.4f}')
    logger.info(f'     {"Bin":<26} {"Pool":>12} {"Subset":>12}')
    for label,fc_,sc_ in zip(labels,fc,sc):
        logger.info(f'     {label:<26} {fc_:>12,} {sc_:>12,}')

def assert_bin_contributions(y_sub,loglo=-4,loghi=2,tolerance=0.5):
    '''
    Purpose: Assert that each non-empty precipitation bin contributes approximately
        equal samples to the subset, validating the pointwise equal-allocation strategy.
    Args:
    - y_sub (np.ndarray): z-scored log1p(tp) for the subsample
    - loglo (float): log10 lower bound of wet bins in mm (default -4)
    - loghi (float): log10 upper bound of wet bins in mm (default 2)
    - tolerance (float): max allowed fractional deviation from mean bin count (default 0.5)
    '''
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        flat = json.load(f)
    tp        = np.expm1(np.asarray(y_sub)*flat['tp_std']+flat['tp_mean'])
    nbins     = int(loghi-loglo)
    logbounds = np.linspace(loglo,loghi,nbins+1)
    logtpi    = np.log10(tp.clip(min=10**(loglo-1)))
    counts    = [(tp<=10**loglo).sum()]
    for i in range(nbins):
        lo,hi = logbounds[i],logbounds[i+1]
        counts.append(((logtpi>lo)&(logtpi<=hi)).sum())
    counts.append((tp>10**loghi).sum())
    nonempty  = [c for c in counts if c>0]
    if len(nonempty)<2:
        return
    expected  = float(np.mean(nonempty))
    for c in nonempty:
        assert abs(c-expected)/max(expected,1)<=tolerance, \
            f'Bin imbalance: {c} samples vs expected ~{expected:.0f} ({100*tolerance:.0f}% tolerance)'

def fit(xsub,ysub,predictors,srconfig,procs,timeout,tmpdir,lossspace='logz'):
    '''
    Purpose: Instantiate and fit a PySRRegressor on the given data subset.
        Operators, complexity penalties, and operator constraints are read from srconfig so they
        can be tuned in configs.json without touching this script. Parallelism is achieved via
        Julia worker processes (procs) rather than threads.
    Args:
    - xsub (pd.DataFrame): predictor features with shape (subsetsize, nfeatures)
    - ysub (np.ndarray): target values with shape (subsetsize,)
    - predictors (list[str]): variable names corresponding to columns of xsub
    - srconfig (dict): SR experiment configuration with keys 'searchparams', 'operators',
        'complexity', 'constraints', 'nestedconstraints', and 'seed'
    - procs (int): number of Julia worker processes
    - timeout (int): search timeout in seconds; acts as a safety net alongside iterations
    - tmpdir (str): temporary directory for Julia equation files
    - lossspace (str): 'logz' (default) computes MSE in z-scored log1p space;
        'native' denormalizes both prediction and target to mm before computing MSE
    Returns:
    - PySRRegressor: fitted model containing the full Pareto frontier of discovered equations
    '''
    sp       = srconfig['searchparams']
    ops      = srconfig['operators']
    compl    = srconfig['complexity']
    constr   = {k:tuple(v) for k,v in srconfig.get('constraints',{}).items()}
    nested   = srconfig.get('nestedconstraints',{})
    popcount = sp.get('populations',3*procs)
    iters    = sp.get('targettotal',sp['iterations']*popcount)//popcount
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        flat = json.load(f)
    zmin = (0.0 - flat['tp_mean']) / flat['tp_std']
    if lossspace == 'native':
        tp_std  = float(flat['tp_std'])
        tp_mean = float(flat['tp_mean'])
        loss = (f'loss(x, y) = (max(expm1(x * {tp_std:.8f} + {tp_mean:.8f}), 0.0)'
                f' - expm1(y * {tp_std:.8f} + {tp_mean:.8f}))^2')
    else:
        loss = f'loss(x, y) = (max(x, {zmin:.8f}) - y)^2'
    model    = PySRRegressor(
        niterations=iters,
        populations=popcount,
        population_size=sp['populationsize'],
        ncycles_per_iteration=sp['cyclesperiteration'],
        weight_optimize=sp['weightoptimize'],
        binary_operators=ops['binary'],
        unary_operators=ops['unary'],
        complexity_of_operators=ops['complexity'],
        complexity_of_variables=[compl['ofvariables'].get(p,1) for p in predictors]
            if isinstance(compl['ofvariables'],dict) else compl['ofvariables'],
        complexity_of_constants=compl['ofconstants'],
        maxsize=sp['maxsize'],
        maxdepth=sp['maxdepth'],
        constraints=constr,
        nested_constraints=nested,
        extra_sympy_mappings={'square':lambda x:x**2},
        loss=loss,
        model_selection='best',
        batching=True,
        batch_size=sp['batchsize'],
        random_state=srconfig['seed'],
        parallelism='multiprocessing',
        procs=procs,
        tempdir=tmpdir,
        temp_equation_file=True,
        delete_tempfiles=True,
        timeout_in_seconds=timeout,
        progress=False)
    model.fit(xsub.values,ysub,variable_names=predictors)
    return model

def save(model,runname,config):
    '''
    Purpose: Save a fitted PySRRegressor and its equation Pareto frontier to disk.
    Args:
    - model (PySRRegressor): fitted symbolic regression model
    - runname (str): run identifier used for output filenames
    - config (Config): project configuration object
    Returns:
    - None
    '''
    outdir  = os.path.join(config.modelsdir,'sr')
    os.makedirs(outdir,exist_ok=True)
    pklpath = os.path.join(outdir,f'{runname}_pareto.pkl')
    csvpath = os.path.join(outdir,f'{runname}_equations.csv')
    with open(pklpath,'wb') as f:
        pickle.dump(model,f)
    model.equations_.to_csv(csvpath,index=False)
    best = select_pareto_elbow(model.equations_)
    logger.info(f'   Best equation: {best["equation"]}')
    logger.info(f'   Complexity: {best["complexity"]}, Loss: {best["loss"]:.6f}')
    logger.info(f'   Saved to {pklpath}')

if __name__=='__main__':
    config = Config()
    sr     = config.sr
    runs   = sr['runs']
    logger.info('Spinning up...')
    selectedruns,procs,timeout,itersoverride,subsetoverride = parse()
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        pklpath = os.path.join(config.modelsdir,'sr',f'{name}_pareto.pkl')
        if os.path.exists(pklpath):
            logger.info(f'Skipping `{name}`, model already exists')
            continue
        logger.info(f'Running `{name}`...')
        fieldvars  = runconfig['fieldvars']
        localvars  = runconfig.get('localvars',[])
        predictors = fieldvars+localvars
        subsetsize = subsetoverride if subsetoverride is not None else sr['subsetsize']
        if itersoverride is not None:
            sr['searchparams']['iterations'] = itersoverride
            sr['searchparams'].pop('targettotal',None)
        sp       = sr['searchparams']
        popcount = sp.get('populations',3*procs)
        iterseff = sp.get('targettotal',sp['iterations']*popcount)//popcount
        logger.info(f'   Loading normalized training and validation splits...')
        xtrain,ytrain,refdatrain,vmtrain = load_data('train',runconfig,config,time_offset=0)
        xvalid,yvalid,_,vmvalid = load_data('valid',runconfig,config,time_offset=int(refdatrain.sizes['time']))
        xfit = pd.concat([xtrain[vmtrain],xvalid[vmvalid]]).reset_index(drop=True)
        yfit = np.concatenate([ytrain[vmtrain],yvalid[vmvalid]])
        del xtrain,xvalid,ytrain,yvalid,refdatrain
        strategy = sr.get('samplingStrategy','pointwise')
        logger.info(f'   Subsampling ~{subsetsize} samples using {strategy!r} strategy...')
        if strategy=='pointwise':
            xsub,ysub = subsample_pointwise(xfit,yfit,subsetsize,sr['seed'])
            assert_bin_contributions(ysub)
        else:
            xsub,ysub = subsample_timestep(xfit,yfit,subsetsize,sr['seed'])
        print_subsample_diagnostics(yfit,ysub)
        del xfit,yfit
        logger.info(f'   Starting PySR search ({iterseff} iters × {popcount} populations, {procs} workers, {timeout}s timeout)...')
        tmpdir = tempfile.mkdtemp(prefix='pysr_')
        try:
            model = fit(xsub,ysub,predictors,sr,procs,timeout,tmpdir,
                        lossspace=runconfig.get('lossspace','logz'))
        finally:
            shutil.rmtree(tmpdir,ignore_errors=True)
        save(model,name,config)
        del model