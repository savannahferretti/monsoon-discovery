#!/usr/bin/env python

import os
os.environ.setdefault('JULIA_NUM_THREADS', '1')

import gc
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
from pysr import PySRRegressor,jl
from scripts.utils import Config
from scripts.data.classes import PredictionWriter

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse():
    '''
    Purpose: Parse command-line arguments for running the training script.
    Returns:
    - tuple[set[str] | None, int, int, int | None, int | None]: selected run names (or None for
        all), number of Julia worker processes, search timeout in seconds, and optional overrides
        for niterations and subsetsize (None means use the value from configs.json)
    '''
    parser = argparse.ArgumentParser(description='Train PySR symbolic regression models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated run names to train, or `all`')
    parser.add_argument('--procs',type=int,default=127,help='Number of Julia worker processes (default: 127)')
    parser.add_argument('--timeout',type=int,default=19800,help='PySR search timeout in seconds (default: 19800)')
    parser.add_argument('--niterations',type=int,default=None,help='Override niterations from config (useful for quick tests)')
    parser.add_argument('--subsetsize',type=int,default=None,help='Override subsetsize from config (useful for quick tests)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {n.strip() for n in args.runs.split(',')}
    return selectedruns,args.procs,args.timeout,args.niterations,args.subsetsize

def kernel_integrate(fields,weights,dsig,mask=None):
    '''
    Purpose: Integrate vertical field profiles using kernel weights and sigma-level thicknesses.
    Args:
    - fields (np.ndarray): profile data with shape (nsamp, nfieldvars, nsig)
    - weights (np.ndarray): kernel weights with shape (nfieldvars, nsig)
    - dsig (np.ndarray): sigma thickness weights with shape (nsig,)
    - mask (np.ndarray | None): surface mask with shape (nsamp, nsig), or None to skip masking
    Returns:
    - np.ndarray: integrated features with shape (nsamp, nfieldvars)
    '''
    weighted = fields * weights[None,:,:] * dsig[None,None,:]
    if mask is not None:
        weighted = weighted * mask[:,None,:]
    return weighted.sum(axis=2)

def load_data(splitname,runconfig,config,time_offset=0):
    '''
    Purpose: Load a normalized data split and construct predictor features for symbolic regression.
        If `weightsfrom` is set in runconfig, integrates vertical field profiles using kernel weights
        from a previously trained NN model, averaging across seeds. Otherwise reads scalar field
        variables directly. All field profile variables are on sigma levels with dimension `sig`.
        A '__time_idx__' column is added to X so that subsample() can select whole timesteps.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - runconfig (dict): run configuration with keys 'fieldvars', 'localvars', and optionally 'weightsfrom'
    - config (Config): project configuration object
    - time_offset (int): added to each time index so that train and valid time indices are globally unique
    Returns:
    - tuple[pd.DataFrame, np.ndarray, xr.DataArray, np.ndarray]:
        (X, y, refda, validmask) where:
        - X: predictor features with shape (ntotal, nfeatures+1), NaN where invalid; last column is '__time_idx__'
        - y: target values with shape (ntotal,)
        - refda: reference DataArray with (time, lat, lon) coordinates
        - validmask: boolean array with shape (ntotal,) indicating finite samples
    '''
    fieldvars   = runconfig['fieldvars']
    localvars   = runconfig.get('localvars',[])
    weightsfrom = runconfig.get('weightsfrom')
    seeds       = config.nn['seeds']
    filepath  = os.path.join(config.splitsdir,f'norm_{splitname}.h5')
    splitds   = xr.open_dataset(filepath,engine='h5netcdf')
    refda     = splitds[config.targetvar].transpose('time','lat','lon')
    ntime     = splitds.sizes['time']
    nlat      = splitds.sizes.get('lat',1)
    nlon      = splitds.sizes.get('lon',1)
    columns   = {}
    if weightsfrom:
        nsig  = splitds.sizes['sig']
        dsig  = splitds['dsig'].values
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
            weights = wds['k'].isel(seed=0).values
            wds.close()
            seedfeats.append(kernel_integrate(fields3d,weights,dsig,surfmask))
        feats = np.mean(seedfeats,axis=0)
        for i,var in enumerate(fieldvars):
            columns[var] = feats[:,i]
    else:
        for var in fieldvars:
            da = splitds[var]
            if 'time' in da.dims:
                arr = da.transpose('time','lat','lon').values.ravel()
            else:
                arr = np.tile(da.values,(ntime,1,1)).ravel()
            columns[var] = arr
    for var in localvars:
        da = splitds[var]
        if 'time' in da.dims:
            arr = da.transpose('time','lat','lon').values.ravel()
        else:
            arr = np.tile(da.values,(ntime,1,1)).ravel()
        columns[var] = arr
    columns['__time_idx__'] = np.repeat(np.arange(ntime),nlat*nlon)+time_offset
    X         = pd.DataFrame(columns)
    y         = refda.values.ravel()
    validmask = np.isfinite(X.drop(columns=['__time_idx__'])).all(axis=1).values & np.isfinite(y)
    splitds.close()
    return X,y,refda,validmask

def subsample(X,y,subsetsize,seed,loglo=-4,loghi=2):
    '''
    Purpose: Subsample complete timesteps with log-uniform coverage of the precipitation
        distribution. Timesteps are grouped by their domain-maximum precipitation in raw mm
        (recovered by inverting the log1p normalization) and divided into one-decade-wide
        log10 bins from loglo to loghi, plus one dry bin for timesteps below 10^loglo mm.
        An equal number of timesteps is drawn from each non-empty bin (with replacement when
        a bin has fewer timesteps than needed), and ALL valid spatial points within each
        selected timestep are retained. This avoids geographic bias and ensures the model
        sees complete spatial snapshots weighted equally across precipitation regimes.
    Args:
    - X (pd.DataFrame): features including a '__time_idx__' column added by load_data()
    - y (np.ndarray): normalized log1p(tp) target values with shape (nsamples,)
    - subsetsize (int): approximate total samples; actual count is n_selected_timesteps x avg_pts_per_time
    - seed (int): random seed for reproducibility
    - loglo (float): log10 lower bound of wet bins in mm (default -4 → 0.0001 mm)
    - loghi (float): log10 upper bound of wet bins in mm (default 2 → 100 mm)
    Returns:
    - tuple[pd.DataFrame, np.ndarray]: (Xsub, ysub) without the '__time_idx__' column
    '''
    statsfile = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        flat    = json.load(f)
    tp      = np.expm1(np.asarray(y)*flat['tp_std']+flat['tp_mean'])
    rng         = np.random.default_rng(seed)
    time_idx    = X['__time_idx__'].values
    uniq_times,start_idx,_ = np.unique(time_idx,return_index=True,return_counts=True)
    sorted_tp   = tp[np.argsort(time_idx,kind='stable')]
    tmax        = np.maximum.reduceat(sorted_tp,start_idx)
    avg_pts     = len(time_idx)/len(uniq_times)
    nbins       = int(loghi-loglo)
    n_t_total   = max(nbins+1,int(round(subsetsize/avg_pts)))
    n_per_bin   = max(1,n_t_total//(nbins+1))
    logbounds   = np.linspace(loglo,loghi,nbins+1)
    def draw(tidx,n):
        return rng.choice(tidx,n,replace=len(tidx)<n)
    selected = []
    dry_mask = tmax<=10**loglo
    if dry_mask.any():
        selected.append(draw(uniq_times[dry_mask],n_per_bin))
    log_tmax = np.log10(tmax.clip(min=10**(loglo-1)))
    for i in range(nbins):
        lo,hi  = logbounds[i],logbounds[i+1]
        in_bin = uniq_times[(log_tmax>lo)&(log_tmax<=hi)]
        if len(in_bin)>0:
            selected.append(draw(in_bin,n_per_bin))
    sel_times = np.unique(np.concatenate(selected))
    keep      = np.isin(time_idx,sel_times)
    subidx    = np.where(keep)[0]
    rng.shuffle(subidx)
    Xsub = X.iloc[subidx].drop(columns=['__time_idx__']).reset_index(drop=True)
    return Xsub,y[subidx]

def fit(Xsub,ysub,predictors,srconfig,procs,timeout,tmpdir,ymin=None):
    '''
    Purpose: Instantiate and fit a PySRRegressor on the given data subset.
        Operators, complexity penalties, and operator constraints are read from srconfig so they
        can be tuned in configs.json without touching this script. When physical_constraints is
        enabled, a clipped MSE loss enforces the lower bound ymin (the normalized value of tp=0)
        so that equations producing negative precipitation are penalized during the search.
        Parallelism is achieved via Julia worker processes (`procs`) rather than threads.
    Args:
    - Xsub (pd.DataFrame): predictor features with shape (subsetsize, nfeatures)
    - ysub (np.ndarray): target values with shape (subsetsize,)
    - predictors (list[str]): variable names corresponding to columns of Xsub
    - srconfig (dict): SR experiment configuration with keys 'searchparams', 'operators',
        'complexity', 'constraints', 'nested_constraints', 'seed', and 'physical_constraints';
        searchparams must include 'ncycles_per_iteration' and 'weight_optimize'
    - procs (int): number of Julia worker processes; populations is set to 3*procs
    - timeout (int): search timeout in seconds; acts as a safety net alongside niterations
    - tmpdir (str): temporary directory for Julia equation files
    - ymin (float | None): normalized lower bound on predictions (tp=0 in native units);
        only used when physical_constraints is True
    Returns:
    - PySRRegressor: fitted model containing the full Pareto frontier of discovered equations
    '''
    sp     = srconfig['searchparams']
    ops    = srconfig['operators']
    compl  = srconfig['complexity']
    constr = {k:tuple(v) for k,v in srconfig.get('constraints',{}).items()}
    nested = srconfig.get('nested_constraints',{})
    if srconfig.get('physical_constraints',False) and ymin is not None:
        loss = f'loss(x, y) = (max(x, {ymin:.6f}) - y)^2'
    else:
        loss = 'loss(x, y) = (x - y)^2'
    model = PySRRegressor(
        niterations=sp['niterations'],
        populations=3*procs,
        population_size=sp['population_size'],
        ncycles_per_iteration=sp['ncycles_per_iteration'],
        weight_optimize=sp['weight_optimize'],
        binary_operators=ops['binary'],
        unary_operators=ops['unary'],
        complexity_of_operators=ops['complexity'],
        complexity_of_variables=compl['of_variables'],
        complexity_of_constants=compl['of_constants'],
        maxsize=sp['maxsize'],
        maxdepth=sp['maxdepth'],
        constraints=constr,
        nested_constraints=nested,
        extra_sympy_mappings={'pow_abs':lambda x,y:x**y},
        loss=loss,
        model_selection='best',
        turbo=False,
        batching=True,
        batch_size=sp['batch_size'],
        random_state=srconfig['seed'],
        parallelism='multiprocessing',
        procs=procs,
        tempdir=tmpdir,
        temp_equation_file=True,
        delete_tempfiles=True,
        timeout_in_seconds=timeout,
        progress=False)
    model.fit(Xsub.values,ysub,variable_names=predictors)
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
    best = model.get_best()
    logger.info(f'   Best equation: {best["equation"]}')
    logger.info(f'   Complexity: {best["complexity"]}, Loss: {best["loss"]:.6f}')
    logger.info(f'   Saved to {pklpath}')

if __name__=='__main__':
    jl.seval('pow_abs(x, y) = abs(x)^y')
    jl.seval('log_abs(x) = log(abs(x))')
    jl.seval('sqrt_abs(x) = sqrt(abs(x))')
    jl.seval('sigmoid(x) = one(x) / (one(x) + exp(-x))')
    config = Config()
    sr     = config.sr
    runs   = sr['runs']
    logger.info('Spinning up...')
    selectedruns,procs,timeout,niterations_override,subsetsize_override = parse()
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
        predictors = fieldvars + localvars
        subsetsize = subsetsize_override if subsetsize_override is not None else sr['subsetsize']
        if niterations_override is not None:
            sr['searchparams']['niterations'] = niterations_override
        logger.info(f'   Loading normalized training and validation splits...')
        Xtrain,ytrain,refda_train,vmtrain = load_data('train',runconfig,config,time_offset=0)
        Xvalid,yvalid,_,vmvalid = load_data('valid',runconfig,config,time_offset=int(refda_train.sizes['time']))
        Xfit = pd.concat([Xtrain[vmtrain],Xvalid[vmvalid]]).reset_index(drop=True)
        yfit = np.concatenate([ytrain[vmtrain],yvalid[vmvalid]])
        del Xtrain,Xvalid,ytrain,yvalid,refda_train
        gc.collect()
        logger.info(f'   Subsampling ~{subsetsize} samples across log-uniform precipitation bins...')
        ymin = float(yfit.min()) if sr.get('physical_constraints',False) else None
        Xsub,ysub = subsample(Xfit,yfit,subsetsize,sr['seed'])
        del Xfit,yfit
        gc.collect()
        if ymin is not None:
            logger.info(f'   Physical constraint: predictions clipped to 0 in transformed/normalized space...')
        logger.info(f'   Starting PySR search ({sr["searchparams"]["niterations"]} iterations, {procs} processes, {timeout}s timeout)...')
        tmpdir = tempfile.mkdtemp(prefix='pysr_')
        try:
            model = fit(Xsub,ysub,predictors,sr,procs,timeout,tmpdir,ymin=ymin)
        finally:
            shutil.rmtree(tmpdir,ignore_errors=True)
        save(model,name,config)
        del model
        gc.collect()
