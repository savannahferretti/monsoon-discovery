#!/usr/bin/env python

import os
os.environ.setdefault('JULIA_NUM_THREADS', '1')

import gc
import shutil
import logging
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
    - tuple[set[str] | None, int, int]: selected run names (or None for all), number of Julia
        worker processes, and search timeout in seconds
    '''
    parser = argparse.ArgumentParser(description='Train PySR symbolic regression models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated run names to train, or `all`')
    parser.add_argument('--procs',type=int,default=63,help='Number of Julia worker processes (default: 63)')
    parser.add_argument('--timeout',type=int,default=19800,help='PySR search timeout in seconds (default: 19800)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {n.strip() for n in args.runs.split(',')}
    return selectedruns,args.procs,args.timeout

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

def load_data(splitname,runconfig,config):
    '''
    Purpose: Load a normalized data split and construct predictor features for symbolic regression.
        If `featsfrom` is set in runconfig, integrates vertical field profiles using kernel weights
        from a previously trained NN model, averaging across seeds. Otherwise reads scalar field
        variables directly. All field profile variables are on sigma levels with dimension `sig`.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - runconfig (dict): run configuration with keys 'fieldvars', 'localvars', and optionally 'featsfrom'
    - config (Config): project configuration object
    Returns:
    - tuple[pd.DataFrame, np.ndarray, xr.DataArray, np.ndarray]:
        (X, y, refda, validmask) where:
        - X: predictor features with shape (ntotal, nfeatures), NaN where invalid
        - y: target values with shape (ntotal,)
        - refda: reference DataArray with (time, lat, lon) coordinates
        - validmask: boolean array with shape (ntotal,) indicating finite samples
    '''
    fieldvars = runconfig['fieldvars']
    localvars = runconfig.get('localvars',[])
    featsfrom = runconfig.get('featsfrom')
    seeds     = config.nn['seeds']
    filepath  = os.path.join(config.splitsdir,f'norm_{splitname}.h5')
    splitds   = xr.open_dataset(filepath,engine='h5netcdf')
    refda     = splitds[config.targetvar].transpose('time','lat','lon')
    ntime     = splitds.sizes['time']
    columns   = {}
    if featsfrom:
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
            wpath   = os.path.join(config.weightsdir,f'{featsfrom}_{seed}_weights.nc')
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
    X         = pd.DataFrame(columns)
    y         = refda.values.ravel()
    validmask = np.isfinite(X).all(axis=1).values & np.isfinite(y)
    splitds.close()
    return X,y,refda,validmask

def subsample(X,y,subsetsize,seed):
    '''
    Purpose: Stratified subsample by target value to ensure uniform coverage of all precipitation regimes.
        Sorts samples by target, divides into `subsetsize` equal-width bins, and draws one random
        sample from each bin so that the subset spans the full distribution from driest to wettest.
    Args:
    - X (pd.DataFrame): predictor features with shape (nsamples, nfeatures)
    - y (np.ndarray): target values with shape (nsamples,)
    - subsetsize (int): number of samples to draw, one per sorted bin
    - seed (int): random seed for reproducibility
    Returns:
    - tuple[pd.DataFrame, np.ndarray]: (Xsub, ysub) each with shape (subsetsize, nfeatures) and (subsetsize,)
    '''
    rng       = np.random.default_rng(seed)
    sortedidx = np.argsort(y)
    bins      = np.array_split(sortedidx,subsetsize)
    subidx    = np.array([rng.choice(b) for b in bins if len(b)>0])
    return X.iloc[subidx].reset_index(drop=True),y[subidx]

def fit(Xsub,ysub,predictors,srconfig,procs,timeout,tmpdir):
    '''
    Purpose: Instantiate and fit a PySRRegressor on the given data subset.
        Uses a large niterations value so that the search runs until `timeout_in_seconds` fires,
        which is the recommended pattern for time-limited HPC runs. Parallelism is achieved via
        Julia worker processes (`procs`) rather than threads to avoid contention.
    Args:
    - Xsub (pd.DataFrame): predictor features with shape (subsetsize, nfeatures)
    - ysub (np.ndarray): target values with shape (subsetsize,)
    - predictors (list[str]): variable names corresponding to columns of Xsub
    - srconfig (dict): SR experiment configuration with keys 'searchparams' and 'seed'
    - procs (int): number of Julia worker processes
    - timeout (int): search timeout in seconds
    - tmpdir (str): temporary directory for Julia equation files
    Returns:
    - PySRRegressor: fitted model containing the full Pareto frontier of discovered equations
    '''
    sp    = srconfig['searchparams']
    model = PySRRegressor(
        niterations=sp['niterations'],
        populations=sp['populations'],
        population_size=sp['population_size'],
        binary_operators=['+','-','*','/','safe_pow'],
        unary_operators=['exp','log'],
        complexity_of_operators={'+':1,'-':1,'*':1,'/':3,'safe_pow':3,'exp':4,'log':4},
        complexity_of_variables=2,
        complexity_of_constants=1,
        maxsize=sp['maxsize'],
        maxdepth=sp['maxdepth'],
        constraints={'safe_pow':(-1,1)},
        nested_constraints={
            'exp':{'exp':0,'log':0,'safe_pow':0},
            'safe_pow':{'safe_pow':0},
            'log':{'log':0,'exp':0}},
        extra_sympy_mappings={'safe_pow':lambda x,y:x**y},
        loss='loss(x, y) = (x - y)^2',
        model_selection='best',
        turbo=True,
        batching=True,
        batch_size=sp['batch_size'],
        random_state=srconfig['seed'],
        deterministic=True,
        multithreading=False,
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
    jl.seval('safe_pow(x, y) = abs(x)^y')
    config = Config()
    sr     = config.sr
    runs   = sr['runs']
    logger.info('Spinning up...')
    selectedruns,procs,timeout = parse()
    logger.info(f'procs={procs}, timeout={timeout}s')
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
        logger.info(f'   Loading normalized splits...')
        Xtrain,ytrain,_,vmtrain = load_data('train',runconfig,config)
        Xvalid,yvalid,_,vmvalid = load_data('valid',runconfig,config)
        Xfit = pd.concat([Xtrain[vmtrain],Xvalid[vmvalid]]).reset_index(drop=True)
        yfit = np.concatenate([ytrain[vmtrain],yvalid[vmvalid]])
        del Xtrain,Xvalid,ytrain,yvalid
        gc.collect()
        logger.info(f'   {len(yfit)} valid samples from train+valid splits')
        logger.info(f'   Subsampling {sr["subsetsize"]} stratified samples...')
        Xsub,ysub = subsample(Xfit,yfit,sr['subsetsize'],sr['seed'])
        del Xfit,yfit
        gc.collect()
        logger.info(f'   Starting PySR search (procs={procs}, timeout={timeout}s)...')
        tmpdir = tempfile.mkdtemp(prefix='pysr_')
        try:
            model = fit(Xsub,ysub,predictors,sr,procs,timeout,tmpdir)
        finally:
            shutil.rmtree(tmpdir,ignore_errors=True)
        save(model,name,config)
        del model
        gc.collect()
