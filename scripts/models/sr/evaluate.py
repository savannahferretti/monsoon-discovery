#!/usr/bin/env python

import os
import json
import logging
import argparse
import pickle
import numpy as np
import xarray as xr
from scripts.utils import Config
from scripts.data.classes import PredictionWriter
from scripts.models.sr.train import load_data

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse():
    '''
    Purpose: Parse command-line arguments for running the evaluation script.
    Returns:
    - tuple[set[str] | None, str]: selected run names (or None for all), and split name to evaluate
    '''
    parser = argparse.ArgumentParser(description='Evaluate PySR symbolic regression models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated run names to evaluate, or `all`')
    parser.add_argument('--split',type=str,default='test',help='Split to evaluate: train|valid|test (default: test)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {n.strip() for n in args.runs.split(',')}
    return selectedruns,args.split

def load(name,seed,modelsdir):
    '''
    Purpose: Load a saved PySRRegressor from disk.
    Args:
    - name (str): run identifier matching the saved filename
    - seed (int): training seed matching the saved filename
    - modelsdir (str): base models directory containing the sr/ subdirectory
    Returns:
    - PySRRegressor | None: loaded model, or None if the file is not found
    '''
    filepath = os.path.join(modelsdir,'sr',f'{name}_{seed}_pareto.pkl')
    if not os.path.exists(filepath):
        logger.error(f'   Model not found: {filepath}')
        return None
    with open(filepath,'rb') as f:
        return pickle.load(f)

def predict_pareto(model,x,zfloor,writer,validmask,refda):
    '''
    Purpose: Evaluate every equation on the Pareto frontier and return predictions keyed by complexity.
    Args:
    - model (PySRRegressor): fitted model whose equations_ DataFrame holds the frontier
    - x (np.ndarray): feature matrix with shape (nvalidsamples, nfeatures)
    - zfloor (float): z-scored floor corresponding to 0 mm precipitation
    - writer (PredictionWriter): used for unflatten and denormalization stats
    - validmask (np.ndarray): boolean mask selecting valid grid points from the full flat array
    - refda (xr.DataArray): reference DataArray supplying (time, lat, lon) coordinates
    Returns:
    - dict[int, np.ndarray]: mapping from complexity to gridded array with shape (time, lat, lon)
    '''
    preds = {}
    for i in range(len(model.equations_)):
        row     = model.equations_.iloc[i]
        flat    = zfloor+np.maximum(model.predict(x,index=i),0.0)
        gridded = np.maximum(np.expm1(writer.unflatten(flat,validmask,refda)*writer.std+writer.mean),0.0).astype(np.float32)
        preds[int(row['complexity'])] = gridded
    return preds

def assemble_predictions(seedpreds,seeds,writer,refda):
    '''
    Purpose: Assemble per-seed Pareto frontier predictions into a single xr.Dataset.
        Complexities missing for a given seed are filled with NaN.
    Args:
    - seedpreds (list[dict[int, np.ndarray]]): one dict per seed, mapping complexity → gridded array
        with shape (time, lat, lon)
    - seeds (list[int]): seed values corresponding to entries in seedpreds
    - writer (PredictionWriter): supplies targetvar, longname, and units metadata
    - refda (xr.DataArray): reference DataArray with (time, lat, lon) coordinates
    Returns:
    - xr.Dataset: predictions with dims (time, lat, lon, seed, complexity)
    '''
    allcomplexities = sorted(set().union(*[set(p.keys()) for p in seedpreds]))
    nanarray        = np.full(refda.shape,np.nan,dtype=np.float32)
    stacked         = np.stack(
        [np.stack([seeddict.get(c,nanarray) for c in allcomplexities],axis=-1) for seeddict in seedpreds],
        axis=-2)
    coords = {dim:refda.coords[dim] for dim in refda.dims}
    coords['seed']       = xr.DataArray(seeds,dims=['seed'],attrs=dict(long_name='Training seed'))
    coords['complexity'] = xr.DataArray(allcomplexities,dims=['complexity'],attrs=dict(long_name='Equation complexity'))
    da = xr.DataArray(stacked,dims=('time','lat','lon','seed','complexity'),coords=coords)
    da.attrs = dict(long_name=writer.longname,units=writer.units)
    return da.to_dataset(name=writer.targetvar)

if __name__=='__main__':
    config    = Config()
    sr        = config.sr
    runs      = sr['runs']
    seeds     = sr['seeds']
    targetvar = config.targetvar
    logger.info('Spinning up...')
    selectedruns,split = parse()
    writer    = PredictionWriter(config.splitsdir,targetvar=targetvar)
    statsfile = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        stats = json.load(f)
    zfloor    = (0.0-stats[f'{targetvar}_mean'])/stats[f'{targetvar}_std']
    cachedkey  = None
    cacheddata = None
    for name,runconfig in runs.items():
        if selectedruns is not None and name not in selectedruns:
            continue
        predpath = os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')
        if os.path.exists(predpath):
            logger.info(f'Skipping `{name}`, predictions already exist')
            continue
        fieldvars   = runconfig['fieldvars']
        localvars   = runconfig.get('localvars',[])
        weightsfrom = runconfig.get('weightsfrom')
        cachekey    = (tuple(fieldvars),tuple(localvars),weightsfrom,split)
        if cachekey!=cachedkey:
            logger.info(f'   Loading normalized {split} split for fieldvars={fieldvars}, localvars={localvars}...')
            x,y,refda,validmask = load_data(split,runconfig,config)
            cachedkey  = cachekey
            cacheddata = (x,y,refda,validmask)
        else:
            x,y,refda,validmask = cacheddata
        predictors = fieldvars+localvars
        xvalid     = x[validmask][predictors].reset_index(drop=True)
        seedpreds  = []
        for seedidx,seed in enumerate(seeds):
            model = load(name,seed,config.modelsdir)
            if model is None:
                break
            logger.info(f'   Evaluating `{name}` seed {seedidx+1}/{len(seeds)} ({seed}) ({validmask.sum()} valid samples, {len(model.equations_)} Pareto equations)...')
            seedpreds.append(predict_pareto(model,xvalid.values,zfloor,writer,validmask,refda))
            del model
        else:
            logger.info(f'   Saving predictions for `{name}`...')
            predds = assemble_predictions(seedpreds,seeds,writer,refda)
            writer.save(predds,name,'predictions',split,config.predsdir)
            del predds
        del xvalid,seedpreds
