#!/usr/bin/env python

import os
import json
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from scripts.utils import Config
from scripts.data.classes import PredictionWriter
from scripts.models.sr.train import kernel_integrate,load_data

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

def load(name,modelsdir):
    '''
    Purpose: Load a saved PySRRegressor from disk.
    Args:
    - name (str): run identifier matching the saved filename
    - modelsdir (str): base models directory containing the sr/ subdirectory
    Returns:
    - PySRRegressor | None: loaded model, or None if the file is not found
    '''
    filepath = os.path.join(modelsdir,'sr',f'{name}_pareto.pkl')
    if not os.path.exists(filepath):
        logger.error(f'   Model not found: {filepath}')
        return None
    with open(filepath,'rb') as f:
        return pickle.load(f)

def predict_pareto(model,X,zmin,writer,validmask,refda):
    '''
    Purpose: Evaluate every equation on the Pareto frontier and return predictions
        with complexity as a dimension.
    Args:
    - model (PySRRegressor): fitted model whose equations_ DataFrame holds the frontier
    - X (np.ndarray): feature matrix with shape (nvalidsamples, nfeatures)
    - zmin (float): z-scored floor corresponding to 0 mm precipitation
    - writer (PredictionWriter): used for unflatten and denormalization stats
    - validmask (np.ndarray): boolean mask selecting valid grid points from the full flat array
    - refda (xr.DataArray): reference DataArray supplying (time, lat, lon) coordinates
    Returns:
    - xr.Dataset: predictions in native units with dims (time, lat, lon, complexity)
    '''
    arrs,complexities = [],[]
    for i in range(len(model.equations_)):
        row     = model.equations_.iloc[i]
        flat    = np.maximum(model.predict(X,index=i),zmin)
        gridded = np.maximum(np.expm1(writer.unflatten(flat,validmask,refda)*writer.std+writer.mean),0.0).astype(np.float32)
        arrs.append(gridded)
        complexities.append(int(row['complexity']))
    predstack = np.stack(arrs,axis=-1)
    coords    = {dim:refda.coords[dim] for dim in refda.dims}
    coords['complexity'] = xr.DataArray(complexities,dims=['complexity'],attrs=dict(long_name='Equation complexity'))
    da = xr.DataArray(predstack,dims=('time','lat','lon','complexity'),coords=coords)
    da.attrs = dict(long_name=writer.longname,units=writer.units)
    return da.to_dataset(name=writer.targetvar)

if __name__=='__main__':
    config    = Config()
    sr        = config.sr
    runs      = sr['runs']
    targetvar = config.targetvar
    logger.info('Spinning up...')
    selectedruns,split = parse()
    writer     = PredictionWriter(config.splitsdir,targetvar=targetvar)
    statsfile  = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..','data','splits','stats.json'))
    with open(statsfile,'r',encoding='utf-8') as f:
        flat = json.load(f)
    zmin = (0.0 - flat[f'{targetvar}_mean']) / flat[f'{targetvar}_std']
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
            X,y,refda,validmask = load_data(split,runconfig,config)
            cachedkey  = cachekey
            cacheddata = (X,y,refda,validmask)
        else:
            X,y,refda,validmask = cacheddata
        model = load(name,config.modelsdir)
        if model is None:
            continue
        featurecols = fieldvars+localvars
        xvalid      = X[validmask][featurecols].reset_index(drop=True)
        logger.info(f'   Evaluating `{name}` ({validmask.sum()} valid samples, {len(model.equations_)} Pareto equations)...')
        predds = predict_pareto(model,xvalid.values,zmin,writer,validmask,refda)
        logger.info(f'   Saving predictions for `{name}`...')
        writer.save(predds,name,'predictions',split,config.predsdir)
        del model,xvalid,predds
