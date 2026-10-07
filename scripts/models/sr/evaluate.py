#!/usr/bin/env python

import os
import logging
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scripts.utils import Config
from scripts.data.classes import PredictionWriter
from scripts.models.sr.train import load_features
from scripts.models.sr.equations import evaluate,raw_to_precip

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse():
    '''
    Purpose: Parse command-line arguments.
    Returns:
    - tuple[set[str] | None, str]: run names to evaluate (None for all) and split name
    '''
    parser = argparse.ArgumentParser(description='Evaluate PySR symbolic regression models.')
    parser.add_argument('--runs',type=str,default='all',help='Comma-separated run names to evaluate, or `all`')
    parser.add_argument('--split',type=str,default='test',help='Split to evaluate: train|valid|test (default: test)')
    args = parser.parse_args()
    selectedruns = None if args.runs=='all' else {n.strip() for n in args.runs.split(',')}
    return selectedruns,args.split

def predict_frontier(equations,x,predictornames,writer,validmask,refda):
    '''
    Purpose: Predict precipitation with every equation on one seed's Pareto frontier.
    Args:
    - equations (pd.DataFrame): Pareto frontier with 'complexity' and 'equation' columns
    - x (pd.DataFrame): standardized predictors of the valid samples
    - predictornames (list[str]): predictor names
    - writer (PredictionWriter): prediction writer (target statistics and gridding)
    - validmask (np.ndarray): boolean mask of valid samples over the full grid
    - refda (xr.DataArray): reference DataArray with (time, lat, lon) coordinates
    Returns:
    - dict[int, np.ndarray]: complexity → gridded precipitation (mm)
    '''
    preds = {}
    for _,row in equations.iterrows():
        eqstr = str(row['equation'])
        columns = {p:x[p].values for p in predictornames}
        raw  = evaluate(eqstr,columns,{})
        preds[int(row['complexity'])] = writer.unflatten(raw_to_precip(raw,writer.std),validmask,refda)
    return preds

def assemble_predictions(seedpreds,seeds,writer,refda):
    '''
    Purpose: Combine the per-seed frontier predictions into one xr.Dataset, with NaN where a seed has no equation at
    a complexity.
    Args:
    - seedpreds (list[dict[int, np.ndarray]]): one dict per seed from predict_frontier()
    - seeds (list[int]): seed of each entry in seedpreds
    - writer (PredictionWriter): prediction writer (target name and metadata)
    - refda (xr.DataArray): reference DataArray with (time, lat, lon) coordinates
    Returns:
    - xr.Dataset: predictions with dims (time, lat, lon, seed, complexity)
    '''
    allcomplexities = sorted(set().union(*[set(p.keys()) for p in seedpreds]))
    nanarray        = np.full(refda.shape,np.nan,dtype=np.float64)
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
        cachekey    = (tuple(fieldvars),tuple(localvars),weightsfrom,runconfig.get('residualfrom'),split)
        if cachekey!=cachedkey:
            logger.info(f'   Loading normalized {split} split for fieldvars={fieldvars}, localvars={localvars}...')
            x,y,refda,validmask = load_features(split,runconfig,config)
            cachedkey  = cachekey
            cacheddata = (x,y,refda,validmask)
        else:
            x,y,refda,validmask = cacheddata
        predictors     = [c for c in x.columns if c != 'timeidx']
        xvalid         = x[validmask][predictors].reset_index(drop=True)
        seedpreds  = []
        for seedidx,seed in enumerate(seeds):
            csvpath = os.path.join(config.modelsdir,'sr',f'{name}_{seed}_equations.csv')
            if not os.path.exists(csvpath):
                logger.error(f'   CSV not found: {csvpath}')
                break
            equations = pd.read_csv(csvpath)
            logger.info(f'   Evaluating `{name}` seed {seedidx+1}/{len(seeds)} ({seed}) ({validmask.sum()} valid samples, {len(equations)} Pareto equations)...')
            seedpreds.append(predict_frontier(equations,xvalid,predictors,writer,validmask,refda))
        else:
            logger.info(f'   Saving predictions for `{name}`...')
            predds = assemble_predictions(seedpreds,seeds,writer,refda)
            writer.save(predds,name,'predictions',split,config.predsdir)
            del predds
        del xvalid,seedpreds
