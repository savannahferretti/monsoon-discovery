#!/usr/bin/env python

import os
import json
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from timingutils import TimingConfig,parse_names,load_stats,restrict_kernel_seeds,calc_r2,load_dataset
from data import load_split,load_features,load_kernels
from equations import evaluate,raw_to_precip
from scripts.models.sr.train import select_pareto_elbow

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

REGIONS = ('all','land','ocean')

def load_truth(config,split):
    '''
    Purpose: Load true precipitation and region masks for a split.
    Args:
    - config (TimingConfig): configuration object
    - split (str): split name
    Returns:
    - tuple[xr.DataArray, dict[str, np.ndarray]]: truth (time, lat, lon) and region masks of the same shape
    '''
    landfrac = config.timing['landfrac']
    ds    = load_split(config,split)
    truth = ds['tp'].transpose('time','lat','lon')
    lf    = np.broadcast_to(ds['lf'].transpose('lat','lon').values[None,:,:],truth.shape)
    masks = {'all':np.ones(truth.shape,dtype=bool),'land':lf>=landfrac,'ocean':lf<landfrac}
    return truth,masks

def calc_model_r2(config,name,split,truth,masks):
    '''
    Purpose: Seed-averaged R² of saved predictions over all, land, and ocean grid points.
    Args:
    - config (TimingConfig): configuration object
    - name (str): model or equation name
    - split (str): split name
    - truth (xr.DataArray): truth (time, lat, lon)
    - masks (dict[str, np.ndarray]): region masks
    Returns:
    - dict[str, float] | None: R² per region, or None if predictions are missing
    '''
    filepath = os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')
    if not os.path.exists(filepath):
        logger.warning(f'   Missing {filepath}')
        return None
    pred = load_dataset(filepath)['tp']
    if 'complexity' in pred.dims:
        pred = pred.isel(complexity=0)
    seedpreds = [pred.isel(seed=i) for i in range(pred.sizes['seed'])] if 'seed' in pred.dims else [pred]
    scores = {region:[] for region in REGIONS}
    for seedpred in seedpreds:
        seedtruth,seedpred = xr.align(truth,seedpred.transpose('time','lat','lon'),join='inner')
        if seedtruth.shape!=truth.shape:
            raise ValueError(f'{name}: predictions and truth cover different samples')
        for region in REGIONS:
            scores[region].append(calc_r2(seedtruth.values,seedpred.values,masks[region]))
    return {region:float(np.mean(values)) for region,values in scores.items()}

def calc_kernel_stats(config):
    '''
    Purpose: Peak level and spread of the seed-averaged NN-GAUSS kernels, as in notebooks/weights.ipynb.
    Args:
    - config (TimingConfig): configuration object
    Returns:
    - dict[str, dict[str, float]] | None: per-predictor peak and spread, or None if weights are missing
    '''
    split = load_split(config,config.timing['split'])
    sig   = split['sig'].values.astype(np.float64)
    try:
        kernels = load_kernels(config,split['dsig'].values)
    except FileNotFoundError:
        logger.warning(f'   No NN-GAUSS weights in {config.weightsdir}')
        return None
    dsig  = np.abs(np.gradient(sig))
    stats = {}
    for var,k in kernels.items():
        peak = float(sig[np.argmax(k)])
        stats[var] = dict(peak=peak,spread=float(np.sqrt(((sig-peak)**2*k*dsig).sum())),nseeds=len(config.nn['seeds']))
    return stats

def summarize_sr(config,split,truth,masks):
    '''
    Purpose: For each SR run and seed, report the Pareto equation at the manuscript reference complexity and at the
        elbow, with test R² of the elbow equation; also return the full Pareto frontiers.
    Args:
    - config (TimingConfig): configuration object
    - split (str): split name
    - truth (xr.DataArray): truth (time, lat, lon)
    - masks (dict[str, np.ndarray]): region masks
    Returns:
    - tuple[list[dict], dict[str, pd.DataFrame]]: summary rows and Pareto frontiers keyed by `{run}_{seed}`
    '''
    stats = load_stats(config)
    truthflat = truth.values.ravel()
    maskflat  = {region:mask.ravel() for region,mask in masks.items()}
    rows,fronts = [],{}
    for run,runconfig in config.srruns.items():
        refcomplexity = next((eq['refcomplexity'] for eq in config.srequations.values() if eq['runfrom']==run and eq.get('refcomplexity')),None)
        x = None
        for seed in config.sr['seeds']:
            filepath = os.path.join(config.modelsdir,'sr',f'{run}_{seed}_equations.csv')
            if not os.path.exists(filepath):
                continue
            equations = pd.read_csv(filepath)
            fronts[f'{run}_{seed}'] = equations[['complexity','loss','equation']]
            elbow = select_pareto_elbow(equations)
            ref   = equations[equations['complexity']==refcomplexity]
            row   = dict(run=run,seed=seed,refcomplexity=refcomplexity,
                         refequation=str(ref.iloc[0]['equation']) if not ref.empty else None,
                         elbowcomplexity=int(elbow['complexity']),elbowequation=str(elbow['equation']))
            try:
                if x is None:
                    x,_,_,validmask = load_features(config,split,runconfig)
                    columns = {c:x[c].values for c in x.columns if c!='timeidx'}
                pred = raw_to_precip(evaluate(row['elbowequation'],columns,{}),stats)
                pred[~validmask] = np.nan
                for region in REGIONS:
                    row[f'elbow_r2_{region}'] = calc_r2(truthflat,pred,maskflat[region])
            except Exception as e:
                logger.warning(f'   Could not evaluate `{run}` seed {seed} elbow: {e}')
            rows.append(row)
    return rows,fronts

def summarize(variant,split):
    '''
    Purpose: Collect all requested outputs for one variant (or the current setup when variant is None).
    Args:
    - variant (str | None): variant name
    - split (str): split name
    Returns:
    - dict: summary
    '''
    config = TimingConfig(variant)
    label  = variant or 'current'
    logger.info(f'Summarizing `{label}`...')
    restrict_kernel_seeds(config)
    truth,masks = load_truth(config,split)
    r2 = {}
    for name in [*config.nnruns,*config.srequations]:
        scores = calc_model_r2(config,name,split,truth,masks)
        if scores is not None:
            r2[name] = scores
    srrows,fronts = summarize_sr(config,split,truth,masks)
    return dict(variant=label,ntime=int(truth.sizes['time']),r2=r2,kernels=calc_kernel_stats(config),sr=srrows,fronts=fronts)

def format_markdown(summaries,config):
    '''
    Purpose: Format summaries as Markdown tables.
    Args:
    - summaries (list[dict]): per-variant summaries
    - config (TimingConfig): configuration object
    Returns:
    - str: Markdown text
    '''
    labels = {**{name:run['description'] for name,run in config.nn['runs'].items()},
              **{name:eq['description'] for name,eq in config.sr['optimizedeqs'].items()}}
    lines  = ['# Timing test summary','','## Test R² (seed-averaged; land = LF ≥ 0.5)','',
              '| Model | '+' | '.join(f'{s["variant"]} ({r})' for s in summaries for r in REGIONS)+' |',
              '|---|'+'---|'*(3*len(summaries))]
    names = list(dict.fromkeys(name for s in summaries for name in s['r2']))
    for name in names:
        cells = [f'{s["r2"][name][r]:.3f}' if name in s['r2'] else '–' for s in summaries for r in REGIONS]
        lines.append(f'| {labels.get(name,name)} | '+' | '.join(cells)+' |')
    lines += ['','Timesteps in split: '+', '.join(f'{s["variant"]} = {s["ntime"]}' for s in summaries),'',
              '## NN-GAUSS kernels (seed mean)','','| Variant | Predictor | Peak level | Spread | Seeds |','|---|---|---|---|---|']
    for s in summaries:
        for var,k in (s['kernels'] or {}).items():
            lines.append(f'| {s["variant"]} | {var} | {k["peak"]:.2f} | {k["spread"]:.3f} | {k["nseeds"]} |')
    lines += ['','## PySR equations','','| Variant | Run | Seed | Ref. complexity equation | Elbow (complexity) | Elbow R² all / land / ocean |','|---|---|---|---|---|---|']
    for s in summaries:
        for row in s['sr']:
            r2s = ' / '.join(f'{row[f"elbow_r2_{r}"]:.3f}' for r in REGIONS) if 'elbow_r2_all' in row else '–'
            lines.append(f'| {s["variant"]} | {row["run"]} | {row["seed"]} | `{row["refequation"]}` | `{row["elbowequation"]}` ({row["elbowcomplexity"]}) | {r2s} |')
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Summarize timing-test results.')
    parser.add_argument('--variants',type=str,default='current,all',help='Comma-separated variants; `current` is the existing setup, `all` adds every timing variant')
    parser.add_argument('--split',type=str,default=None,help='Split to summarize (default from tests/timing/configs.json)')
    args = parser.parse_args()
    config   = TimingConfig()
    split    = args.split or config.timing['split']
    allnames = list(config.timing['variants'])
    names    = []
    for name in args.variants.split(','):
        name = name.strip()
        names += ['current'] if name=='current' else parse_names(name,allnames)
    names = list(dict.fromkeys(names))
    summaries = [summarize(None if name=='current' else name,split) for name in names]
    os.makedirs(config.resultsdir,exist_ok=True)
    for s in summaries:
        with open(os.path.join(config.resultsdir,f'{s["variant"]}_pareto.txt'),'w',encoding='utf-8') as f:
            for key,front in s['fronts'].items():
                f.write(f'## {key}\n{front.to_string(index=False)}\n\n')
    jsonpath = os.path.join(config.resultsdir,'summary.json')
    with open(jsonpath,'w',encoding='utf-8') as f:
        json.dump([{k:v for k,v in s.items() if k!='fronts'} for s in summaries],f,indent=2)
    mdpath = os.path.join(config.resultsdir,'summary.md')
    with open(mdpath,'w',encoding='utf-8') as f:
        f.write(format_markdown(summaries,config))
    with open(mdpath,'r',encoding='utf-8') as f:
        logger.info(f'Wrote {jsonpath} and {mdpath}:\n{f.read()}')
