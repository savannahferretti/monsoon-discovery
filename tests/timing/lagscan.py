#!/usr/bin/env python

import os
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from timingutils import TimingConfig,load_dataset,calc_r2,COMPUTEDTYPE
from calculate import load_year,get_anchors,get_state_offsets,get_accum_offsets,apply_window
from equations import evaluate
from scripts.data.classes import DataCalculator

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

SNAPSHOTS = list(range(-3,4))
AVERAGES  = [[-3,0],[-2,1],[-1,2],[0,3]]
TARGETS   = {'concurrent':[0,3],'current':[-1,2]}
FORM      = 'cube(bl+c1)+c2'
INITS     = [{'c1':0.3,'c2':0.14},{'c1':0.0,'c2':0.0},{'c1':1.0,'c2':0.0},{'c1':-1.0,'c2':0.5}]
NBINS     = 40
REGIONS   = ('all','land','ocean')

def build_predictors():
    '''
    Purpose: B_L predictors to scan: hourly snapshots at T+k and trapezoidal 3-hour means.
    Returns:
    - dict[str, dict]: label → {offsets, weights, centre (hours after T), kind}
    '''
    predictors = {}
    for k in SNAPSHOTS:
        offsets,weights = get_state_offsets([k,k])
        predictors[f'snapshot T{k:+d}'] = dict(offsets=offsets,weights=weights,centre=float(k),kind='snapshot')
    for start,end in AVERAGES:
        offsets,weights = get_state_offsets([start,end])
        predictors[f'mean T{start:+d}..T{end:+d}'] = dict(offsets=offsets,weights=weights,centre=(start+end)/2,kind='mean')
    return predictors

def calc_hourly_bl(calculator,raw):
    '''
    Purpose: Hourly B_L, computed as in scripts/data/calculate.py.
    Args:
    - calculator (DataCalculator): calculator instance
    - raw (dict[str, xr.DataArray]): regridded hourly t, q, ps
    Returns:
    - np.ndarray: B_L with dims (lat, lon, time)
    '''
    t,q,ps      = raw['t'],raw['q'],raw['ps']
    p           = calculator.create_p_array(q)
    thetae      = calculator.calc_thetae(p,t,q)
    thetaestar  = calculator.calc_thetae(p,t)
    pbltop      = ps-100.0
    lfttop      = ps*0.0+500.0
    thetaeb     = calculator.calc_layer_average(thetae,ps,pbltop)
    thetael     = calculator.calc_layer_average(thetae,pbltop,lfttop)
    thetaelstar = calculator.calc_layer_average(thetaestar,pbltop,lfttop)
    wb,wl       = calculator.calc_weights(ps,pbltop,lfttop)
    return calculator.calc_bl(thetaeb,thetael,thetaelstar,wb,wl).transpose('lat','lon','time').values

def build_data(config,calculator,predictors,threshold,label):
    '''
    Purpose: Window hourly B_L and precipitation for every predictor and target, for all years.
    Args:
    - config (TimingConfig): configuration object
    - calculator (DataCalculator): calculator instance
    - predictors (dict): from build_predictors()
    - threshold (float): precipitation threshold (mm per window)
    - label (str): 'end' or 'start' accumulation labelling
    Returns:
    - tuple[dict, dict, np.ndarray]: predictor arrays and target arrays with dims (lat, lon, time), and the year of
        each window
    '''
    alloffsets = [o for p in predictors.values() for o in p['offsets']]+[o for w in TARGETS.values() for o in get_accum_offsets(w,label)]
    xs  = {name:[] for name in predictors}
    ys  = {name:[] for name in TARGETS}
    yrs = []
    for year in config.years:
        logger.info(f'Processing {year}...')
        raw,hours = load_year(calculator,year,COMPUTEDTYPE,names=['t','q','ps','tp'])
        bl      = calc_hourly_bl(calculator,raw)
        tp      = raw['tp'].transpose('lat','lon','time').values
        anchors = get_anchors(hours,min(alloffsets),max(alloffsets))
        for name,p in predictors.items():
            xs[name].append(apply_window(bl,anchors,p['offsets'],p['weights']))
        for name,window in TARGETS.items():
            offsets = get_accum_offsets(window,label)
            total   = apply_window(tp,anchors,offsets,np.ones(len(offsets)))
            ys[name].append(np.where(total>=threshold,total,0.0))
        yrs.append(np.full(len(anchors),year))
        del raw,bl,tp
    return {k:np.concatenate(v,axis=-1) for k,v in xs.items()},{k:np.concatenate(v,axis=-1) for k,v in ys.items()},np.concatenate(yrs)

def fit_srbl(x,y,zmin):
    '''
    Purpose: Fit the SR-BL form to standardized B_L and the standardized log1p target (same objective as the
        timing optimizer), from several starts.
    Args:
    - x (np.ndarray): standardized B_L
    - y (np.ndarray): standardized target
    - zmin (float): standardized value of zero precipitation
    Returns:
    - dict[str, float]: fitted constants
    '''
    def loss(params):
        raw = evaluate(FORM,{'bl':x},{'c1':params[0],'c2':params[1]})
        return float(np.mean((zmin+np.maximum(raw,0.0)-y)**2))
    best = min((minimize(loss,np.array([init['c1'],init['c2']]),method='L-BFGS-B') for init in INITS),key=lambda res:res.fun)
    return {'c1':float(best.x[0]),'c2':float(best.x[1])}

def fit_binned(x,y,xtest):
    '''
    Purpose: Model-free reference: mean precipitation in quantile bins of the predictor (fit set), applied to the
        test set.
    Args:
    - x (np.ndarray): fit-set predictor
    - y (np.ndarray): fit-set precipitation (mm)
    - xtest (np.ndarray): test-set predictor
    Returns:
    - np.ndarray: test-set predictions (mm)
    '''
    edges = np.unique(np.quantile(x,np.linspace(0,1,NBINS+1)))
    idx   = np.clip(np.searchsorted(edges,x,side='right')-1,0,len(edges)-2)
    means = np.bincount(idx,weights=y,minlength=len(edges)-1)/np.maximum(np.bincount(idx,minlength=len(edges)-1),1)
    pred = means[np.clip(np.searchsorted(edges,xtest,side='right')-1,0,len(edges)-2)]
    return np.where(np.isfinite(xtest),pred,np.nan)

def plot(df,filepath):
    '''
    Purpose: Test R² against the predictor's centre time, one panel per target window.
    Args:
    - df (pd.DataFrame): scan results
    - filepath (str): output image path
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(1,len(TARGETS),figsize=(4.2*len(TARGETS),3.2),sharey=True,constrained_layout=True)
    for ax,(target,window) in zip(np.atleast_1d(axs),TARGETS.items()):
        ax.axvspan(*window,color='0.9',zorder=0,label='rain window')
        for model,style in (('srbl','-'),('binned',':')):
            sub = df[(df.target==target)&(df.kind=='snapshot')].sort_values('centre')
            ax.plot(sub.centre,sub[f'{model}_r2_all'],style,marker='o',color='k',label=f'snapshot ({model})')
            sub = df[(df.target==target)&(df.kind=='mean')]
            ax.scatter(sub.centre,sub[f'{model}_r2_all'],marker='s' if model=='srbl' else 'D',color='tab:orange',zorder=3,label=f'3-h mean ({model})')
        ax.set_title(f'{target} rain window (T{window[0]:+d} to T{window[1]:+d} h)')
        ax.set_xlabel('B_L time relative to T (h; centre for means)')
    np.atleast_1d(axs)[0].set_ylabel('Test R²')
    np.atleast_1d(axs)[0].legend(fontsize=7,frameon=False)
    fig.savefig(filepath,dpi=200)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Scan B_L snapshot times and 3-hour means against fixed precipitation windows with the SR-BL form.')
    parser.parse_args()
    config     = TimingConfig()
    timing     = config.timing
    predictors = build_predictors()
    calculator = DataCalculator(author=config.author,email=config.email,filedir=config.rawdir,savedir=None,latrange=config.latrange,lonrange=config.lonrange)
    xs,ys,years = build_data(config,calculator,predictors,timing['threshold'],timing['accumlabel'])
    lf     = load_dataset(os.path.join(config.mainfilepaths['interim'],'lf.nc'))['lf'].transpose('lat','lon').values
    nwin   = len(years)
    lf     = np.broadcast_to(lf[:,:,None],lf.shape+(nwin,)).reshape(-1)
    yearsf = np.broadcast_to(years[None,None,:],xs[next(iter(xs))].shape).reshape(-1)
    fitset  = (yearsf>=config.trainrange[0])&(yearsf<=config.validrange[1])
    trainset = (yearsf>=config.trainrange[0])&(yearsf<=config.trainrange[1])
    testset = (yearsf>=config.testrange[0])&(yearsf<=config.testrange[1])
    masks   = {'all':np.ones(testset.sum(),dtype=bool),'land':lf[testset]>=timing['landfrac'],'ocean':lf[testset]<timing['landfrac']}
    rows = []
    for target in TARGETS:
        y = ys[target].reshape(-1)
        ylog = np.log1p(y)
        ymean,ystd = ylog[trainset].mean(),ylog[trainset].std()
        z,zmin = (ylog-ymean)/ystd,-ymean/ystd
        for name,p in predictors.items():
            x = xs[name].reshape(-1)
            xmean,xstd = np.nanmean(x[trainset]),np.nanstd(x[trainset])
            xz    = (x-xmean)/xstd
            valid = np.isfinite(xz)&np.isfinite(y)
            fit,test = fitset&valid,testset
            consts = fit_srbl(xz[fit],z[fit],zmin)
            srbl   = np.expm1(ystd*np.maximum(evaluate(FORM,{'bl':xz[test]},consts),0.0))
            binned = fit_binned(x[fit],y[fit],x[test])
            row = dict(target=target,predictor=name,kind=p['kind'],centre=p['centre'],**consts)
            for region,mask in masks.items():
                row[f'srbl_r2_{region}']   = calc_r2(y[test],srbl,mask)
                row[f'binned_r2_{region}'] = calc_r2(y[test],binned,mask)
            logger.info(f'{target:>10} | {name:<16} | SR-BL R² = {row["srbl_r2_all"]:.3f} | binned R² = {row["binned_r2_all"]:.3f}')
            rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(config.resultsdir,exist_ok=True)
    df.to_csv(os.path.join(config.resultsdir,'lagscan.csv'),index=False)
    columns = ['target','predictor','srbl_r2_all','srbl_r2_land','srbl_r2_ocean','binned_r2_all']
    lines   = ['| '+' | '.join(columns)+' |','|'+'---|'*len(columns)]
    lines  += ['| '+' | '.join(f'{row[c]:.3f}' if isinstance(row[c],float) else str(row[c]) for c in columns)+' |' for _,row in df.iterrows()]
    with open(os.path.join(config.resultsdir,'lagscan.md'),'w',encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    plot(df,os.path.join(config.resultsdir,'lagscan.png'))
    logger.info(f'Wrote lagscan.csv, lagscan.md, and lagscan.png to {config.resultsdir}')
