#!/usr/bin/env python

import os
import shutil
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from timingutils import TimingConfig,parse_names
from scripts.data.classes import DataCalculator

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

RAWFILES = {
    't':'ERA5_air_temperature',
    'q':'ERA5_specific_humidity',
    'ps':'ERA5_surface_pressure',
    'shf':'ERA5_mean_surface_sensible_heat_flux',
    'lhf':'ERA5_mean_surface_latent_heat_flux',
    'tp':'ERA5_total_accumulated_precipitation'}
METADATA = {
    'rh':('Relative humidity','%'),
    'thetae':('Equivalent potential temperature','K'),
    'thetaestar':('Saturated equivalent potential temperature','K'),
    'bl':('Average buoyancy in the lower troposphere','m/s²'),
    'shf':('Surface sensible heat flux','W/m²'),
    'lhf':('Surface latent heat flux','W/m²'),
    'tp':('Total precipitation','mm')}
STATEVARS  = ['rh','thetae','thetaestar','bl']
FLUXVARS   = ['shf','lhf']
STATICVARS = ['lf','dsig']
SIGS       = np.arange(0.5,1.05,0.05,dtype=np.float32)

def parse():
    '''
    Purpose: Parse command-line arguments.
    Returns:
    - tuple[str, int | None]: variants argument and optional check year
    '''
    parser = argparse.ArgumentParser(description='Build interim data for the timing variants.')
    parser.add_argument('--variants',type=str,default='all',help='Comma-separated variant names, or `all`')
    parser.add_argument('--check',type=int,default=None,help='Instead of building variants, rebuild the current setup for this year and compare it with the existing interim files')
    args = parser.parse_args()
    return args.variants,args.check

def get_state_offsets(window):
    '''
    Purpose: Hourly offsets and weights for an instantaneous sample (start==end) or a trapezoidal time mean of
        instantaneous values over [start, end].
    Args:
    - window (list[int]): [start, end] in hours relative to the window start T
    Returns:
    - tuple[list[int], np.ndarray]: offsets and weights (weights sum to 1)
    '''
    start,end = window
    if start==end:
        return [start],np.array([1.0])
    offsets = list(range(start,end+1))
    weights = np.ones(len(offsets))
    weights[[0,-1]] = 0.5
    return offsets,weights/weights.sum()

def get_accum_offsets(window,label):
    '''
    Purpose: Hourly timestamps whose 1-hour accumulations or mean rates tile the interval [start, end].
    Args:
    - window (list[int]): [start, end] in hours relative to the window start T
    - label (str): 'end' if the value at t covers (t-1h, t]; 'start' if it covers [t, t+1h)
    Returns:
    - list[int]: offsets
    '''
    start,end = window
    return list(range(start+1,end+1)) if label=='end' else list(range(start,end))

def build_specs(timing,variants):
    '''
    Purpose: Translate each variant's windows into hourly offsets and weights.
    Args:
    - timing (dict): timing configuration block
    - variants (list[str]): variant names
    Returns:
    - dict[str, dict]: per-variant {'state':(offsets,weights), 'flux':(...), 'target':(...)}
    '''
    label = timing['accumlabel']
    specs = {}
    for variant in variants:
        spec = timing['variants'][variant]
        fluxoffsets   = get_accum_offsets(spec['fluxwindow'],label)
        targetoffsets = get_accum_offsets(spec['targetwindow'],label)
        specs[variant] = {
            'state':get_state_offsets(spec['statewindow']),
            'flux':(fluxoffsets,np.full(len(fluxoffsets),1.0/len(fluxoffsets))),
            'target':(targetoffsets,np.ones(len(targetoffsets)))}
    return specs

def get_anchors(hours,minoffset,maxoffset):
    '''
    Purpose: Indices of 3-hourly window starts (00, 03, ..., 21 UTC) whose required hourly offsets all exist.
    Args:
    - hours (pd.DatetimeIndex): contiguous hourly timestamps
    - minoffset (int): most negative offset required
    - maxoffset (int): most positive offset required
    Returns:
    - np.ndarray: anchor indices into hours
    '''
    idx = np.arange(len(hours))
    return idx[(hours.hour%3==0)&(idx+minoffset>=0)&(idx+maxoffset<len(hours))]

def apply_window(arr,anchors,offsets,weights):
    '''
    Purpose: Weighted sum of an hourly array (time last) at anchors+offsets.
    Args:
    - arr (np.ndarray): hourly array with time as the last axis
    - anchors (np.ndarray): anchor indices
    - offsets (list[int]): hourly offsets
    - weights (np.ndarray): weight per offset
    Returns:
    - np.ndarray: windowed array (float32) with time as the last axis
    '''
    out = np.zeros(arr.shape[:-1]+(len(anchors),),dtype=np.float64)
    for offset,weight in zip(offsets,weights):
        out += weight*arr[...,anchors+offset]
    return out.astype(np.float32)

def load_year(calculator,year):
    '''
    Purpose: Load, subset to one year, and regrid all hourly raw variables.
    Args:
    - calculator (DataCalculator): calculator instance
    - year (int): year
    Returns:
    - tuple[dict[str, xr.DataArray], pd.DatetimeIndex]: regridded hourly variables and their timestamps
    '''
    raw = {}
    for name,longname in RAWFILES.items():
        da = calculator.retrieve(longname)
        da = da.sel(time=da.time.dt.year==year)
        raw[name] = calculator.regrid(da).load()
    hours    = pd.DatetimeIndex(raw['t'].time.values)
    expected = pd.date_range(hours[0],hours[-1],freq='1h')
    if len(hours)!=len(expected) or not (hours==expected).all():
        raise ValueError(f'Raw timestamps for {year} are not contiguous hourly')
    for name,da in raw.items():
        if not (pd.DatetimeIndex(da.time.values)==hours).all():
            raise ValueError(f'Timestamps of `{name}` differ from temperature in {year}')
    return raw,hours

def calc_hourly(calculator,raw):
    '''
    Purpose: Compute hourly predictors exactly as in scripts/data/calculate.py, but before any time resampling.
    Args:
    - calculator (DataCalculator): calculator instance
    - raw (dict[str, xr.DataArray]): regridded hourly raw variables
    Returns:
    - dict[str, np.ndarray]: hourly arrays with dims (lat, lon, [sig,] time)
    '''
    t,q,ps      = raw['t'],raw['q'],raw['ps']
    p           = calculator.create_p_array(q)
    rh          = calculator.calc_rh(p,t,q)
    thetae      = calculator.calc_thetae(p,t,q)
    thetaestar  = calculator.calc_thetae(p,t)
    pbltop      = ps-100.0
    lfttop      = xr.full_like(ps,500.0)
    thetaeb     = calculator.calc_layer_average(thetae,ps,pbltop)
    thetael     = calculator.calc_layer_average(thetae,pbltop,lfttop)
    thetaelstar = calculator.calc_layer_average(thetaestar,pbltop,lfttop)
    wb,wl       = calculator.calc_weights(ps,pbltop,lfttop)
    bl          = calculator.calc_bl(thetaeb,thetael,thetaelstar,wb,wl)
    hourly = {
        'rh':calculator.interpolate_to_sigma(rh,ps,SIGS).values,
        'thetae':calculator.interpolate_to_sigma(thetae,ps,SIGS).values,
        'thetaestar':calculator.interpolate_to_sigma(thetaestar,ps,SIGS).values,
        'bl':bl.transpose('lat','lon','time').values}
    for name in ('shf','lhf','tp'):
        hourly[name] = raw[name].transpose('lat','lon','time').values
    return hourly

def to_dataarray(arr,name,lat,lon,times):
    '''
    Purpose: Wrap a windowed array as an xr.DataArray.
    Args:
    - arr (np.ndarray): array with dims (lat, lon, [sig,] time)
    - name (str): variable name
    - lat (np.ndarray): latitudes
    - lon (np.ndarray): longitudes
    - times (pd.DatetimeIndex): window start times
    Returns:
    - xr.DataArray: DataArray
    '''
    if arr.ndim==4:
        return xr.DataArray(arr,dims=('lat','lon','sig','time'),coords={'lat':lat,'lon':lon,'sig':SIGS,'time':times},name=name)
    return xr.DataArray(arr,dims=('lat','lon','time'),coords={'lat':lat,'lon':lon,'time':times},name=name)

def window_variant(hourly,spec,anchors,threshold):
    '''
    Purpose: Apply a variant's state, flux, and target windows to hourly arrays.
    Args:
    - hourly (dict[str, np.ndarray]): hourly arrays
    - spec (dict): offsets and weights from build_specs()
    - anchors (np.ndarray): anchor indices
    - threshold (float): precipitation threshold (mm per window)
    Returns:
    - dict[str, np.ndarray]: windowed arrays
    '''
    out = {}
    for name in STATEVARS:
        out[name] = apply_window(hourly[name],anchors,*spec['state'])
    for name in FLUXVARS:
        out[name] = apply_window(hourly[name],anchors,*spec['flux'])
    tp = apply_window(hourly['tp'],anchors,*spec['target'])
    out['tp'] = np.where(tp>=threshold,tp,0.0).astype(np.float32)
    return out

def run_check(config,calculator,year):
    '''
    Purpose: Rebuild the current setup (state at T; fluxes and precipitation from the hourly values stamped T, T+1,
        T+2) for one year with this code, and compare with the existing interim files to validate the reimplementation.
    Args:
    - config (TimingConfig): configuration object
    - calculator (DataCalculator): calculator instance
    - year (int): year to check
    '''
    logger.info(f'Rebuilding the current setup for {year}...')
    raw,hours = load_year(calculator,year)
    lat,lon   = raw['t'].lat.values,raw['t'].lon.values
    hourly    = calc_hourly(calculator,raw)
    del raw
    spec    = {'state':([0],np.array([1.0])),'flux':([0,1,2],np.full(3,1/3)),'target':([0,1,2],np.ones(3))}
    anchors = get_anchors(hours,0,2)
    rebuilt = window_variant(hourly,spec,anchors,config.timing['threshold'])
    times   = hours[anchors]
    for name,arr in rebuilt.items():
        filepath = os.path.join(config.mainfilepaths['interim'],f'{name}.nc')
        with xr.open_dataarray(filepath,engine='h5netcdf') as existing:
            dims     = ('lat','lon','sig','time') if arr.ndim==4 else ('lat','lon','time')
            existing = existing.sel(time=times).transpose(*dims).values
        diff   = np.abs(arr.astype(np.float64)-existing)
        scale  = np.nanmax(np.abs(existing))
        nmismatch = int(np.sum(~np.isclose(arr,existing,rtol=1e-4,atol=1e-6*scale,equal_nan=True)))
        logger.info(f'   {name}: max |diff| = {np.nanmax(diff):.3e} (field max {scale:.3e}), mismatched samples = {nmismatch}/{arr.size}')

if __name__=='__main__':
    config    = TimingConfig()
    timing    = config.timing
    variants,checkyear = parse()
    calculator = DataCalculator(
        author=config.author,
        email=config.email,
        filedir=config.rawdir,
        savedir=None,
        latrange=config.latrange,
        lonrange=config.lonrange)
    if checkyear is not None:
        run_check(config,calculator,checkyear)
    else:
        variants  = parse_names(variants,list(timing['variants']))
        allspecs  = build_specs(timing,list(timing['variants']))
        alloffsets = [offset for spec in allspecs.values() for key in ('state','flux','target') for offset in spec[key][0]]
        minoffset,maxoffset = min(alloffsets),max(alloffsets)
        logger.info(f'Window starts need hourly data from T{minoffset:+d} h to T{maxoffset:+d} h (shared across all variants)')
        todo = []
        for variant in variants:
            interimdir = TimingConfig(variant).interimdir
            if all(os.path.exists(os.path.join(interimdir,f'{name}.nc')) for name in [*METADATA,*STATICVARS]):
                logger.info(f'Skipping `{variant}`, interim files already exist')
            else:
                todo.append(variant)
        for variant in todo:
            spec = allspecs[variant]
            logger.info(f'`{variant}`: state offsets {spec["state"][0]} (weights {np.round(spec["state"][1],3).tolist()}), flux offsets {spec["flux"][0]}, target offsets {spec["target"][0]}')
        collected = {variant:{name:[] for name in METADATA} for variant in todo}
        for year in (config.years if todo else []):
            logger.info(f'Processing {year}...')
            raw,hours = load_year(calculator,year)
            lat,lon   = raw['t'].lat.values,raw['t'].lon.values
            hourly    = calc_hourly(calculator,raw)
            del raw
            anchors = get_anchors(hours,minoffset,maxoffset)
            times   = hours[anchors]
            for variant in todo:
                windowed = window_variant(hourly,allspecs[variant],anchors,timing['threshold'])
                for name,arr in windowed.items():
                    collected[variant][name].append(to_dataarray(arr,name,lat,lon,times))
            del hourly
        for variant in todo:
            interimdir = TimingConfig(variant).interimdir
            calculator.savedir = interimdir
            logger.info(f'Saving `{variant}` to {interimdir}...')
            for name,(longname,units) in METADATA.items():
                da = xr.concat(collected[variant][name],dim='time')
                calculator.save(calculator.create_dataset(da,name,longname,units))
            for name in STATICVARS:
                shutil.copy(os.path.join(config.mainfilepaths['interim'],f'{name}.nc'),os.path.join(interimdir,f'{name}.nc'))
                xr.open_dataarray(os.path.join(interimdir,f'{name}.nc'),engine='h5netcdf').close()
                logger.info(f'   Copied static {name}.nc from the current interim directory')
            del collected[variant]
