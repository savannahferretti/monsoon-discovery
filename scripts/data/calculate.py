#!/usr/bin/env python

import os
import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scripts.utils import Config
from scripts.data.classes import DataCalculator

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

HOURLYVARS = {
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
    'lf':('Land fraction','0-1'),
    'shf':('Surface sensible heat flux','W/m²'),
    'lhf':('Surface latent heat flux','W/m²'),
    'pr':('Precipitation rate','mm/hr'),
    'tp':('Total precipitation','mm'),
    'dsig':('Sigma thickness weights','0-1')}
WINDOWKINDS = {'rh':'state','thetae':'state','thetaestar':'state','bl':'state','shf':'flux','lhf':'flux','tp':'accum'}
SIGS = np.round(np.arange(0.5,1.05,0.05),2)

def get_windows(timewindow):
    '''
    Purpose: Hourly offsets and weights of a window [T, T+timewindow]. State variables use the trapezoidal mean of the
        hourly values at T, ..., T+timewindow. ERA5 accumulations and mean rates stamped t cover (t-1h, t], so fluxes
        (mean) and precipitation (sum) use the stamps T+1, ..., T+timewindow.
    Args:
    - timewindow (int): window length (hours)
    Returns:
    - dict[str,tuple[np.ndarray,np.ndarray]]: offsets and weights for 'state', 'flux', and 'accum'
    '''
    stateweights = np.ones(timewindow+1)
    stateweights[[0,-1]] = 0.5
    accumoffsets = np.arange(1,timewindow+1)
    return {
        'state':(np.arange(timewindow+1),stateweights/stateweights.sum()),
        'flux':(accumoffsets,np.full(timewindow,1.0/timewindow)),
        'accum':(accumoffsets,np.ones(timewindow))}

def get_starts(hours,year,months,timewindow):
    '''
    Purpose: Indices of window starts T (every timewindow hours from 00 UTC) whose full window exists, checked against
        the number of windows in the configured months.
    Args:
    - hours (pd.DatetimeIndex): contiguous hourly timestamps for one year
    - year (int): year
    - months (list[int]): configured months
    - timewindow (int): window length (hours)
    Returns:
    - np.ndarray: window start indices into hours
    '''
    idx      = np.arange(len(hours))
    starts   = idx[(hours.hour%timewindow==0)&(idx+timewindow<len(hours))]
    expected = sum(pd.Period(f'{year}-{month:02d}').days_in_month for month in months)*24//timewindow
    if len(starts)!=expected:
        raise ValueError(f'{year} has {len(starts)} windows, expected {expected}; the raw files must include 00:00 after the last month')
    return starts

def apply_window(arr,starts,offsets,weights):
    '''
    Purpose: Weighted sum of an hourly array over each window.
    Args:
    - arr (np.ndarray): hourly array with time as the last axis
    - starts (np.ndarray): window start indices
    - offsets (np.ndarray): hourly offsets from the window start
    - weights (np.ndarray): weight per offset
    Returns:
    - np.ndarray: windowed array with time as the last axis
    '''
    out = np.zeros(arr.shape[:-1]+(len(starts),),dtype=np.float64)
    for offset,weight in zip(offsets,weights):
        out += weight*arr[...,starts+offset]
    return out

def load_year(calculator,year):
    '''
    Purpose: Load and regrid one year of the hourly ERA5 variables in float64, and check their timestamps.
    Args:
    - calculator (DataCalculator): calculator instance
    - year (int): year
    Returns:
    - tuple[dict[str,xr.DataArray],pd.DatetimeIndex]: regridded hourly variables and their timestamps
    '''
    raw = {}
    for name,longname in HOURLYVARS.items():
        da = calculator.retrieve(longname)
        raw[name] = calculator.regrid(da.sel(time=da.time.dt.year==year).load().astype(np.float64))
    hours = pd.DatetimeIndex(raw['t'].time.values)
    if not (hours==pd.date_range(hours[0],periods=len(hours),freq='1h')).all():
        raise ValueError(f'Raw timestamps for {year} are not contiguous hourly')
    for name,da in raw.items():
        if not np.array_equal(da.time.values,raw['t'].time.values):
            raise ValueError(f'Timestamps of `{name}` differ from temperature in {year}')
    return raw,hours

def calc_hourly(calculator,raw):
    '''
    Purpose: Compute hourly predictors before time windowing: RH, θₑ, and θₑ* on sigma levels, BL, and the surface
        fluxes and precipitation.
    Args:
    - calculator (DataCalculator): calculator instance
    - raw (dict[str,xr.DataArray]): regridded hourly raw variables
    Returns:
    - dict[str,np.ndarray]: hourly arrays with dims (lat, lon, [sig,] time)
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
    return {
        'rh':calculator.interpolate_to_sigma(rh,ps,SIGS).clip(0.0,100.0).values,
        'thetae':calculator.interpolate_to_sigma(thetae,ps,SIGS).values,
        'thetaestar':calculator.interpolate_to_sigma(thetaestar,ps,SIGS).values,
        'bl':bl.transpose('lat','lon','time').values,
        'shf':raw['shf'].transpose('lat','lon','time').values,
        'lhf':raw['lhf'].transpose('lat','lon','time').values,
        'tp':raw['tp'].transpose('lat','lon','time').values}

def calc_pr(calculator,year,timewindow,times):
    '''
    Purpose: Mean IMERG precipitation rate over each window; half-hourly stamps mark the start of each interval, so
        stamps T, ..., T+timewindow-0.5h cover [T, T+timewindow].
    Args:
    - calculator (DataCalculator): calculator instance
    - year (int): year
    - timewindow (int): window length (hours)
    - times (pd.DatetimeIndex): ERA5 window start times for the year
    Returns:
    - xr.DataArray: windowed precipitation rate with dims (lat, lon, time)
    '''
    da = calculator.retrieve('IMERG_V06_precipitation_rate')
    da = calculator.regrid(da.sel(time=da.time.dt.year==year).load().astype(np.float64))
    windows = pd.DatetimeIndex(da.time.values).floor(f'{timewindow}h')
    pr = da.assign_coords(window=('time',windows)).groupby('window').mean().rename({'window':'time'})
    pr = pr.where(pr>=1e-6,0.0).transpose('lat','lon','time')
    if not np.array_equal(pr.time.values,times.values):
        raise ValueError(f'IMERG windows for {year} differ from ERA5 windows')
    return pr

def to_dataarray(arr,lat,lon,times):
    '''
    Purpose: Wrap a windowed array as an xr.DataArray.
    Args:
    - arr (np.ndarray): array with dims (lat, lon, [sig,] time)
    - lat (np.ndarray): latitudes
    - lon (np.ndarray): longitudes
    - times (pd.DatetimeIndex): window start times
    Returns:
    - xr.DataArray: DataArray
    '''
    if arr.ndim==4:
        return xr.DataArray(arr,dims=('lat','lon','sig','time'),coords={'lat':lat,'lon':lon,'sig':SIGS,'time':times})
    return xr.DataArray(arr,dims=('lat','lon','time'),coords={'lat':lat,'lon':lon,'time':times})

if __name__=='__main__':
    config     = Config()
    calculator = DataCalculator(
        author=config.author,
        email=config.email,
        filedir=config.rawdir,
        savedir=config.interimdir,
        latrange=config.latrange,
        lonrange=config.lonrange)
    todo = [name for name in METADATA if not os.path.exists(os.path.join(config.interimdir,f'{name}.nc'))]
    if not todo:
        logger.info('Skipping, all interim files already exist')
    else:
        timewindow = config.timewindow
        windows    = get_windows(timewindow)
        logger.info(f'{timewindow}-hourly windows: state offsets {windows["state"][0].tolist()} (weights {np.round(windows["state"][1],4).tolist()}), flux/precipitation offsets {windows["flux"][0].tolist()}')
        collected = {name:[] for name in [*WINDOWKINDS,'pr']}
        lf = None
        for year in config.years:
            logger.info(f'Processing {year}...')
            raw,hours = load_year(calculator,year)
            if lf is None:
                lf = calculator.regrid(calculator.retrieve('ERA5_land_fraction').load().astype(np.float64))
            starts  = get_starts(hours,year,config.months,timewindow)
            times   = hours[starts]
            lat,lon = raw['t'].lat.values,raw['t'].lon.values
            hourly  = calc_hourly(calculator,raw)
            del raw
            for name,kind in WINDOWKINDS.items():
                values = apply_window(hourly[name],starts,*windows[kind])
                if name=='tp':
                    values = np.where(values>=1e-4,values,0.0)
                collected[name].append(to_dataarray(values,lat,lon,times))
            collected['pr'].append(calc_pr(calculator,year,timewindow,times))
            del hourly
        fields = {name:xr.concat(das,dim='time') for name,das in collected.items()}
        fields['lf']   = lf
        fields['dsig'] = calculator.calc_dsig(SIGS)
        logger.info('Saving datasets...')
        for name in todo:
            longname,units = METADATA[name]
            calculator.save(calculator.create_dataset(fields[name],name,longname,units))
