#!/usr/bin/env python

import os
import sys
import logging
import argparse
import warnings
import numpy as np
import pandas as pd

TIMINGDIR = os.path.dirname(os.path.abspath(__file__))
REPODIR   = os.path.dirname(os.path.dirname(TIMINGDIR))
if REPODIR not in sys.path:
    sys.path.insert(0,REPODIR)

from scripts.utils import Config,load
from scripts.data.classes import DataCalculator

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

PROFILEVARS = ['rh','thetae','thetaestar']
SCALARVARS  = ['bl','shf','lhf','tp']
RTOL        = 1e-5

def calc_affected(config,year,times,sigs):
    '''
    Purpose: Flag windows where any hour has σ·ps outside the pressure-level range, i.e. where clamping applies.
    Args:
    - config (Config): configuration object
    - year (int): year
    - times (pd.DatetimeIndex): window start times to flag
    - sigs (np.ndarray): sigma levels
    Returns:
    - np.ndarray: boolean array with dims (lat, lon, sig, time)
    '''
    calculator = DataCalculator(author=config.author,email=config.email,filedir=config.rawdir,savedir=None,latrange=config.latrange,lonrange=config.lonrange)
    da = calculator.retrieve('ERA5_surface_pressure')
    ps = calculator.regrid(da.sel(time=da.time.dt.year==year).load().astype(np.float64)).transpose('lat','lon','time')
    hours   = pd.DatetimeIndex(ps.time.values)
    levmin,levmax = config.levrange
    target  = sigs[None,None,:,None]*ps.values[:,:,None,:]
    outside = (target<levmin)|(target>levmax)
    starts  = hours.get_indexer(times)
    affected = np.zeros(outside.shape[:-1]+(len(times),),dtype=bool)
    for offset in range(config.timewindow+1):
        affected |= outside[...,starts+offset]
    return affected

def compare(name,new,old,affected=None):
    '''
    Purpose: Log agreement between new and timing-test values, separately for clamped windows if given.
    Args:
    - name (str): variable name
    - new (np.ndarray): new values
    - old (np.ndarray): timing-test values
    - affected (np.ndarray | None): boolean mask of windows where clamping applies
    '''
    scale = np.nanmax(np.abs(old))
    close = np.isclose(new,old,rtol=RTOL,atol=RTOL*scale,equal_nan=True)
    diff  = np.abs(new-old)
    if affected is None:
        logger.info(f'   {name}: mismatched {int((~close).sum())}/{close.size}, max |diff| {np.nanmax(diff):.3e} (field max {scale:.3e})')
        return
    keep = ~affected
    logger.info(f'   {name}, unclamped windows: mismatched {int((~close[keep]).sum())}/{int(keep.sum())}, max |diff| {np.nanmax(diff[keep]):.3e} (field max {scale:.3e})')
    logger.info(f'   {name}, clamped windows (differences expected): mismatched {int((~close[affected]).sum())}/{int(affected.sum())}, max |diff| {np.nanmax(diff[affected]) if affected.any() else 0.0:.3e}')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compare one year of the new interim files with the concurrent timing-test interim files.')
    parser.add_argument('--year',type=int,default=2015,help='Year to compare (default: 2015)')
    year   = parser.parse_args().year
    config = Config()
    olddir = os.path.join(TIMINGDIR,'data','concurrent','interim')
    logger.info(f'Comparing {year}: {config.interimdir} vs {olddir}')
    newtp  = load(os.path.join(config.interimdir,'tp.nc'))['tp']
    oldtp  = load(os.path.join(olddir,'tp.nc'))['tp']
    newtimes = pd.DatetimeIndex(newtp.time.values[newtp.time.dt.year.values==year])
    oldtimes = pd.DatetimeIndex(oldtp.time.values[oldtp.time.dt.year.values==year])
    shared   = newtimes.intersection(oldtimes)
    extra    = newtimes.difference(oldtimes)
    logger.info(f'New windows in {year}: {len(newtimes)} (expected {sum(pd.Period(f"{year}-{m:02d}").days_in_month for m in config.months)*24//config.timewindow}), timing test: {len(oldtimes)}, shared: {len(shared)}')
    logger.info(f'Windows only in the new files: {[str(t) for t in extra]}')
    sigs = None
    affected = None
    for name in SCALARVARS+PROFILEVARS:
        new = load(os.path.join(config.interimdir,f'{name}.nc'))[name]
        old = load(os.path.join(olddir,f'{name}.nc'))[name]
        dims = ('lat','lon','sig','time') if 'sig' in new.dims else ('lat','lon','time')
        if not (np.array_equal(new.lat.values,old.lat.values) and np.array_equal(new.lon.values,old.lon.values)):
            raise ValueError(f'`{name}` grids differ')
        logger.info(f'   {name}: new windows finite = {bool(np.isfinite(new.sel(time=extra).values).all())}' if len(extra) else f'   {name}: no new windows')
        newvalues = new.sel(time=shared).transpose(*dims).values
        oldvalues = old.sel(time=shared).transpose(*dims).values
        if 'sig' in new.dims:
            if affected is None:
                sigs     = new.sig.values
                affected = calc_affected(config,year,shared,sigs)
                logger.info(f'   Windows where clamping applies: {100*affected.mean():.2f}% of profile values')
            compare(name,newvalues,oldvalues,affected)
        else:
            compare(name,newvalues,oldvalues)
