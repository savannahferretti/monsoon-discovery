#!/usr/bin/env python

import os
import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from timingutils import TimingConfig

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

ERA5STORE  = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
SOLARVARS  = ['toa_incident_solar_radiation','mean_top_downward_short_wave_radiation_flux',
              'surface_solar_radiation_downwards','mean_surface_downward_short_wave_radiation_flux']
SOLARLAT   = 0.0
SOLARLON   = 0.0
SOLARDATES = ('2020-03-15','2020-03-25T23:00')

def calc_solar_noon(dates,lon):
    '''
    Purpose: Approximate solar noon (UTC hours) from the equation of time (Spencer 1971).
    Args:
    - dates (pd.DatetimeIndex): dates
    - lon (float): longitude (°E)
    Returns:
    - float: mean solar noon in UTC hours
    '''
    gamma = 2*np.pi*(dates.dayofyear-1)/365.0
    eot   = 229.18*(0.000075+0.001868*np.cos(gamma)-0.032077*np.sin(gamma)-0.014615*np.cos(2*gamma)-0.040849*np.sin(2*gamma))
    return float(np.mean(12.0-lon/15.0-eot/60.0))

def check_era5():
    '''
    Purpose: Infer whether ERA5 accumulations and mean rates at timestamp t cover (t-1h, t] or [t, t+1h) by
        comparing the centroid of the mean diurnal cycle of solar radiation with solar noon.
    '''
    ds    = xr.open_zarr(ERA5STORE,storage_options=dict(token='anon'))
    noon  = calc_solar_noon(pd.date_range(*SOLARDATES,freq='1D'),SOLARLON)
    logger.info(f'ERA5: solar noon at ({SOLARLAT}°N, {SOLARLON}°E) ≈ {noon:.2f} UTC')
    for varname in SOLARVARS:
        if varname not in ds:
            logger.info(f'   `{varname}` not in store, skipping')
            continue
        da       = ds[varname].sel(latitude=SOLARLAT,longitude=SOLARLON).sel(time=slice(*SOLARDATES)).load()
        profile  = da.groupby('time.hour').mean().values
        hours    = np.arange(24)
        centroid = float((hours*profile).sum()/profile.sum())
        offset   = centroid-noon
        if offset>0.25:
            verdict = 'value at t covers the hour ENDING at t'
        elif offset<-0.25:
            verdict = 'value at t covers the hour STARTING at t'
        else:
            verdict = 'value at t is instantaneous'
        logger.info(f'   `{varname}`: centroid = {centroid:.2f} UTC, centroid - noon = {offset:+.2f} h → {verdict}')
        logger.info(f'      Hourly mean (04–09 UTC): {np.array2string(profile[4:10],precision=1)}')
        logger.info(f'      Hourly mean (16–20 UTC): {np.array2string(profile[16:21],precision=1)}')

def check_imerg():
    '''
    Purpose: Print the IMERG time coordinate and bounds to confirm that each half-hourly timestamp marks the
        start of its interval.
    '''
    import fsspec
    import planetary_computer
    import pystac_client as pystac
    catalog = pystac.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',modifier=planetary_computer.sign_inplace)
    assets  = catalog.get_collection('gpm-imerg-hhr').assets['zarr-abfs']
    mapper  = fsspec.get_mapper(assets.href,**assets.extra_fields['xarray:storage_options'])
    ds      = xr.open_zarr(mapper,chunks={},consolidated=True)
    logger.info('IMERG:')
    logger.info(f'   time attrs: {dict(ds.time.attrs)}')
    logger.info(f'   first timestamps: {ds.time.values[:3]}')
    for name in ('time_bnds','time_bounds'):
        if name in ds:
            logger.info(f'   {name} (first 3): {ds[name].isel(time=slice(0,3)).values}')
    logger.info(f'   variables: {list(ds.data_vars)}')

def check_raw(config):
    '''
    Purpose: Confirm that the raw files are hourly, JJA-only, and share timestamps.
    Args:
    - config (TimingConfig): configuration object
    '''
    logger.info('Raw files:')
    for longname in ('ERA5_air_temperature','ERA5_mean_surface_sensible_heat_flux','ERA5_total_accumulated_precipitation','IMERG_V06_precipitation_rate'):
        filepath = os.path.join(config.rawdir,f'{longname}.nc')
        if not os.path.exists(filepath):
            logger.info(f'   {longname}: not found')
            continue
        with xr.open_dataset(filepath,engine='h5netcdf') as ds:
            times = pd.DatetimeIndex(ds.time.values)
        steps = np.unique(np.diff(times.values).astype('timedelta64[m]'))
        logger.info(f'   {longname}: {len(times)} steps, {times[0]} → {times[-1]}, first 3 = {list(times[:3].strftime("%m-%d %H:%M"))}, unique steps = {steps[:4]}')

if __name__=='__main__':
    config = TimingConfig()
    check_raw(config)
    check_era5()
    check_imerg()
