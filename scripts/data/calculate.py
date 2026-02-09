#!/usr/bin/env python

import logging
import warnings
import numpy as np
import xarray as xr
from scripts.utils import Config
from scripts.data.classes import DataCalculator

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

if __name__=='__main__':
    config     = Config()
    calculator = DataCalculator(
        author=config.author,
        email=config.email,
        filedir=config.rawdir,
        savedir=config.interimdir,
        latrange=config.latrange,
        lonrange=config.lonrange)
    logger.info('Importing all raw variables...')
    ps  = calculator.retrieve('ERA5_surface_pressure')
    t   = calculator.retrieve('ERA5_air_temperature')
    q   = calculator.retrieve('ERA5_specific_humidity')
    lf  = calculator.retrieve('ERA5_land_fraction')
    pr  = calculator.retrieve('IMERG_V06_precipitation_rate')
    logger.info('Resampling/regridding variables...')
    ps  = calculator.regrid(ps).load()
    t   = calculator.regrid(t).load()
    q   = calculator.regrid(q).load()
    lf  = calculator.regrid(lf).load()
    pr  = calculator.regrid(calculator.resample(pr)).clip(min=0).load()
    logger.info('Calculating equivalent potential temperature terms...')
    p           = calculator.create_p_array(q)
    thetae      = calculator.calc_thetae(p,t,q)
    thetaestar  = calculator.calc_thetae(p,t)
    thetaesurf  = calculator.calc_thetae(p,t,q,ps)
    thetaeprime = thetaesurf-thetaestar
    thetaeplus  = thetaestar-thetae    
    logger.info('Calculating layer averages...')
    pbltop      = ps-100.0
    lfttop      = xr.full_like(ps,500.0)
    thetaeb     = calculator.calc_layer_average(thetae,ps,pbltop)
    thetael     = calculator.calc_layer_average(thetae,pbltop,lfttop)
    thetaelstar = calculator.calc_layer_average(thetaestar,pbltop,lfttop)
    logger.info('Calculating BL terms...')
    wb,wl          = calculator.calc_weights(ps,pbltop,lfttop)
    cape,subsat,bl = calculator.calc_bl_terms(thetaeb,thetael,thetaelstar,wb,wl)
    logger.info('Calculating quadrature weights...')
    dlev = calculator.calc_dlev(t)
    logger.info('Calculating surface mask...')
    surfmask = xr.where(p<=ps,1.0,0.0).astype(np.float32)
    logger.info('Creating datasets...')
    dslist = [
        calculator.create_dataset(t,'t','Air temperature','K'),
        calculator.create_dataset(q,'q','Specific humidity','kg/kg'),
        calculator.create_dataset(thetaeprime,'thetaeprime','Difference between surface and unsaturated equivalent potential temperature','K'),
        calculator.create_dataset(thetaeplus,'thetaeplus','Difference between unsaturated and saturated equivalent potential temperature','K'),
        calculator.create_dataset(bl,'bl','Average buoyancy in the lower troposphere','m/s²'),
        calculator.create_dataset(cape,'cape','Undilute buoyancy in the lower troposphere','K'),
        calculator.create_dataset(subsat,'subsat','Lower free-tropospheric subsaturation','K'),
        calculator.create_dataset(ps,'ps','Surface pressure','hPa'),
        calculator.create_dataset(lf,'lf','Land fraction','0-1'),
        calculator.create_dataset(pr,'pr','Precipitation rate','mm/hr'),
        calculator.create_dataset(dlev,'dlev','Vertical thickness weights','hPa'),
        calculator.create_dataset(surfmask,'surfmask','Binary surface validity mask','0-1')]
    logger.info('Saving datasets...')
    for ds in dslist:
        calculator.save(ds)
        del ds