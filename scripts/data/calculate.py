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
    se  = calculator.retrieve('ERA5_surface_elevation')
    lhf = calculator.retrieve('ERA5_mean_surface_latent_heat_flux')
    shf = calculator.retrieve('ERA5_mean_surface_sensible_heat_flux')
    tp  = calculator.retrieve('ERA5_total_accumulated_precipitation')
    pr  = calculator.retrieve('IMERG_V06_precipitation_rate')
    logger.info('Regridding variables...')
    ps  = calculator.regrid(ps).load()
    t   = calculator.regrid(t).load()
    q   = calculator.regrid(q).load()
    lf  = calculator.regrid(lf).load()
    se  = calculator.regrid(se).load()
    if 'time' in se.dims:
        se = se.mean(dim='time')
    lhf = calculator.regrid(lhf).load()
    shf = calculator.regrid(shf).load()
    tp  = calculator.regrid(tp)
    pr  = calculator.regrid(pr)
    logger.info('Resampling variables to 3-hourly...')
    ps  = calculator.resample(ps,'first')
    t   = calculator.resample(t,'first')
    q   = calculator.resample(q,'first')
    lhf = calculator.resample(lhf,'mean')
    shf = calculator.resample(shf,'mean')
    tp  = calculator.resample(tp,'sum').where(lambda x:x>=1e-4,0.0).load()
    pr  = calculator.resample(pr,'mean').where(lambda x:x>=1e-6,0.0).load()
    logger.info('Calculating relative humidity and equivalent potential temperature terms...')
    p          = calculator.create_p_array(q)
    rh         = calculator.calc_rh(p,t,q)
    thetae     = calculator.calc_thetae(p,t,q)
    thetaestar = calculator.calc_thetae(p,t)
    logger.info('Calculating layer averages...')
    pbltop      = ps-100.0
    lfttop      = xr.full_like(ps,500.0)
    thetaeb     = calculator.calc_layer_average(thetae,ps,pbltop)
    thetael     = calculator.calc_layer_average(thetae,pbltop,lfttop)
    thetaelstar = calculator.calc_layer_average(thetaestar,pbltop,lfttop)
    logger.info('Calculating BL terms...')
    wb,wl          = calculator.calc_weights(ps,pbltop,lfttop)
    cape,subsat,bl = calculator.calc_bl_terms(thetaeb,thetael,thetaelstar,wb,wl)
    logger.info('Interpolating profile variables to sigma coordinates...')
    siglevels = np.arange(0.5,1.05,0.05)
    t          = calculator.interpolate_to_sigma(t,ps,siglevels)
    q          = calculator.interpolate_to_sigma(q,ps,siglevels)
    rh         = calculator.interpolate_to_sigma(rh,ps,siglevels)
    thetae     = calculator.interpolate_to_sigma(thetae,ps,siglevels)
    thetaestar = calculator.interpolate_to_sigma(thetaestar,ps,siglevels)
    logger.info('Calculating sigma quadrature weights...')
    dsig = calculator.calc_dsig(siglevels)
    logger.info('Creating datasets...')
    dslist = [
        calculator.create_dataset(rh,'rh','Relative humidity','%'),
        calculator.create_dataset(thetae,'thetae','Equivalent potential temperature','K'),
        calculator.create_dataset(thetaestar,'thetaestar','Saturated equivalent potential temperature','K'),
        calculator.create_dataset(bl,'bl','Average buoyancy in the lower troposphere','m/s²'),
        calculator.create_dataset(cape,'cape','Undilute buoyancy in the lower troposphere','K'),
        calculator.create_dataset(subsat,'subsat','Lower free-tropospheric subsaturation','K'),
        calculator.create_dataset(ps,'ps','Surface pressure','hPa'),
        calculator.create_dataset(lf,'lf','Land fraction','0-1'),
        calculator.create_dataset(se,'se','Surface elevation','m'),
        calculator.create_dataset(lhf,'lhf','Mean surface latent heat flux','W/m²'),
        calculator.create_dataset(shf,'shf','Mean surface sensible heat flux','W/m²'),
        calculator.create_dataset(pr,'pr','Precipitation rate','mm/hr'),
        calculator.create_dataset(tp,'tp','Total accumulated precipitation','mm'),
        calculator.create_dataset(dsig,'dsig','Sigma thickness weights','0-1')]
    logger.info('Saving datasets...')
    for ds in dslist:
        calculator.save(ds)
        del ds