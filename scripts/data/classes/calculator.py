#!/usr/bin/env python

import os
import xesmf
import logging
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

logger = logging.getLogger(__name__)

class DataCalculator:

    def __init__(self,author,email,filedir,savedir,latrange,lonrange):
        '''
        Purpose: Initialize DataCalculator with configuration parameters.
        Args:
        - author (str): author name
        - email (str): author email
        - filedir (str): directory containing input NetCDF files
        - savedir (str): directory to save output files
        - latrange (tuple[float,float]): latitude range
        - lonrange (tuple[float,float]): longitude range
        '''
        self.author     = author
        self.email      = email
        self.filedir    = filedir
        self.savedir    = savedir
        self.latrange   = latrange
        self.lonrange   = lonrange
        self.regridders = {}

    def retrieve(self,longname):
        '''
        Purpose: Lazily import in a NetCDF file as an xr.DataArray with ascending pressure levels, if applicable
        (e.g., [500,550,600,...] hPa).
        Args:
        - longname (str): variable description
        Returns:
        - xr.DataArray: DataArray with levels ordered (if applicable)
        '''
        filename = f'{longname}.nc'
        filepath = os.path.join(self.filedir,filename)
        da = xr.open_dataarray(filepath,engine='h5netcdf')
        if 'lev' in da.dims:
            if not np.all(np.diff(da.lev.values)>0):
                da = da.sortby('lev')
                logger.info(f'   Levels for {filename} were reordered to ascending')
        return da

    def create_p_array(self,refda):
        '''
        Purpose: Create a pressure xr.DataArray from the 'lev' dimension.
        Args:
        - refda (xr.DataArray): reference DataArray containing 'lev'
        Returns:
        - xr.DataArray: pressure DataArray
        '''
        p = refda.lev.expand_dims({'lat':refda.lat,'lon':refda.lon,'time':refda.time})
        return p

    def regrid(self,da):
        '''
        Purpose: Regrid an xr.DataArray to 1.0° × 1.0° grid over the target domain.
        Args:
        - da (xr.DataArray): input DataArray with radius (for better interpolation)
        Returns:
        - xr.DataArray: regridded DataArray
        '''
        targetlats = np.arange(self.latrange[0],self.latrange[1]+1.0,1.0)
        targetlons = np.arange(self.lonrange[0],self.lonrange[1]+1.0,1.0)
        key = (da.lat.values.tobytes(),da.lon.values.tobytes())
        if key not in self.regridders:
            targetgrid = xr.Dataset({'lat':(['lat'],targetlats),'lon':(['lon'],targetlons)})
            method     = 'conservative' if 'lf' in da.name else 'bilinear'
            self.regridders[key] = xesmf.Regridder(da,targetgrid,method=method)
        da = self.regridders[key](da,keep_attrs=True)
        return da

    def resample(self,da,method):
        '''
        Purpose: Coarsen an xr.DataArray to 3-hourly.
        Args:
        - da (xr.DataArray): input DataArray
        - method (str): 'first' (for instantaneous variables) | 'mean' (for rates/fluxes) | 'sum' (for accumulations)
        Returns:
        - xr.DataArray: 3-hourly DataArray
        '''
        if method=='first':
            return da.sel(time=da.time.dt.hour%3==0)
        windows = pd.DatetimeIndex(da.time.values).floor('3h')
        da = da.assign_coords(window=('time',windows))
        result = da.groupby('window').mean() if method=='mean' else da.groupby('window').sum()
        return result.rename({'window':'time'})

    def calc_es(self,t):
        '''
        Purpose: Calculate saturation vapor pressure (eₛ) using Eqs. 17 and 18 from Huang J. (2018), J. Appl.
        Meteorol. Climatol.
        Args:
        - t (xr.DataArray): temperature DataArray (K)
        Returns:
        - xr.DataArray: eₛ DataArray (hPa)
        '''
        tc = t-273.15
        eswat = np.exp(34.494-(4924.99/(tc+237.1)))/((tc+105.0)**1.57)
        esice = np.exp(43.494-(6545.8/(tc+278.0)))/((tc+868.0)**2.0)
        es = xr.where(tc>0.0,eswat,esice)/100.0
        return es

    def calc_qs(self,p,t):
        '''
        Purpose: Calculate saturation specific humidity (qₛ) using Eq. 4 from Miller SFK. (2018), Atmos.
        Humidity Eq. Plymouth State Wea. Ctr.
        Args:
        - p (xr.DataArray): pressure DataArray (hPa)
        - t (xr.DataArray): temperature DataArray (K)
        Returns:
        - xr.DataArray: qₛ DataArray (kg/kg)
        '''
        rv = 461.50
        rd = 287.04
        epsilon = rd/rv
        es = self.calc_es(t)
        qs = (epsilon*es)/(p-es*(1.0-epsilon))
        return qs

    def calc_rh(self,p,t,q):
        '''
        Purpose: Calculate relative humidity (RH) using qₛ.
        Args:
        - p (xr.DataArray): pressure DataArray (hPa)
        - t (xr.DataArray): temperature DataArray (K)
        - q (xr.DataArray): specific humidity DataArray (kg/kg)
        Returns:
        - xr.DataArray: RH DataArray (%)
        '''
        qs = self.calc_qs(p,t)
        rh = (q/qs)*100.0
        rh = rh.clip(min=0.0,max=100.0)
        return rh

    def calc_thetae(self,p,t,q=None):
        '''
        Purpose: Calculate (unsaturated or saturated) equivalent potential temperature (θₑ) using Eqs. 43 and 55
        from Bolton D. (1980), Mon. Wea. Rev.
        Args:
        - p (xr.DataArray): pressure DataArray (hPa)
        - t (xr.DataArray): temperature DataArray (K)
        - q (xr.DataArray, optional): specific humidity DataArray (kg/kg); if None, saturated θₑ computed
        Returns:
        - xr.DataArray: unsaturated or saturated θₑ DataArray (K)
        '''
        if q is None:
            q = self.calc_qs(p,t)
        p0 = 1000.0
        rv = 461.5
        rd = 287.04
        epsilon = rd/rv
        r  = q/(1.0-q)
        e  = (p*r)/(epsilon+r)
        tl = 2840.0/(3.5*np.log(t)-np.log(e)-4.805)+55.0
        thetae = t*(p0/p)**(0.2854*(1.0-0.28*r))*np.exp((3.376/tl-0.00254)*1000.0*r*(1.0+0.81*r))
        return thetae

    def get_level_above(self,ptarget,levels,side):
        '''
        Purpose: Find the pressure level immediately above a target pressure, i.e., the next smallest
        pressure (higher altitude).
        Args:
        - ptarget (xr.DataArray or np.ndarray): target pressures
        - levels (np.ndarray): 1D array of ascending pressure levels (e.g., [500,550,600,...] hPa)
        - side (str): 'left' or 'right' tie-breaking for np.searchsorted
        Returns:
        - np.ndarray: array of pressure levels immediately above each target (same shape as 'ptarget')
        '''
        searchidx = np.searchsorted(levels,ptarget,side=side)
        levabove  = levels[np.maximum(searchidx-1,0)]
        return levabove

    def get_level_below(self,ptarget,levels,side):
        '''
        Purpose: Find the pressure level immediately below a target pressure, i.e., the next largest
        pressure (lower altitude).
        Args:
        - ptarget (xr.DataArray or np.ndarray): target pressures
        - levels (np.ndarray): 1D array of ascending pressure levels (e.g., [500,550,600,...] hPa)
        - side (str): 'left' or 'right' tie-breaking for np.searchsorted
        Returns:
        - np.ndarray: array of pressure levels immediately below each target (same shape as 'ptarget')
        '''
        searchidx = np.searchsorted(levels,ptarget,side=side)
        levbelow  = levels[np.minimum(searchidx,len(levels)-1)]
        return levbelow

    def calc_layer_average(self,da,a,b):
        '''
        Purpose: Calculate the pressure-weighted mean of an xr.DataArray between two pressure levels 'a' (bottom of layer) and
        'b' (top of layer), with `a > b`.
        Args:
        - da (xr.DataArray): input DataArray with 'lev' dimension
        - a (xr.DataArray): DataArray of bottom boundary pressures (higher values, lower altitude)
        - b (xr.DataArray): DataArray of top boundary pressures (lower values, higher altitude)
        Returns:
        - xr.DataArray: layer-averaged DataArray
        '''
        da = da.load()
        a  = a.load()
        b  = b.load()
        levabove = xr.apply_ufunc(self.get_level_above,a,kwargs={'levels':np.array(da.lev),'side':'right'})
        levbelow = xr.apply_ufunc(self.get_level_below,a,kwargs={'levels':np.array(da.lev),'side':'right'})
        valueabove = da.sel(lev=levabove)
        valuebelow = da.sel(lev=levbelow)
        correction = -valueabove/2*(levbelow-levabove)*(a<da.lev[-1])
        levbelow   = levbelow+(levbelow==levabove)
        lowerintegral = (a-levabove)*valueabove+(valuebelow-valueabove)*(a-levabove)**2/(levbelow-levabove)/2+correction
        lowerintegral = lowerintegral.fillna(0)
        innerintegral = (da*(da.lev<=a)*(da.lev>=b)).fillna(0).integrate('lev')
        levabove = xr.apply_ufunc(self.get_level_above,b,kwargs={'levels':np.array(da.lev),'side':'left'})
        levbelow = xr.apply_ufunc(self.get_level_below,b,kwargs={'levels':np.array(da.lev),'side':'left'})
        valueabove = da.sel(lev=levabove)
        valuebelow = da.sel(lev=levbelow)
        correction = -valuebelow/2*(levbelow-levabove)*(b>da.lev[0])
        levabove   = levabove-(levbelow==levabove)
        upperintegral = (levbelow-b)*valueabove+(valuebelow-valueabove)*(levbelow-levabove)*(
            1-((b-levabove)/(levbelow-levabove))**2)/2+correction
        upperintegral = upperintegral.fillna(0)
        layeraverage  = (lowerintegral+innerintegral+upperintegral)/(a-b)
        return layeraverage

    def calc_weights(self,ps,pbltop,lfttop):
        '''
        Purpose: Calculate weights for the boundary layer (PBL) and lower free troposphere (LFT) using Eqs. 5a and 5b from Adames AF,
        Ahmed F, and Neelin JD. 2021. J. Atmos. Sci.
        Args:
        - ps (xr.DataArray): surface pressure DataArray (hPa)
        - pbltop (xr.DataArray): DataArray of pressures at the top of the PBL (hPa)
        - lfttop (xr.DataArray): DataArray of pressures at the top of the LFT (hPa)
        Returns:
        - tuple[xr.DataArray,xr.DataArray]: PBL and LFT weights
        '''
        pblthickness = ps-pbltop
        lftthickness = pbltop-lfttop
        wb = (pblthickness/lftthickness)*np.log((pblthickness+lftthickness)/pblthickness)
        wl = 1.0-wb
        return wb,wl

    def calc_bl_terms(self,thetaeb,thetael,thetaelstar,wb,wl):
        '''
        Purpose: Calculate CAPEL, SUBSATL, and BL following Eq. 1 from Ahmed F and Neelin JD. 2021. Geophys. Res. Lett.
        Args:
        - thetaeb (xr.DataArray): DataArray of θₑ averaged over the PBL (K)
        - thetael (xr.DataArray): DataArray of θₑ averaged over the LFT (K)
        - thetaelstar (xr.DataArray): DataArray of saturated θₑ averaged over the LFT (K)
        - wb (xr.DataArray): DataArray of PBL weights
        - wl (xr.DataArray): DataArray of LFT weights
        Returns:
        - tuple[xr.DataArray,xr.DataArray,xr.DataArray]: CAPEL, SUBSATL, and BL DataArrays
        '''
        g       = 9.81
        kappal  = 3.0
        thetae0 = 340.0
        cape    = ((thetaeb-thetaelstar)/thetaelstar)*thetae0
        subsat  = ((thetaelstar-thetael)/thetaelstar)*thetae0
        bl      = (g/(kappal*thetae0))*((wb*cape)-(wl*subsat))
        return cape,subsat,bl

    def calc_dsig(self,sigs):
        '''
        Purpose: Compute quadrature weights for numerical integration over the 'sig' dimension; weights Δσ represent
        sigma thickness. Spacing between adjacent grid points is estimated using centered finite differences.
        Args:
        - sigs (np.ndarray): 1D array of ascending sigma levels (e.g., [0.5, 0.55, ..., 1.0])
        Returns:
        - xr.DataArray: quadrature weights for Δσ
        '''
        sigs   = np.asarray(sigs,dtype=np.float32)
        values = np.abs(np.concatenate([[sigs[1]-sigs[0]],0.5*(sigs[2:]-sigs[:-2]),[sigs[-1]-sigs[-2]]]))
        dsig   = xr.DataArray(values,dims=('sig',),coords={'sig':sigs})
        return dsig

    def interpolate_to_sigma(self,da,ps,sigs):
        '''
        Purpose: Interpolate an xr.DataArray from pressure levels onto a uniform sigma (σ = p/pₛ) grid
        via piecewise-linear interpolation in pressure space. Columns with invalid surface pressures
        are masked.
        Args:
        - da (xr.DataArray): input DataArray containing 'lev'
        - ps (xr.DataArray): surface pressure DataArray (hPa)
        - sigs (np.ndarray): 1D array of ascending sigma levels (e.g., [0.5, 0.55, ..., 1.0])
        Returns:
        - xr.DataArray: interpolated DataArray with 'sig'
        '''
        da   = da.transpose('lat','lon','lev','time').load()
        ps   = ps.transpose('lat','lon','time').load()
        levs = da.lev.values
        ptarget  = sigs[None,None,:,None]*ps.values[:,:,None,:]
        upper    = np.clip(np.searchsorted(levs,ptarget,side='right'),1,len(levs)-1)
        lower    = upper-1
        weight   = (ptarget-levs[lower])/(levs[upper]-levs[lower])
        result   = (1.0-weight)*np.take_along_axis(da.values,lower,axis=2)+weight*np.take_along_axis(da.values,upper,axis=2)
        result   = np.where(np.isfinite(ptarget)&(ptarget>0),result,np.nan).astype(np.float32)
        interped = xr.DataArray(result,dims=('lat','lon','sig','time'),
                                coords={'lat':da.lat,'lon':da.lon,'sig':sigs,'time':da.time},
                                name=da.name,attrs=da.attrs)
        return interped

    def create_dataset(self,da,shortname,longname,units):
        '''
        Purpose: Wrap an xr.DataArray into a xr.Dataset with metadata.
        Args:
        - da (xr.DataArray): input DataArray
        - shortname (str): variable name
        - longname (str): variable description
        - units (str): variable units
        Returns:
        - xr.Dataset: Dataset containing the variable and metadata
        '''
        dims = [dim for dim in ('lat','lon','sig','time') if dim in da.dims]
        da = da.transpose(*dims)
        ds = da.to_dataset(name=shortname)
        ds[shortname].attrs = dict(long_name=longname,units=units)
        if 'lat' in ds.coords:
            ds.lat.attrs  = dict(long_name='Latitude',units='°N')
        if 'lon' in ds.coords:
            ds.lon.attrs  = dict(long_name='Longitude',units='°E')
        if 'sig' in ds.coords:
            ds.sig.attrs  = dict(long_name='Sigma level',units='0-1')
        if 'time' in ds.coords:
            ds.time.attrs = dict(long_name='Time')
        ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {self.author} ({self.email})')
        logger.info(f'   {longname} size: {ds.nbytes*1e-9:.6f} GB')
        return ds

    def save(self,ds,timechunksize=736):
        '''
        Purpose: Save an xr.Dataset to NetCDF and verify by reopening.
        Args:
        - ds (xr.Dataset): Dataset to save
        - timechunksize (int): chunk size for time dimension (defaults to 736 for 3-month chunks on 3-hourly data)
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(self.savedir,exist_ok=True)
        shortname = list(ds.data_vars)[0]
        filename  = f'{shortname}.nc'
        filepath  = os.path.join(self.savedir,filename)
        logger.info(f'   Attempting to save {filename}...')
        ds.load()
        ds[shortname].encoding = {}
        chunks = []
        for dim,size in zip(ds[shortname].dims,ds[shortname].shape):
            chunks.append(min(timechunksize,size) if dim=='time' else size)
        encoding = {shortname:{'chunksizes':tuple(chunks)}}
        try:
            ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
            xr.open_dataset(filepath,engine='h5netcdf').close()
            logger.info('      File write successful')
            return True
        except Exception:
            logger.exception('      Failed to save or verify')
            return False