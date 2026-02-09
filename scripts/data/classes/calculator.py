#!/usr/bin/env python

import os
import xesmf
import logging
import numpy as np
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
        self.author   = author
        self.email    = email
        self.filedir  = filedir
        self.savedir  = savedir
        self.latrange = latrange
        self.lonrange = lonrange

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
        p = refda.lev.expand_dims({'lat':refda.lat,'lon':refda.lon,'time':refda.time}).transpose('lat','lon','lev','time')
        return p

    def resample(self,da):
        '''
        Purpose: Compute a centered hourly mean (uses the two half-hour samples that straddle each hour; 
        falls back to one at boundaries).
        Args:
        - da (xr.DataArray): input DataArray
        Returns:
        - xr.DataArray: DataArray resampled at on-the-hour timestamps
        '''
        da = da.rolling(time=2,center=True,min_periods=1).mean()
        da = da.sel(time=da.time.dt.minute==0)
        return da

    def regrid(self,da):
        '''
        Purpose: Regrid a DataArray to 1.0° × 1.0° grid over the target domain.
        Args:
        - da (xr.DataArray): input DataArray with radius (for better interpolation)
        Returns:
        - xr.DataArray: regridded DataArray
        '''
        targetlats = np.arange(self.latrange[0],self.latrange[1]+1.0,1.0)
        targetlons = np.arange(self.lonrange[0],self.lonrange[1]+1.0,1.0)
        targetgrid = xr.Dataset({'lat':(['lat'],targetlats),'lon':(['lon'],targetlons)})
        regridder  = xesmf.Regridder(da,targetgrid,method='conservative')
        da = regridder(da,keep_attrs=True)
        return da

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

    def calc_thetae(self,p,t,q=None,ps=None):
        '''
        Purpose: Calculate (unsaturated, saturated, or surface) equivalent potential temperature (θₑ) using Eqs. 43 and 55 
        from Bolton D. (1980), Mon. Wea. Rev.     
        Args:
        - p (xr.DataArray): pressure DataArray (hPa)
        - t (xr.DataArray): temperature DataArray (K)
        - q (xr.DataArray, optional): specific humidity DataArray (kg/kg); if None, saturated θₑ computed
        - ps (xr.DataArray, optional): surface pressure DataArray (hPa); if given, θₑ at the surface will be calculated
          (ps > 1,000 hPa are clamped to 1,000 hPa to prevent extrapolation beyond the available pressure levels from ERA5)
        Returns:
        - xr.DataArray: unsaturated, saturated, and/or surface) θₑ DataArray (K)
        '''
        if q is None:
            q = self.calc_qs(p,t)
        if ps is not None:
            psclamped = xr.where(ps>1000.,1000.,ps)
            t = t.interp(lev=psclamped)
            q = q.interp(lev=psclamped)
            p = psclamped
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
    
    def calc_bl_terms(self,thetaeb,thetael,thetaelsat,wb,wl):
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
        cape    = ((thetaeb-thetaelsat)/thetaelsat)*thetae0
        subsat  = ((thetaelsat-thetael)/thetaelsat)*thetae0
        bl      = (g/(kappal*thetae0))*((wb*cape)-(wl*subsat))
        return cape,subsat,bl

    def calc_dlev(self,refda):
        '''
        Purpose: Compute quadrature weights for numerical integration over a the 'lev' dimension;
        weights Δp (hPa) represent vertical thickness. Spacing between adjacent grid points is estimated 
        using centered finite differences.
        Args:
        - refda (xr.DataArray): reference DataArray with'lev' dimension
        Returns:
        - xr.DataArray: quadrature weights for Δp (hPa)
        '''
        levs  = refda.lev.values
        dlevvalues = np.abs(np.concatenate([[levs[1]-levs[0]],0.5*(levs[2:]-levs[:-2]),[levs[-1]-levs[-2]]]))
        dlev  = xr.DataArray(dlevvalues.astype(np.float32),dims=('lev',),coords={'lev':refda.lev})
        return dlev

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
        dims = [dim for dim in ('lat','lon','lev','time') if dim in da.dims]
        da = da.transpose(*dims)
        ds = da.to_dataset(name=shortname)
        ds[shortname].attrs = dict(long_name=longname,units=units)
        if 'lat' in ds.coords:
            ds.lat.attrs  = dict(long_name='Latitude',units='°N')
        if 'lon' in ds.coords:
            ds.lon.attrs  = dict(long_name='Longitude',units='°E')
        if 'lev' in ds.coords:
            ds.lev.attrs  = dict(long_name='Pressure level',units='hPa')
        if 'time' in ds.coords:
            ds.time.attrs = dict(long_name='Time')
        ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {self.author} ({self.email})')
        logger.info(f'   {longname} size: {ds.nbytes*1e-9:.3f} GB')
        return ds

    def save(self,ds,timechunksize=2208):
        '''
        Purpose: Save an xr.Dataset to NetCDF and verify by reopening.
        Args:
        - ds (xr.Dataset): Dataset to save
        - timechunksize (int): chunk size for time dimension (defaults to 2,208 for 3-month chunks)
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