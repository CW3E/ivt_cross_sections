#!/usr/bin/python3
"""
Filename:    calc_funcs.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: functions to calculate variables needed for the ivt cross sections
"""

import sys, os
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

def ivt(u_wind,v_wind,specific_humidity,pressure,sfc_pressure=None):
    '''

    Author: Brian Kawzenuk adapted by Ricardo Vilela
    Parameters: 
        u_wind: 3-d u component of wind with shape (pressure_levels, lon, lat) [m/s]
        v_wind: 3-d v component of wind with shape (pressure_levels, lon, lat) [m/s]
        specific_humidity: 3-d specific with shape humidity (pressure_levels, lon, lat) [kg/kg] 
        pressure: 3-d with shape (pressure_levels, lon, lat) [Pa]
        sfc_rpessure: 2-d with shape (lon,lat) [Pa]
    
    Returns:
        uIVT: 2-d u component of IVT with shape (lon, lat) [kg/(m*s)]
        vIVT: 2-d v component of IVT with shape (lon, lat) [kg/(m*s)]
        IVT: 3-d resultant IVT with shape (lon, lat) [kg/(m*s)]
    '''


    #masking u v and humidity in grids "underground" ie where sfc pressure is lower than pressure level
    if sfc_pressure is not None:
        sfc_mask = pressure < sfc_pressure[np.newaxis, :, :]    
        u_wind = np.ma.masked_array(u_wind, mask=~sfc_mask)
        v_wind = np.ma.masked_array(v_wind, mask=~sfc_mask)
        specific_humidity = np.ma.masked_array(specific_humidity, mask=~sfc_mask)
        
    #200hPa cap
    toa_cap_in_Pa = 20000
    toa_mask = pressure > toa_cap_in_Pa
    u_wind = np.ma.masked_array(u_wind, mask=~toa_mask)
    v_wind = np.ma.masked_array(v_wind, mask=~toa_mask)
    specific_humidity = np.ma.masked_array(specific_humidity, mask=~toa_mask)


    # enumerate through pressure levels so we select the layers
    m = -1
    uflux = np.empty(specific_humidity.shape)
    vflux = np.empty(specific_humidity.shape)

    while m < pressure.shape[0] - 2:
       m += 1
       layer_diff = np.abs(pressure[m,:,:] - pressure[m+1,:,:])
       ql = (specific_humidity[m, :, :] + specific_humidity[m+1, :, :]) / 2
       ul = (u_wind[m, :, :] + u_wind[m+1, :, :]) / 2
       vl = (v_wind[m, :, :] + v_wind[m+1, :, :]) / 2
       qfl = (ql / 9.8) * layer_diff
       uflux[m, :, :] = qfl * ul
       vflux[m, :, :] = qfl * vl

    uIVT = np.sum(uflux, axis=0)
    vIVT = np.sum(vflux, axis=0)
    IVT = np.sqrt(uIVT**2 + vIVT**2)


    return uIVT, vIVT, IVT



def specific_humidity(temperature, pressure, relative_humidity):
    """
    Author: Brian Kawzenuk

    Calculate specific humidity.
    
    Parameters:
    - temperature: temperature in Kelvin
    - pressure: pressure in Pa
    - relative_humidity: relative humidity as a fraction (e.g., 0.5 for 50%)
    
    Returns:
    - specific humidity
    """

    # Constants
    epsilon = 0.622  # Ratio of molecular weight of water vapor to dry air

    # Calculate saturation vapor pressure
    e_s = 611.2 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))

    # Calculate vapor pressure
    e = relative_humidity * e_s
    # Calculate specific humidity
    q = epsilon * e / (pressure - e)
    
    return q


def calculate_zero_degree_isotherm(height):
    ## get freezing level on pressure levels
    hgts = height.values * units(height.units)
        
    # calculating freezing level on pressure
    z0_pres = mpcalc.height_to_pressure_std(hgts)
    return z0_pres.magnitude

def calculate_air_density(pressure, temperature, relative_humidity):
    ## add metpy units to each var
    pres = pressure * units.hPa
    temp = temperature.values * units(temperature.units)
    rh = relative_humidity.values * units(relative_humidity.units)
    
    ## calculate mixing ratio from relative humidity
    mr = mpcalc.mixing_ratio_from_relative_humidity(pres, temp, rh)
    
    # calculate density with pressure, temperature, and mixing ratio
    den = mpcalc.density(pres, temp, mr)
    
    return den.magnitude

def calculate_wvflux(uwind, vwind, density, specific_humidity):

    UVtmp = np.sqrt((uwind**2)+(vwind**2))
    wvflux = density*UVtmp*specific_humidity
    
    return wvflux

def calc_relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity):
    
    pres = pressure * units.hPa
    temp = temperature.values * units(temperature.units)
    q = specific_humidity.values * units(specific_humidity.units)
    
    rh = mpcalc.relative_humidity_from_specific_humidity(pres, temp, q)
    print(rh.units)
    ## put into a xarray dataarray
    rh = xr.DataArray(rh.magnitude, name="rh", 
                             dims=("hybrid", "latitude","longitude"), 
                             coords={"hybrid": temperature.hybrid.values, 
                                     "latitude": temperature.latitude.values, 
                                     "longitude": temperature.longitude.values},
                             attrs={'units': 'percent'})
    # rh = rh.assign_attrs(units=rh.units)
    return rh