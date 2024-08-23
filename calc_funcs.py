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