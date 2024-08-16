#!/usr/bin/python3
"""
Filename:    read_deterministic_data.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: functions to read deterministic data from GFS, ECMWF, and WWRF
"""

import sys
import os
import glob
import shutil
import subprocess
import re
import xarray as xr
import numpy as np
import pandas as pd
import datetime
import metpy.calc as mpcalc
from metpy.units import units
import cartopy.crs as ccrs

sys.path.append('../file-reader-toolkit') # point this to where you have the CW3E cmaps_toolkit repo located
import calculations

def zero_degree_isotherm(height):
    # calculating freezing level on pressure
    z0_pres = mpcalc.height_to_pressure_std(height)
    return z0_pres

def read_gfs_deterministic(filename, vardict, show_catalog=False):

    '''
    author: Ricardo Vilela
    email: rbatistavilela@ucsd.edu

    function usage:

    filename example:
    gfs_2024041512_f003.grb

    vardict example:
    vardict = {
            "u_wind":{"typeOfLevel":'isobaricInhPa',"shortName":"u"}, #U-component of wind
            "v_wind":{"typeOfLevel":'isobaricInhPa',"shortName":"v"}, #V-component of wind
            "iwv":{"typeOfLevel":'atmosphereSingleLayer',"shortName":"pwat"}, #Integrated precipitable water or IWV
            "temperature":{"typeOfLevel":'isobaricInhPa',"shortName":"t"}, #Temperature
            "rh":{"typeOfLevel":'isobaricInhPa',"shortName":"r"}, #Relative Humidity
            "sfc_pressure":{'name': 'Surface pressure', 'typeOfLevel': 'surface', 'level': 0, 'paramId': 134, 'shortName': 'sp'}, #surface pressure
            "sea_level_pressure":{'name': 'Pressure reduced to MSL', 'typeOfLevel': 'meanSea', 'level': 0, 'paramId': 260074, 'shortName': 'prmsl'} #mean sea level pressure
        }
    
    Output:
    Dictionary of objects for each variable set in the vardict argument. Ex.
    uwind = selected_vars["u_wind"].values
    uwind_latitude = selected_vars["u_wind"].latitude.values
    uwind_longitude = selected_vars["u_wind"].longitude.values

    for 3d variables:
    uwind_pressure_levels = selected_vars["u_wind"].isobaricInhPa.values

    '''
    selected_vars = {}
    #iterating over all variables in the vardict and storing each one in a new dictionary called selected_vars
    for var in vardict.keys():

        ds = xr.open_dataset(filename,
                        engine='cfgrib',filter_by_keys=vardict[var])
        
        
        #exceptions when shortName does not match variable name in the grib
        if var == 'u_wind_10m':
            selected_vars[var] = ds["u10"]
        elif var == 'v_wind_10m':
            selected_vars[var] = ds["v10"]
        else:    
            selected_vars[var] = ds[vardict[var]["shortName"]]    

    return selected_vars

def read_ecmwf_S2D(filename, vardict, show_catalog=False):
    
    '''
    author: Ricardo Vilela
    email: rbatistavilela@ucsd.edu

    function usage:
    
    filename example:
    S2D04151200041515001.grb

    vardict example:

    vardict = {
               
                "u_wind":{"shortName":'u'}, #U-component of wind
                "v_wind":{"shortName":'v'}, #V-component of wind
                "temperature":{"shortName":'t'}, #Temperature
                "specific_humidity":{"shortName":'q'}, #Specific humidity
                
                
                }
    Output:
    Dictionary of objects for each variable set in the vardict argument. Ex.
    uwind = selected_vars["u_wind"].values
    uwind_latitude = selected_vars["u_wind"].latitude.values
    uwind_longitude = selected_vars["u_wind"].longitude.values

    These files are in model levels instead of pressure levels, in other words, pressure is not a dimension but an extra variable built based on a lookup table of coefficients and the hybrid dimension.

    retrieving the pressure array
    pressure_array = selected_vars["pressure"].values


    '''
    
    print('[INFO] reading ECMWF file: '+filename)    


    selected_vars = {}
    
    # reads surface pressure first (this field is needed in order to calculate the pressure levels for each grid)
    sfc_pressure_ds = xr.open_dataset(filename,
                        engine='cfgrib',filter_by_keys={"shortName":'lnsp'})
    

    
    sfc_pressure = 2.71828**sfc_pressure_ds.lnsp.values

    #read the coefficient lookup table
    coeff = pd.read_csv("utils/ecmwf_coeffs.txt",names=["A","B"],sep=" ")
    coeffA = np.array(coeff["A"])[:, np.newaxis, np.newaxis]
    coeffB = np.array(coeff["B"])[:, np.newaxis, np.newaxis]

    #creating the pressure based on the coefficient table and the surface pressure
    pressure = (coeffA + (sfc_pressure * coeffB))/100

    #iterating over all variables in the vardict and storing each one in a new dictionary called selected_vars
    for var in vardict.keys():
    
        ds = xr.open_dataset(filename,
                        engine='cfgrib',filter_by_keys=vardict[var])

        selected_vars[var] = ds[vardict[var]["shortName"]]


        
    ngrids = pressure.shape[0]*pressure.shape[1]*pressure.shape[2]
    
    #creating pressure 3d data array and its attributes
    pressure = xr.DataArray(pressure, name="pressure", dims=("hybrid", "latitude","longitude"), coords={"hybrid": coeff.index, "latitude":sfc_pressure_ds.latitude.values, "longitude":sfc_pressure_ds.longitude.values, "time":sfc_pressure_ds.time.values, "valid_time":sfc_pressure_ds.valid_time.values})
    pressure.attrs = {'GRIB_paramId': 0, 'GRIB_dataType': 'fc', 'GRIB_numberOfPoints': ngrids, 'GRIB_typeOfLevel': 'hybrid', 'GRIB_stepUnits': 1, 'GRIB_stepType': 'instant', 'GRIB_gridType': 'regular_ll', 'GRIB_NV': 276, 'GRIB_Nx': len(pressure.longitude.values), 'GRIB_Ny': len(pressure.latitude.values), 'GRIB_Nz': len(pressure.hybrid.values), 'GRIB_cfName': 'unknown', 'GRIB_cfVarName': 'pressure', 'GRIB_gridDefinitionDescription': 'Latitude/longitude', 'GRIB_iDirectionIncrementInDegrees': 0.1, 'GRIB_iScansNegatively': 0, 'GRIB_jDirectionIncrementInDegrees': 0.1, 'GRIB_jPointsAreConsecutive': 0, 'GRIB_jScansPositively': 0, 'GRIB_latitudeOfFirstGridPointInDegrees': np.nanmax(pressure.latitude.values), 'GRIB_latitudeOfLastGridPointInDegrees': np.nanmin(pressure.latitude.values), 'GRIB_longitudeOfFirstGridPointInDegrees': np.nanmin(pressure.longitude.values), 'GRIB_longitudeOfLastGridPointInDegrees': np.nanmax(pressure.longitude.values), 'GRIB_missingValue': 3.4028234663852886e+38, 'GRIB_name': 'Grid wise atmospheric pressure', 'GRIB_shortName': 'pressure', 'GRIB_units': 'Numeric', 'long_name': 'Atmospheric Pressure', 'units': 'hPa', 'standard_name': 'unknown'}
    
    
    #adding pressure variable to the object dictionary
    selected_vars["pressure"] = pressure
    

    #standardize latitude longitude time and level dimension name and position in the array

    return selected_vars


def read_ecmwf_S1D(filename, vardict, show_catalog=False):
    
    '''
    author: Ricardo Vilela
    email: rbatistavilela@ucsd.edu

    function usage:
    
    filename example:
    S1D04151200041515001

    vardict example:

    vardict = {
               
               "msl_pressure":{"shortName":'msl'}, #msl pressure
                
                
                }
    Output:
    Dictionary of objects for each variable set in the vardict argument. Ex.
    mslp = selected_vars["mslp"].values


    '''
    
    print('[INFO] reading ECMWF file: '+filename)
        
    selected_vars = {}

    #iterating over all variables in the vardict and storing each one in a new dictionary called selected_vars
    for var in vardict.keys():
    
        ds = xr.open_dataset(filename,
                        engine='cfgrib',filter_by_keys=vardict[var])

        if var == 'u_wind_10m':
            selected_vars[var] = ds["u10"]
        elif var == 'v_wind_10m':
            selected_vars[var] = ds["v10"]
        else:    
            selected_vars[var] = ds[vardict[var]["shortName"]]   
        
    return selected_vars

class load_GFS_datasets:
    '''
    Loads variables needed for ivt cross section plots from GFS grb files
    
    Parameters
    ----------
    F : int
        the forecast lead requested
        
    fdate : str
        string of date for the filename in YYYYMMDDHH format
  
    Returns
    -------
    xarray : 
        xarray dataset object with variables
    
    '''
    def __init__(self, F, fdate=None):
        self.F = F
        ## today's year
        year = pd.Timestamp.today().year
        path_to_data = '/data/downloaded/Forecasts/GFS_025d/{0}/*'.format(year)
        if fdate is None:
            list_of_files = glob.glob(path_to_data)
            self.fpath = max(list_of_files, key=os.path.getctime)
            regex = re.compile(r'\d+')
            self.date_string = regex.findall(self.fpath)[-1]
        elif fdate is not None:
            self.date_string = fdate
        fname = '/gfs_{0}_f{1}.grb'.format(self.date_string, str(self.F).zfill(3))

        ## for now: copy the files to local space
        repo_path = '/home/dnash/comet_data/tmp'
        # shutil.copy(self.fpath+fname, repo_path+fname) # copy file over to data folder     
        self.fname = repo_path+fname

        self.model_init_date = datetime.datetime.strptime(self.date_string, '%Y%m%d%H')

    def calc_vars(self):
        ## dictionary of variables we need for the cross section
        gfs_vardict = {
        "u_wind":{"typeOfLevel":'isobaricInhPa',"shortName":"u"}, #U-component of wind
        "v_wind":{"typeOfLevel":'isobaricInhPa',"shortName":"v"}, #V-component of wind
        "iwv":{"typeOfLevel":'atmosphereSingleLayer',"shortName":"pwat"}, #Integrated precipitable water or IWV
        "temperature":{"typeOfLevel":'isobaricInhPa',"shortName":"t"}, #Temperature
        "rh":{"typeOfLevel":'isobaricInhPa',"shortName":"r"}, #Relative Humidity
        "sfc_pressure":{'name': 'Surface pressure', 'typeOfLevel': 'surface', 'level': 0, 'paramId': 134, 'shortName': 'sp'}, #surface pressure
        "freezing_level": {'typeOfLevel': 'isothermZero', 'shortName': 'gh'} ## freezing level 
}
        
        #gfs is a dictionary of datasets
        gfs = read_gfs_deterministic(filename=self.fname,vardict=gfs_vardict, show_catalog=False)
        
        ## get freezing level on pressure levels
        hgts = gfs['freezing_level'].values * units(gfs['freezing_level'].units)
        freezing_level = zero_degree_isotherm(hgts).magnitude
        
        #### Calculating IVT #####
        #extending pressure vector to 3d array to match rh shape
        pressure_3d = np.tile(gfs["rh"].isobaricInhPa.values[:, np.newaxis, np.newaxis], (1, gfs["rh"].values.shape[1], gfs["rh"].values.shape[2]))

        # calculating specific humidity from relative humidity for gfs
        gfs_q = calculations.specific_humidity(temperature=gfs["temperature"].values, pressure=pressure_3d*100, relative_humidity=gfs["rh"].values/100)

        #calculating ivt 
        gfs_uivt, gfs_vivt, gfs_ivt = calculations.ivt(u_wind=gfs["u_wind"].values,v_wind=gfs["v_wind"].values,specific_humidity=gfs_q, pressure=pressure_3d*100, sfc_pressure=gfs["sfc_pressure"].values)

        ## calculating wvflux
        wv_flux = np.sqrt((gfs["u_wind"].values*gfs_q)**2 + (gfs["v_wind"].values*gfs_q)**2)
        
        
        # BUILD DATASET
        ## 3D vars
        u_wind_da = xr.DataArray(gfs["u_wind"].values, name="u", dims=("pressure", "latitude","longitude"), coords={"pressure": gfs["u_wind"].isobaricInhPa.values, "latitude": gfs["u_wind"].latitude.values, "longitude":gfs["u_wind"].longitude.values})
        v_wind_da = xr.DataArray(gfs["v_wind"].values, name="v", dims=("pressure", "latitude","longitude"), coords={"pressure": gfs["v_wind"].isobaricInhPa.values, "latitude": gfs["v_wind"].latitude.values, "longitude":gfs["v_wind"].longitude.values})
        wvflux_da = xr.DataArray(wv_flux, name="wvflux", dims=("pressure", "latitude","longitude"), coords={"pressure": gfs["v_wind"].isobaricInhPa.values, "latitude": gfs["v_wind"].latitude.values, "longitude":gfs["v_wind"].longitude.values})


        ## 2D vars
        iwv_da = xr.DataArray(gfs["iwv"].values, name="iwv", dims=("latitude","longitude"), coords={"latitude": gfs["iwv"].latitude.values, "longitude":gfs["iwv"].longitude.values})
        ivt_da = xr.DataArray(gfs_ivt, name="ivt", dims=("latitude","longitude"), coords={"latitude": gfs["rh"].latitude.values, "longitude":gfs["rh"].longitude.values})
        sfc_pressure_da = xr.DataArray(gfs["sfc_pressure"].values, name="sfc_pressure", dims=("latitude","longitude"), coords={"latitude": gfs["sfc_pressure"].latitude.values, "longitude":gfs["sfc_pressure"].longitude.values})
        freezing_level_da = xr.DataArray(freezing_level, name="freezing_level", dims=("latitude","longitude"), coords={"latitude": gfs['freezing_level'].latitude.values, "longitude":gfs['freezing_level'].longitude.values})

        ## build model dataset
        model_data =xr.Dataset(data_vars = {"ivt":ivt_da, 
                                            "iwv":iwv_da, 
                                            'sfc_pressure':sfc_pressure_da, 
                                            'freezing_level':freezing_level_da,
                                            'u':u_wind_da,
                                            'v':v_wind_da,
                                            'wvflux':wvflux_da}, 
                                attrs={"model":"GFS", "init":gfs["sfc_pressure"].time.values, "valid_time":gfs["sfc_pressure"].valid_time.values, "datacrs":ccrs.PlateCarree(central_longitude=0)}
                                ) 


        return model_data