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
        "w_wind":{"typeOfLevel":'isobaricInhPa',"shortName":"w"}, #W-component of wind
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
        w_wind_da = xr.DataArray(gfs["w_wind"].values, name="w", dims=("pressure", "latitude","longitude"), coords={"pressure": gfs["w_wind"].isobaricInhPa.values, "latitude": gfs["w_wind"].latitude.values, "longitude":gfs["w_wind"].longitude.values})
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
                                            'w':w_wind_da,
                                            'wvflux':wvflux_da}, 
                                attrs={"model":"GFS", "init":gfs["sfc_pressure"].time.values, "valid_time":gfs["sfc_pressure"].valid_time.values, "datacrs":ccrs.PlateCarree(central_longitude=0)}
                                ) 


        return model_data