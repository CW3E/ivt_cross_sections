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
import cartopy.crs as ccrs

sys.path.append('../file-reader-toolkit') # point this to where you have the CW3E cmaps_toolkit repo located
import calculations
import calc_funcs as cfuncs

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
        freezing_level = cfuncs.calculate_zero_degree_isotherm( gfs['freezing_level'])
        
        #### Calculating IVT #####
        #extending pressure vector to 3d array to match rh shape
        pressure_3d = np.tile(gfs["rh"].isobaricInhPa.values[:, np.newaxis, np.newaxis], (1, gfs["rh"].values.shape[1], gfs["rh"].values.shape[2]))

        # calculating specific humidity from relative humidity for gfs
        gfs_q = calculations.specific_humidity(temperature=gfs["temperature"].values, pressure=pressure_3d*100, relative_humidity=gfs["rh"].values/100)

        #calculating ivt 
        gfs_uivt, gfs_vivt, gfs_ivt = calculations.ivt(u_wind=gfs["u_wind"].values,v_wind=gfs["v_wind"].values,specific_humidity=gfs_q, pressure=pressure_3d*100, sfc_pressure=gfs["sfc_pressure"].values)

        ## calculating wvflux
        # wv_flux = np.sqrt((gfs["u_wind"].values*gfs_q)**2 + (gfs["v_wind"].values*gfs_q)**2)
        density = cfuncs.calculate_air_density(pressure=pressure_3d, temperature=gfs["temperature"], relative_humidity=gfs["rh"])
        
        wv_flux = cfuncs.calculate_wvflux(uwind=gfs["u_wind"].values, vwind=gfs["v_wind"].values, density=density, specific_humidity=gfs_q)
        
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
    
class load_ECMWF_datasets:
    '''
    Loads variables needed for ivt cross section plots from ECMWF grb files
    
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
        
        if fdate is not None:
            date_string = fdate
            fpath = '/data/downloaded/Forecasts/ECMWF/NRT_data/{0}'.format(fdate)
            date_string = fdate
            print(date_string)

        else:
            path_to_data = '/data/downloaded/Forecasts/ECMWF/NRT_data/*'
            list_of_files = glob.glob(path_to_data)
            fpath = max(list_of_files, key=os.path.getctime)
            regex = re.compile(r'\d+')
            date_string = regex.findall(fpath)[-1]
            print(date_string)


        init_time = datetime.datetime.strptime(date_string,'%Y%m%d%H%M')
        lead_time = datetime.timedelta(hours=F)

        ecmwf_s2d_filename = "/S2D{init:%m%d%H%M}{valid:%m%d%H%M}1.grb".format(init=init_time, valid=init_time+lead_time)
        ecmwf_s1d_filename = "/S1D{init:%m%d%H%M}{valid:%m%d%H%M}1".format(init=init_time, valid=init_time+lead_time)

         ## for now: copy the files to local space
        repo_path = '/home/dnash/comet_data/tmp'
        # shutil.copy(fpath+ecmwf_s2d_filename, repo_path+ecmwf_s2d_filename) # copy file over to data folder
        # shutil.copy(fpath+ecmwf_s1d_filename, repo_path+ecmwf_s1d_filename) # copy file over to data folder 

        self.ecmwf_s2d_filename = repo_path+"/S2D{init:%m%d%H%M}{valid:%m%d%H%M}1.grb".format(init=init_time, valid=init_time+lead_time)
        self.ecmwf_s1d_filename = repo_path+"/S1D{init:%m%d%H%M}{valid:%m%d%H%M}1".format(init=init_time, valid=init_time+lead_time)

    def calc_vars(self):
        ecmwf_s2d_vardict = {
                    "u_wind":{"shortName":'u'},#U-component of wind
                    "v_wind":{"shortName":'v'}, #V-component of wind
                    "specific_humidity":{"shortName":'q'},#Specific humidity
                    "temperature":{"shortName":"t"} #Temperature
                        }

        ecmwf_s1d_vardict = {
                        "sfc_pressure":{"shortName":'sp'},
                        "iwv":{"shortName":'tcw'},
                        "freezing_level": {'shortName': 'deg0l'}
                        }

        ##reading ecmwf s1d and s2d files
        ##ecmwf is a dictionary of datasets
        ecmwf_s2d = read_ecmwf_S2D(filename=self.ecmwf_s2d_filename,vardict=ecmwf_s2d_vardict, show_catalog=False)
        ecmwf_s1d = read_ecmwf_S1D(filename=self.ecmwf_s1d_filename,vardict=ecmwf_s1d_vardict, show_catalog=False)

        ### calculating ivt
        ecmwf_uivt, ecmwf_vivt, ecmwf_ivt = calculations.ivt(u_wind=ecmwf_s2d["u_wind"].values,v_wind=ecmwf_s2d["v_wind"].values,specific_humidity=ecmwf_s2d["specific_humidity"], pressure=ecmwf_s2d["pressure"]*100)

        ## calculating wvflux
        rh = cfuncs.calc_relative_humidity_from_specific_humidity(ecmwf_s2d["pressure"].values, ecmwf_s2d["temperature"], ecmwf_s2d["specific_humidity"])
        density = cfuncs.calculate_air_density(pressure=ecmwf_s2d["pressure"].values, temperature=ecmwf_s2d["temperature"], relative_humidity=rh)
        
        wv_flux = cfuncs.calculate_wvflux(uwind=ecmwf_s2d["u_wind"].values, vwind=ecmwf_s2d["v_wind"].values, density=density, specific_humidity=ecmwf_s2d["specific_humidity"].values)
        
        wv_flux = xr.DataArray(wv_flux, name="wvflux", 
                             dims=("hybrid", "latitude","longitude"), 
                             coords={"hybrid": rh.hybrid.values, 
                                     "latitude": rh.latitude.values, 
                                     "longitude": rh.longitude.values})

        ## creating a 3D latitude array for plotting
        a = ecmwf_s2d["u_wind"].latitude
        b = ecmwf_s2d["u_wind"]
        a2, b2 = xr.broadcast(a, b)
        a2.name = '3D_lat'
        a2 = a2.transpose('hybrid', 'latitude', 'longitude')

        ## putting u, v, and pressure and 3D lat into a dataset
        ds_lst = [ecmwf_s2d["v_wind"], ecmwf_s2d["u_wind"], wv_flux, ecmwf_s2d["pressure"], a2]
        ds1 = xr.merge(ds_lst)

        ## get freezing level on pressure levels
        freezing_level = cfuncs.calculate_zero_degree_isotherm( ecmwf_s1d["freezing_level"])

        ## build final dataset
        var_dict = {'ivt': (['latitude', 'longitude'], ecmwf_ivt),
                    'iwv': (['latitude', 'longitude'], ecmwf_s1d["iwv"].values),
                    'sfc_pressure': (['latitude', 'longitude'], ecmwf_s1d["sfc_pressure"].values),
                    'freezing_level': (['latitude', 'longitude'], freezing_level)}

        model_data = xr.Dataset(var_dict,
                                coords={'latitude': (['latitude'], ecmwf_s1d["sfc_pressure"].latitude.values),
                                        'longitude': (['longitude'], ecmwf_s1d["sfc_pressure"].longitude.values)},
                               attrs={"model":"ECMWF", "init":ecmwf_s1d["sfc_pressure"].time.values, 
                                      "valid_time":ecmwf_s1d["sfc_pressure"].valid_time.values, 
                                      "datacrs":ccrs.PlateCarree(central_longitude=0)})

        ## merge vertical level data and single level data
        model_data = xr.merge([model_data, ds1])
        
        return model_data