#!/usr/bin/python3
"""
Filename:    read_deterministic_data.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: functions to read deterministic data from GFS, ECMWF, and WWRF
"""

import sys
import os

gfs_vardict = {
        "u_wind":{"typeOfLevel":'isobaricInhPa',"shortName":"u"}, #U-component of wind
        "v_wind":{"typeOfLevel":'isobaricInhPa',"shortName":"v"}, #V-component of wind
        "iwv":{"typeOfLevel":'atmosphereSingleLayer',"shortName":"pwat"}, #Integrated precipitable water or IWV
        "temperature":{"typeOfLevel":'isobaricInhPa',"shortName":"t"}, #Temperature
        "rh":{"typeOfLevel":'isobaricInhPa',"shortName":"r"}, #Relative Humidity
        "sfc_pressure":{'name': 'Surface pressure', 'typeOfLevel': 'surface', 'level': 0, 'paramId': 134, 'shortName': 'sp'}, #surface pressure
    }

#gfs is a dictionary of datasets
gfs = reader.read_gfs_deterministic(filename=gfs_filename,vardict=gfs_vardict, show_catalog=False)


#extending pressure vector to 3d array to match rh shape
pressure_3d = np.tile(gfs["rh"].isobaricInhPa.values[:, np.newaxis, np.newaxis], (1, gfs["rh"].values.shape[1], gfs["rh"].values.shape[2]))


# calculating specific humidity from relative humidity for gfs
gfs_q = calculations.specific_humidity(temperature=gfs["temperature"].values, pressure=pressure_3d*100, relative_humidity=gfs["rh"].values/100)


#calculating ivt 
gfs_uivt, gfs_vivt, gfs_ivt = calculations.ivt(u_wind=gfs["u_wind"].values,v_wind=gfs["v_wind"].values,specific_humidity=gfs_q, pressure=pressure_3d*100, sfc_pressure=gfs["sfc_pressure"].values)

# BUILD DATASET 
iwv_da = xr.DataArray(gfs["iwv"].values, name="iwv", dims=("latitude","longitude"), coords={"latitude": gfs["iwv"].latitude.values, "longitude":gfs["iwv"].longitude.values})
ivt_da = xr.DataArray(gfs_ivt, name="ivt", dims=("latitude","longitude"), coords={"latitude": gfs["rh"].latitude.values, "longitude":gfs["rh"].longitude.values})
uivt_da = xr.DataArray(gfs_uivt,name="uivt", dims=("latitude","longitude"), coords={"latitude": gfs["rh"].latitude.values, "longitude":gfs["rh"].longitude.values})
vivt_da = xr.DataArray(gfs_vivt,name="vivt", dims=("latitude","longitude"), coords={"latitude": gfs["rh"].latitude.values, "longitude":gfs["rh"].longitude.values})
sfc_pressure_da = xr.DataArray(gfs["sfc_pressure"].values, name="sfc_pressure", dims=("latitude","longitude"), coords={"latitude": gfs["sfc_pressure"].latitude.values, "longitude":gfs["sfc_pressure"].longitude.values})

model_data =xr.Dataset(data_vars = {"ivt":ivt_da, 
                                    "iwv":iwv_da, 
                                    'uivt':uivt_da, 
                                    'vivt':vivt_da, 
                                    'sfc_pressure':sfc_pressure_da, 
                                    }, 
                        attrs={"model":"GFS", "init":gfs["sfc_pressure"].time.values, "valid_time":gfs["sfc_pressure"].valid_time.values, "datacrs":ccrs.PlateCarree(central_longitude=0)}
                        ) 