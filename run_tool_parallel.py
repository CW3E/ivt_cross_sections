"""
Filename:    run_tool.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: For GEFS, and ECMWF: output .png files of ivt cross section plot for different logitudinal cross sections and lead times.
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing

from read_deterministic_data import load_GFS_datasets, load_ECMWF_datasets
from plotter import plot_ivt_cross_sections
from calc_funcs import format_timedelta_to_HHMMSS

model_name = sys.argv[1]
print('Creating ivt cross sections for {0}'.format(model_name))
start_time = pd.Timestamp.today()

## FIRST LOOP - LEAD TIME ##
F_lst = np.arange(12, 132, 12)

## create list that has the start and end points for the cross section
lon_lst = np.arange(185., 260., 5.)
line_lst = []
for i, lon in enumerate(lon_lst):
    ## create list of lines where [start_lat, start_lon, end_lat, end_lon]
    line = [25., lon, 65., lon]
    line_lst.append(line)
    
    
for i, F in enumerate(F_lst):
    #######################
    ### LOAD MODEL DATA ###
    #######################
    print('... Loading {0} data for {1} hour lead'.format(model_name, F))
    if model_name == 'ECMWF':
        s = load_ECMWF_datasets(F=F, fdate=None)
        model_data= s.calc_vars()

    elif model_name == 'GFS':
        s = load_GFS_datasets(F=F, fdate=None)
        model_data= s.calc_vars()
        
    model_data
    ## write intermediate data files
    out_fname = '/data/projects/operations/ivt_cross_sections/data/tmp_{0}_{1}.nc'.format(model_name, F)
    model_data.to_netcdf(path=out_fname, mode = 'w', format='NETCDF4')
    model_data.close() ## close data
    
    
def multiP(model_data, line_lst, F):
    ## SECOND LOOP - LOOP THROUGH LONGITUDE FOR CROSS SECTION ##
    for k, current_line in enumerate(line_lst):
        ## subset vertical data and IVT data to current line
        if model_name == 'ECMWF':
            cross = model_data.sel(latitude = slice(current_line[2]+0.1, current_line[0]-0.1))
            cross = cross.sel(longitude=current_line[1], method='nearest')
        elif model_name == 'GFS':
            cross = model_data.sel(latitude = slice(current_line[2], current_line[0]), longitude=current_line[1])

        cross = cross.sortby('latitude')
        
        ### Create Plots
        p = multiprocessing.Process(target=plot_ivt_cross_sections, args=(model_data, cross, line_lst, current_line, model_name, F))
        p.start()


        
for i, F in enumerate(F_lst):
    #######################
    ### LOAD MODEL DATA ###
    #######################
    print('... Loading intermediate {0} data for {1} hour lead'.format(model_name, F))
    out_fname = '/data/projects/operations/ivt_cross_sections/data/tmp_{0}_{1}.nc'.format(model_name, F)
    model_data = xr.open_dataset(out_fname, engine='netcdf4')
        
    if __name__ == "__main__": 
        multiP(model_data, line_lst, F)
    
    model_data.close() ## close data
    
end_time = pd.Timestamp.today()
td = end_time - start_time
td = format_timedelta_to_HHMMSS(td)
print('Plots for {0} took {1} to run'.format(model_name, td))
        
        
    