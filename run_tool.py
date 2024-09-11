"""
Filename:    run_tool.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: For GEFS, and ECMWF: output .png files of ivt cross section plot for different logitudinal cross sections and lead times.
"""
import os
import sys
import numpy as np

from read_deterministic_data import load_GFS_datasets, load_ECMWF_datasets
from plotter import plot_ivt_cross_sections

model_name = sys.argv[1]
print('Creating ivt cross sections for {0}'.format(model_name))

## FIRST LOOP - LEAD TIME ##
F_lst = np.arange(6, 132, 12)

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
        s = load_ECMWF_datasets(F=F, fdate=None) #fdate='2024073000'
        model_data= s.calc_vars()

    elif model_name == 'GFS':
        s = load_GFS_datasets(F=F, fdate=None) #fdate='2024072212'
        model_data= s.calc_vars()
        
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
        print('...... Creating figure for {0}'.format(current_line[1]))
        plot_ivt_cross_sections(model_data, cross, line_lst, current_line, model_name, F)
        
        
    