#!/usr/bin/python3
"""
Filename:    plotter.py
Author:      Deanna Nash, dnash@ucsd.edu and Ricardo Vilela, rbatistavilela@ucsd.edu
Description: Functions for plotting IVT cross sections
"""

import os, sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import colorsys
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap # Linear interpolation for color maps
import matplotlib.patches as mpatches
from matplotlib import cm, colors as clr
from matplotlib.colorbar import Colorbar # different way to handle colorbar
import pandas as pd
from matplotlib.gridspec import GridSpec
import itertools
from PIL import Image
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
import copy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib as mpl
mpl.use('agg')

import cw3ecmaps as ccmaps
import domains as domains

def roundPartial(value, resolution):
    return np.round(value / resolution) * resolution

def set_cw3e_font(current_dpi, scaling_factor):
    fm.fontManager.addfont('/home/cw3eit/ARPortal/gefs/scripts/ar_landfall_tool/utils/fonts/helvetica.ttc')

    plt.rcParams.update({
                    'font.family' : 'Helvetica',
                    'figure.dpi': current_dpi,
                    'font.size': 8 * scaling_factor, #changes axes tick label
                    'axes.labelsize': 8 * scaling_factor,
                    'axes.titlesize': 8 * scaling_factor,
                    'xtick.labelsize': 8 * scaling_factor,#do nothing
                    'ytick.labelsize': 8 * scaling_factor, #do nothing
                    'legend.fontsize': 5 * scaling_factor,
                    'lines.linewidth': 0.7 * scaling_factor,
                    'axes.linewidth': 0.2 * scaling_factor,
                    'legend.fontsize': 12 * scaling_factor,
                    'xtick.major.width': 0.8 * scaling_factor,
                    'ytick.major.width': 0.8 * scaling_factor,
                    'xtick.minor.width': 0.6 * scaling_factor,
                    'ytick.minor.width': 0.6 * scaling_factor,
                    'lines.markersize': 6 * scaling_factor
                })

def plot_cw3e_logo(ax, orientation):
    ## location of CW3E logo
    if orientation == 'horizontal':
        im = '/common/CW3E_Logo_Suite/1-Horzontal-PRIMARY_LOGO/Digital/JPG-RGB/CW3E-Logo-Horizontal-FullColor-RGB.jpg'
    else:
        im = '/common/CW3E_Logo_Suite/5-Vertical-Acronym_Only/Digital/PNG/CW3E-Logo-Vertical-Acronym-FullColor.png'
    img = np.asarray(Image.open(im))
    ax.imshow(img)
    ax.axis('off')
    return ax


def draw_basemap(ax, datacrs=ccrs.PlateCarree(), extent=None, xticks=None, yticks=None, grid=False, left_lats=True, right_lats=False, bottom_lons=True, mask_ocean=False, coastline=True):
    """
    Creates and returns a background map on which to plot data. 
    
    Map features include continents and country borders.
    Option to set lat/lon tickmarks and draw gridlines.
    
    Parameters
    ----------
    ax : 
        plot Axes on which to draw the basemap
    
    datacrs : 
        crs that the data comes in (usually ccrs.PlateCarree())
        
    extent : float
        Set map extent to [lonmin, lonmax, latmin, latmax] 
        Default: None (uses global extent)
        
    grid : bool
        Whether to draw grid lines. Default: False
        
    xticks : float
        array of xtick locations (longitude tick marks)
    
    yticks : float
        array of ytick locations (latitude tick marks)
        
    left_lats : bool
        Whether to add latitude labels on the left side. Default: True
        
    right_lats : bool
        Whether to add latitude labels on the right side. Default: False
        
    Returns
    -------
    ax :
        plot Axes with Basemap
    
    Notes
    -----
    - Grayscale colors can be set using 0 (black) to 1 (white)
    - Alpha sets transparency (0 is transparent, 1 is solid)
    
    """
    ## some style dictionaries
    kw_ticklabels = {'color': 'black', 'weight': 'light'}
    kw_grid = {'linewidth': 0.6, 'color': 'k', 'linestyle': '--', 'alpha': 0.4}
    kw_ticks = {'length': 4, 'width': 0.5, 'pad': 2, 'color': 'black',
                             'labelcolor': 'dimgray'}

    # Use map projection (CRS) of the given Axes
    mapcrs = ax.projection    
    
    # Add map features (continents and country borders)
    ax.add_feature(cfeature.LAND, facecolor='0.9')      
    ax.add_feature(cfeature.BORDERS, edgecolor='0.4', linewidth=0.8)
    ax.add_feature(cfeature.STATES, edgecolor='0.2', linewidth=0.2)
    if coastline == True:
        ax.add_feature(cfeature.COASTLINE, edgecolor='0.4', linewidth=0.8)
    if mask_ocean == True:
        ax.add_feature(cfeature.OCEAN, edgecolor='0.4', zorder=12, facecolor='white') # mask ocean
        
    ## Tickmarks/Labels
    ## Add in meridian and parallels
    if mapcrs == ccrs.NorthPolarStereo(central_longitude=0.):
        gl = ax.gridlines(draw_labels=False,
                      linewidth=.5, color='black', alpha=0.5, linestyle='--')
    elif mapcrs == ccrs.SouthPolarStereo(central_longitude=0.):
        gl = ax.gridlines(draw_labels=True,
                      linewidth=.5, color='black', alpha=0.5, linestyle='--')
        
    else:
        gl = ax.gridlines(crs=datacrs, draw_labels=True, **kw_grid)
        gl.top_labels = False
        gl.left_labels = left_lats
        gl.right_labels = right_lats
        gl.bottom_labels = bottom_lons
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = kw_ticklabels
        gl.ylabel_style = kw_ticklabels
        
        # apply tick parameters
        ax.set_xticks(xticks, crs=datacrs)
        ax.set_yticks(yticks, crs=datacrs)
        plt.yticks(color='w', size=1) # hack: make the ytick labels white so the ticks show up but not the labels
        plt.xticks(color='w', size=1) # hack: make the ytick labels white so the ticks show up but not the labels
        ax.ticklabel_format(axis='both', style='plain')
    
    ## Gridlines
    # Draw gridlines if requested
    if (grid == True):
        gl.xlines = True
        gl.ylines = True
    if (grid == False):
        gl.xlines = False
        gl.ylines = False

    ## Map Extent
    # If no extent is given, use global extent
    if extent is None:        
        ax.set_global()
        extent = [-180., 180., -90., 90.]
    # If extent is given, set map extent to lat/lon bounding box
    else:
        ax.set_extent(extent, crs=datacrs)
    
    return ax

def add_subregion_boxes(ax, subregion_xy, width, height, ecolor, datacrs):
    '''This function will add subregion boxes to the given axes.
    subregion_xy 
    [[ymin, xmin], [ymin, xmin]]
    '''
    for i in range(len(subregion_xy)):
        ax.add_patch(mpatches.Rectangle(xy=subregion_xy[i], width=width[i], height=height[i],
                                        fill=False,
                                        edgecolor=ecolor,
                                        linewidth=1.0,
                                        transform=datacrs,
                                        zorder=100))
        
    return ax


def plot_ivt_cross_sections(ds, cross, line_lst, current_line, model_name, F):
    '''
    Plots IVT Cross Section Plot from given data
    
    Parameters
    ----------
    ds : 
        xarray object with 2D IVT
        
    cross : 
        xarray object with longitudinal cross section of wvflux, IVT, IWV, u, v, and freezing level
        
    current_line : list
        [minlat, longitude, maxlat, longitude] current line for cross section
        
    model_name : str
        ECMWF or GFS
    
    F : int
        model lead time
  
    Returns
    -------
    png : 
        ivt cross section figure
    
    '''
    
    ## Set domain name
    domain_name = 'ivtcross'

    ## Set up data based on domain name
    ext = domains.extent[domain_name]['ext']
    dx = domains.extent[domain_name]['xticks']
    dy = domains.extent[domain_name]['yticks']

    datacrs = ccrs.PlateCarree()  ## the projection the data is in
    mapcrs = domains.extent[domain_name]['ccrs'] ## the projection you want your map displayed in

    kw_ticks = {'length': 4, 'width': 0.5, 'pad': 2, 'color': 'black',
                'labelsize': 10, 'labelcolor': 'dimgray'}

    kw_ticklabels = {'size': 8, 'color': 'dimgray', 'weight': 'light'}
    style = {'size': 7, 'color': 'black', 'fontweight': 'normal'}

    ## create title label
    current_lon = current_line[1]
    clon = ((current_lon + 180) % 360) - 180
    if clon > 0:
        lon_lbl = u"{:.0f}\N{DEGREE SIGN}E".format(clon)
    else:
        lon_lbl = u"{:.0f}\N{DEGREE SIGN}W".format(clon*-1)

    wvflux_units = 'kg m$^{-2}$ s$^{-1}$'
    ivt_units = 'kg m$^{-1}$ s$^{-1}$'
    wind_units = '(knots)'
    title = '{0} IVT ({4}), WV Flux ({1}) and Wind {2} | {3}'.format(model_name, wvflux_units, wind_units, lon_lbl, ivt_units)

    init_date = pd.to_datetime(ds.attrs["init"]).strftime('%H UTC %d %b %Y')
    valid_date = pd.to_datetime(ds.attrs["valid_time"]).strftime('%H UTC %d %b %Y')

    left_title = 'Initialized: {0}'.format(init_date)
    right_title = 'F-{0} Valid: {1}'.format(str(F).zfill(3), valid_date)
    
    ## find where IVT >=250 and IWV >=20 for shading
    idx = (cross.ivt >=250) & (cross.iwv >=20.)
    s = pd.Series(xr.where(idx, True, False))
    grp = s.eq(False).cumsum()
    arr = grp.loc[s.eq(True)] \
             .groupby(grp) \
             .apply(lambda x: [x.index.min(), x.index.max()])

    current_dpi=600 #recommended dpi of 600
    base_dpi=100
    scaling_factor = (current_dpi / base_dpi)**0.13

    set_cw3e_font(current_dpi, scaling_factor)

    nrows = 3
    ncols = 4

    ## Use gridspec to set up a plot with a series of subplots that is
    ## n-rows by n-columns
    gs = GridSpec(nrows, ncols, height_ratios=[1.5, 1, 0.05], width_ratios = [1.5, 0.1, 1, 0.05], wspace=0.06, hspace=0.3)
    ## use gs[rows index, columns index] to access grids

    fig = plt.figure(figsize=(13.0, 5))
    fig.dpi = current_dpi
    fname = 'figs/{0}/Cross_Section_latest_25-65N_{1}W_F{2}'.format(model_name, clon, str(F).zfill(3))
    fmt = 'png'

    ####################
    ### PLOT IVT MAP ###
    ####################

    ax = fig.add_subplot(gs[0:2, 0], projection=mapcrs)
    ax = draw_basemap(ax, extent=ext, xticks=dx, yticks=dy, left_lats=True, right_lats=False, grid=True)
    ax.set_extent(ext, datacrs)
    ax.add_feature(cfeature.STATES, edgecolor='0.4', linewidth=0.8)

    ## filled contour (IVT)
    lats = ds.latitude.values
    lons = ds.longitude.values
    data = ds.ivt.values
    cmap, norm, bnds, cbarticks, cbarlbl = ccmaps.cmap('ivt') # get cmap from our custom function

    cf = ax.contourf(lons, lats, data, transform=datacrs, levels=bnds, cmap=cmap, norm=norm, alpha=0.9)

    ## contour lines (IVT)
    cs = ax.contour(lons, lats, data, transform=datacrs, levels=bnds, colors=['black'], linewidths=0.3, alpha=0.9, zorder=100)

    ## black lines where possible cross sections are
    for i, line in enumerate(line_lst):
        ax.plot([line[1], line[3]], [line[0], line[2]], color='k', transform=datacrs, zorder=3)

    ## red line where current cross section is showing
    ax.plot([current_line[1], current_line[3]], [current_line[0], current_line[2]], color='r', transform=datacrs, zorder=3)

    ax.set_title(title + "\n" + left_title, loc="left")

    ## color bar
    cbax = fig.add_subplot(gs[-1, 0]) # colorbar axis (first row, last column)
    cbarticks = list(itertools.compress(bnds, cbarticks)) ## this labels the cbarticks based on the cmap dictionary
    cb = Colorbar(ax = cbax, mappable = cf, orientation = 'horizontal', ticklocation = 'bottom', ticks=cbarticks)
    cb.set_label(cbarlbl)
    cb.ax.tick_params(labelsize=12)

    #####################
    ### CROSS SECTION ###
    #####################
    ax = fig.add_subplot(gs[0, 2])

    ## y-axis is pressure
    ## x-axis is latitude

    xs = cross.latitude.values
    if model_name == 'ECMWF':
        x = cross['3D_lat'].values
    ys = cross.pressure.values
    terline = cross.sfc_pressure.values / 100. ## convert from Pa to hPa
    ht_fill = ax.fill_between(xs, 1000., terline, facecolor='k', edgecolor='k', zorder=10)

    # Filled contours (WV flux)
    cmap, norm, bnds, cbarticks, cbarlbl = ccmaps.cmap('wvflux') # get cmap from our custom function
    if model_name == 'ECMWF':
        cf = ax.contourf(x, ys, cross.wvflux.values, levels=bnds, cmap=cmap, norm=norm, alpha=0.9, extend='neither', zorder=-1)
    if model_name == 'GFS':
        cf = ax.contourf(xs, ys, cross.wvflux.values, levels=bnds, cmap=cmap, norm=norm, alpha=0.9, extend='neither', zorder=-1)

    plt.gca().invert_yaxis()
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.set_ylabel("Pressure (hPa)")
    ax.set_ylim(1000, 200)
    ax.set_yticks([1000., 850., 700., 500., 400., 300., 250., 200.])

    ## add freezing level
    freeze_line = ax.plot(xs, cross.freezing_level, c='k')
    ax.annotate('0', xy=(xs[20],cross.freezing_level[20]), xycoords='data',
                textcoords="offset points", # how to position the text
                xytext=(0,-3), # distance from text to points (x,y)
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec=None, lw=0.0),
                **style)

    # wind vectors
    if model_name == 'ECMWF':
        dw = 10 # how often to plot vector vertically
        dw2 = 17 # how often to plot horizontally
        ax.barbs(x[::dw, ::dw2], ys[::dw, ::dw2], cross.u.values[::dw, ::dw2]*1.94384, cross.v.values[::dw, ::dw2]*1.94384, 
                 linewidth=0.25, length=3.5)
    elif model_name == 'GFS':
        dw = 6 # how often to plot vector
        ax.barbs(xs[::dw], ys[::dw], cross.u.values[::dw, ::dw]*1.944, cross.v.values[::dw, ::dw]*1.944, 
                 linewidth=0.25, length=3.5)

    ## apply xtick parameters (latitude labels)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(xs[0], xs[-1]+1, 5)))
    ax.xaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.tick_params(axis='x', which='major')

    ## add titles
    ax.set_title(right_title, loc="right")

    # # Add color bar
    cbax = fig.add_subplot(gs[0, -1]) # colorbar axis
    cb = Colorbar(ax = cbax, mappable = cf, orientation = 'vertical', ticklocation = 'right', ticks=cbarticks)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_yticklabels(['{:g}'.format(x) for x in cbarticks]) # force cbartick labels to drop trailing zeros

    ## Add CW3E logo
    in_ax = inset_axes(ax, width="10%", height="20%", loc='upper left')
    in_ax = plot_cw3e_logo(in_ax, orientation='vertical')

    ###################
    ### TIME SERIES ###
    ###################
    ax = fig.add_subplot(gs[1:, 2])

    ax.plot(xs, cross.iwv.values, color='blue', linewidth=0.75)
    ax.set_ylabel('IWV (mm)', color='blue')
    ax.axhline(y=20.0, color='blue', linestyle='--', linewidth=0.75)
    ax.set_xlim(xs[0], xs[-1])

    ## add IWV label to line
    ax.annotate('IWV', xy=(xs[20],cross.iwv[20]), xycoords='data',
                textcoords="offset points", # how to position the text
                xytext=(0,-0.5), # distance from text to points (x,y)
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec=None, lw=0.0),
                **style)

    ## set adaptive nice yticks
    iwv_max = roundPartial(cross.iwv.max().values+10, 10)
    yticks = np.arange(0, iwv_max+10, 10)
    ax.yaxis.set_major_locator(mticker.FixedLocator(yticks))
    ax.tick_params(axis='y', which='minor', left=True)
    ax.set_ylim(yticks[0], yticks[-1])

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(xs, cross.ivt.values, color='red')
    ax2.set_ylabel('IVT (kg m$^{-1}$ s$^{-1}$)', color='red')
    ax2.axhline(y=250.0, color='red', linestyle='--')
    ax2.set_xlim(xs[0], xs[-1])

    ## add IVT label to line
    ax2.annotate('IVT', xy=(xs[20],cross.ivt[20]), xycoords='data',
                textcoords="offset points", # how to position the text
                xytext=(0,-2), # distance from text to points (x,y)
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec=None, lw=0.0),
                **style)

    ## set adaptive nice yticks
    ivt_max = roundPartial(cross.ivt.max().values+100, 250)
    yticks = np.arange(0, ivt_max+100, 250)
    ax2.yaxis.set_major_locator(mticker.FixedLocator(yticks))
    ax2.minorticks_on()
    ax2.tick_params(axis='y', which='minor', right=True)
    ax2.set_ylim(yticks[0], yticks[-1])

    ## apply xtick parameters (latitude labels)
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(xs[0], xs[-1]+1, 5)))
    ax.xaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=True)
    ax.tick_params(axis='x', which='major')

    ## add in grey shading where IWV and IVT > thres
    for i in range(len(arr)):
        ax.axvspan(xs[arr.iloc[i][0]], xs[arr.iloc[i][1]], color='grey', alpha=0.2, lw=None)

    ## add in annotation of max IWV and IVT vals
    IWV_ann = 'Max IWV: {0:0.0f} mm'.format(cross.iwv.max().values)
    IVT_ann = 'Max IVT: {0:0.0f} {1}'.format(cross.ivt.max().values, ivt_units)

    xy = [(.01, .93), (.68, .93)]
    for i, lbl in enumerate([IWV_ann, IVT_ann]):
        ax.annotate(lbl, # this is the text
                    xy[i], # these are the coordinates to position the label
                    xycoords='axes fraction',
                    zorder=200,
                    **style)

    fig.savefig('%s.%s' %(fname, fmt), bbox_inches='tight', dpi=fig.dpi, transparent=False)
    fig.clf()