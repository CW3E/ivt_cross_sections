"""
Filename:    domains.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: Dictionaries of domains for CW3E web maps
"""

## each domain can be used as extent argument for plotting: ext = [lonmin, lonmax, latmin, latmax] 
## Example: 
# "name_of_domain":  {'ext': [eastern most longitude, western most longitude, southern most latitude, northern-most latitude], # note that the extent should be in 0 to 360 longitude (unless it crosses the prime meridian, then it should be in -180 to 180 and -90 to 90 latitude
#                         'xticks': [locations of longitude label ticks]
#                         'yticks': [locations of latitude lable ticks]
#                         'center_longitude': the central longitude of the map
#                         'figsize': the figure size in inches, (width, height)
#                        }

import cartopy.crs as ccrs

extent = {"npacific":      {'ext': [115., 245., -10., 70.], # North Pacific (extent in 0-360)
                            'xticks': [125, 140, 155, 170, -175, -160, -145, -130, -115],
                            'yticks': [-10, 0, 10, 20, 30, 40, 50, 60, 70],
                            'ccrs': ccrs.PlateCarree(central_longitude=180),
                            'figsize': (10. ,5.5)
                           },
          "wwrf_npacific":  {'ext': [170., 260., 10., 65.], # North Pacific (extent in 0-360)
                            'xticks': [170, -175, -160, -145, -130, -115, -100],
                            'yticks': [10, 20, 30, 40, 50, 60, 70],
                            'ccrs': ccrs.PlateCarree(central_longitude=180),
                            'figsize': (10. ,5.5)
                           },
            
          
           "nepacific":    {'ext': [180., 245., 10., 70.], # Northeast Pacific (extent in 0-360)
                            'xticks': [-180, -170, -160, -150, -140, -130, -120],
                            'yticks': [10, 20, 30, 40, 50, 60, 70],
                            'ccrs': ccrs.PlateCarree(central_longitude=180),
                            'figsize': (10. ,8.)
                            },  
          
           "uswc":         {'ext': [220., 250., 20., 50.], # U.S. West Coast (extent in 0-360)
                            'xticks': [-140, -135, -130, -125, -120, -115, -110],
                            'yticks': [20, 25, 30, 35, 40, 45, 50],
                            'ccrs': ccrs.PlateCarree(central_longitude=180),
                            'figsize': (10. ,8.5)
                           },  
          
           "namerica":     {'ext': [220, 300, 10., 60.], # North America
                            'xticks': [-140, -130, -120, -110, -100, -90, -80, -70, -60],
                            'yticks': [10, 20, 30, 40, 50, 60],
                            'ccrs': ccrs.PlateCarree(central_longitude=0),
                            'figsize': (10. ,5.7)
                           }, 
          
           "natlantic":    {'ext': [-90., 15., 10., 70.], # North Atlantic (ext in -180 to 180)
                            'xticks': [-90, -75, -60, -45, -30, -15, 0, 15],
                            'yticks': [10, 20, 30, 40, 50, 60, 70],
                            'ccrs': ccrs.PlateCarree(central_longitude=0),
                            'figsize': (10. ,5.2)
                           }, 
          
           "intwest":      {'ext': [-130., -100., 20., 50.], # Interior West
                            'xticks': [-130, -125, -120, -115, -110, -105, -100],
                            'yticks': [20, 25, 30, 35, 40, 45, 50],
                            'ccrs': ccrs.PlateCarree(central_longitude=0),
                            'figsize': (10. ,8.5)
                           },
          
           "antarctica":   {'ext': [-180., 180, -50, -90.], # Antarctica
                            'xticks': [-15, -30,-45, -60, -75, -90, -105, -120, -135, -150, -165, 180, 165, 150, 135, 120, 105, 90, 75, 60, 45, 30, 15, 0],
                            'yticks': [-50, -60, -70, -80, -90],
                            'ccrs': ccrs.SouthPolarStereo(central_longitude=0),
                            'figsize': (10. ,8.5)
                           }, 
           
           "ivtcross":    {'ext': [180., 260., 20., 70.], # Northeast Pacific plus a few (extent in 0-360)
                            'xticks': [-175, -165, -155, -145, -135, -125, -115, -110, -105],
                            'yticks': [20, 30, 40, 50, 60, 70],
                            'ccrs': ccrs.PlateCarree(central_longitude=180),
                            'figsize': (10. ,8.)
                            },  
          }
           
           
