#!/usr/bin/python3
"""
Filename:    read_deterministic_data.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: functions to read deterministic data from GFS, ECMWF, and WWRF
"""

class load_datasets:
    '''
    Copies or downloads the latest 7-d QPF for the current model initialization time
    Loads 7-d QPF
    Loads and calculates IVT vars for plotting
    
    Parameters
    ----------
    forecast : str
        name of the forecast product
        this can be 'ECMWF', 'GFS', 'WWRF'
    loc : str
        name of the location for the .txt file with latitude and longitude of locations for analysis
        this can be 'US-west', 'AK', or 'SAK'
    ptloc : str
        name of the transect of .txt file with latitude and longitude of locations for analysis
        this file should have no header, with latitude, then longitude, separated by a space
        this can be 'coast', 'foothills', or 'inland'
  
    Returns
    -------
    grb : grb file
        grb file downloaded for current run of figures
    
    '''
    def __init__(self, forecast, loc, ptloc, fname=None):
        path_to_data = '/data/downloaded/SCRATCH/cw3eit_scratch/'
        self.path_to_out = '/home/cw3eit/ARPortal/gefs/scripts/ar_landfall_tool/data/'
        self.forecast = forecast
        if self.forecast == 'GEFS':
            self.fpath = path_to_data + 'GEFS/FullFiles/'
            self.ensemble_name = 'GEFS'
            self.datasize_min = 15.
        elif self.forecast == 'ECMWF' or self.forecast == 'ECMWF-GEFS':
            self.fpath = path_to_data + 'ECMWF/' 
            self.ensemble_name = 'ECMWF'
            self.datasize_min = 25.
        elif forecast == 'W-WRF':
            self.fpath = '/data/downloaded/WWRF-NRT/2023-2024/Ensemble_IVT/' 
            self.ensemble_name = 'West-WRF'
            self.datasize_min = 50.
        else:
            print('Forecast product not available! Please choose either GEFS, ECMWF, ECMWF-GEFS, or W-WRF.')
        if fname is not None:
            self.fname = fname
        if fname is None:          
            ## find the most recent file in the currect directory
            list_of_files = glob.glob(self.fpath+'*.nc')
            self.fname = max(list_of_files, key=os.path.getctime)
        
        # pull the initialization date from the filename
        regex = re.compile(r'\d+')
        date_string = regex.findall(self.fname)
        if self.forecast == 'W-WRF':
            self.date_string = date_string[2]
        else:
            self.date_string = date_string[1]

        self.model_init_date = datetime.datetime.strptime(self.date_string, '%Y%m%d%H')
        self.ptloc = ptloc
        self.loc = loc
        
    def download_QPF_dataset(self):
        date = self.model_init_date.strftime('%Y%m%d') # model init date
        hr = self.model_init_date.strftime('%H') # model init hour

        if self.forecast == 'GEFS':
            ## download from NOMADS
            print(date, hr)
            subprocess.check_call(["download_QPF.sh", date, hr], shell=True) # downloads the latest QPF data
        else:
            mmdyhr_init = self.model_init_date.strftime('%m%d%H') # month-day-hr init date
            date1 = datetime.datetime.strptime(self.date_string, '%Y%m%d%H')
            date2 = date1 + datetime.timedelta(days=7) 
            date2 = date2.strftime('%m%d%H') # valid date
            fpath = '/data/downloaded/Forecasts/ECMWF/NRT_data/{0}{1}/'.format(date, hr)
            fname = 'S1D{0}00{1}001'.format(mmdyhr_init, date2)
            shutil.copy(fpath+fname, self.path_to_out+'precip_ECMWF') # copy file over to data folder
            
            
    def load_prec_QPF_dataset(self):
        
        if self.forecast == 'GEFS':
            try: 
                ## this method directly opens data from NOMADS
                date = self.model_init_date.strftime('%Y%m%d') # model init date
                hr = self.model_init_date.strftime('%H') # model init hour
                url = 'https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{0}/gfs_0p25_{1}z'.format(date, hr)
                ds = xr.open_dataset(url, decode_times=False)
                ds = ds.isel(time=7*8) # get 7-day QPF - the variable is already cumulative
                prec = ds['apcpsfc']/25.4 # convert from mm to inches
            except OSError:
                try:
                    ## This method uses the downloaded data
                    self.download_QPF_dataset()
                    ds = xr.open_dataset('precip_GFS.grb', engine='cfgrib', backend_kwargs={"indexpath": ""})
                    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
                    prec = ds['tp']/25.4 # convert from mm to inches
                except OSError:
                    ## build a fake precip dataset of 0s
                    lats = np.arange(-90., 90.25, .25)
                    lons = np.arange(-180., 180.25, .25)
                    var_dict = {'tp': (['lat', 'lon'], np.zeros((len(lats), len(lons))))}
                    prec = xr.Dataset(var_dict, coords={'lat': (['lat'], lats),
                                                        'lon': (['lon'], lons)})
        
        else:
            self.download_QPF_dataset()
            var_lst = ['u10','lsm','msl','d2m','z','t2m','stl1', 'stl2', 'stl3', 'stl4', 'swvl4','swvl2', 'swvl3','sst','sp','v10','sd','skt', 'swvl1','siconc','tcwv','tcw']
            ds = xr.open_dataset(self.path_to_out+'precip_ECMWF', drop_variables=var_lst, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
            prec = ds['tp']*39.3701 # convert from m to inches
            prec = prec.rename({'longitude': 'lon', 'latitude': 'lat'}) # need to rename this to match GEFS
        
        return prec

    def get_lat_lons_from_txt_file(self):
        ## read text file with points
        textpts_fname = self.path_to_out+'{0}/latlon_{1}.txt'.format(self.loc, self.ptloc)
        df = pd.read_csv(textpts_fname, header=None, sep=' ', names=['latitude', 'longitude'], engine='python')
        df['longitude'] = df['longitude']*-1
        df = df
        self.lons = df['longitude'].values
        self.lats = df['latitude'].values


    def calc_ivt_vars(self):
        ## load the forecast data
        ds = xr.open_dataset(self.fname)
        ds = ds.assign_coords({"lon": (((ds.lon + 180) % 360) - 180)}) # Convert DataArray longitude coordinates from 0-359 to -180-179
        ds = ds.sel(lon=slice(-180., -1)) # keep only western hemisphere
        if self.forecast == 'ECMWF':
            ds = ds.rename({'forecast_time': 'forecast_hour'}) # need to rename this to match GEFS
            ds = ds.assign_coords({"forecast_hour": (ds.forecast_hour*3)}) # convert forecast time to forecast hour
        elif self.forecast == 'W-WRF':
            ds = ds.rename({'ensembles': 'ensemble'}) # need to rename this to match GEFS/ECMWF

        ## Calculate IVT for the maps
        ds1 = ds.sel(forecast_hour=slice(0, 24*7)).mean(['forecast_hour', 'ensemble'])
        ds1 = ds1.where(ds1.IVT >= 250)

        # subset ds to the select points
        self.get_lat_lons_from_txt_file()
        x = xr.DataArray(self.lons, dims=['location'])
        y = xr.DataArray(self.lats, dims=['location'])
        ds = ds.sel(lon=x, lat=y, method='nearest')

        ## Calculate probability and duration IVT >= threshold
        thresholds = [100, 150, 250, 500, 750, 1000]
        probability_lst = []
        duration_lst = []
        for i, thres in enumerate(thresholds):
            data_size = ds.IVT.count(dim='ensemble')
            ## set ensemble size to nan if # of ensembles is not > 50% for GEFS and ECMWF
            data_size = data_size.where(data_size >= self.datasize_min)
            tmp = ds.IVT.where(ds.IVT >= thres) # find where IVT exceeds threshold

            # sum the number of ensembles where IVT exceeds threshold
            probability = tmp.count(dim='ensemble') / data_size
            probability_lst.append(probability)

            # sum the number of time steps where IVT exceeds threshold
            duration = tmp.count(dim='forecast_hour')*3 # three hour time intervals
            duration_lst.append(duration)

        # merge duration and probability datasets
        duration_ds = xr.concat(duration_lst, pd.Index(thresholds, name="threshold"))
        prob_ds = xr.concat(probability_lst, pd.Index(thresholds, name="threshold"))

        ## Calculate Vectors
        uvec = ds.uIVT.mean('ensemble') # get the ensemble mean uIVT
        vvec = ds.vIVT.mean('ensemble') # get the ensemble mean vIVT
        # get the ensemble mean IVT where there are enough ensembles
        ensemble_mean = ds.IVT.where(data_size >= self.datasize_min).mean(dim='ensemble')

        # normalize vectors
        u = uvec / ensemble_mean
        v = vvec / ensemble_mean

        control = ds.IVT.sel(ensemble=0)

        ## place into single dataset
        x1 = ensemble_mean.rename('ensemble_mean')
        x2 = control.rename('control')
        x3 = v.rename('v')
        x4 = u.rename('u')
        x5 = duration_ds.rename('duration')
        x6 = prob_ds.rename('probability')

        x_lst = [x1, x2, x3, x4, x5, x6]
        final_ds = xr.merge(x_lst)
        
        ##Add attribute information
        final_ds = final_ds.assign_attrs(model_init_date=self.model_init_date)

        return final_ds, ds1