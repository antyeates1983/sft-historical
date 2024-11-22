"""
    Python tools for reading digitised sunspot polarity measurements from MWO.
    
    A Yeates 2024-Aug
"""
import numpy as np
from scipy.io import netcdf
from scipy.signal import savgol_filter
import datetime
from paths import datapath_mwo_spots

#--------------------------------------------------------------------------------------------------------------
def prep_mwo_spots(crs, mwo_lag_window_min, mwo_lag_window_max, mwo_lag_width, mwo_lag_degree):
    """
    Prepare sunspot data. [Lag corrections are computed but not applied here.]
    """

    # Read MWO sunspot data:
    spotfile = datapath_mwo_spots+'mwo-spots.csv'
    data = np.genfromtxt(spotfile, delimiter=',', dtype=None, invalid_raise=False, names=True)
    date_spt = data['UT_date']
    time_spt = data['UT_time']
    lat_spt = np.deg2rad(data['q'])
    lon_spt = np.deg2rad(data['L'])
    b_spt = data['flux']*100
    try:
        # - offset all values in longitude (based on comparison to Ca-K):
        fid = netcdf.netcdf_file('spot-offsets-all.nc', 'r', mmap=False)
        crs_lags = fid.variables['cr'][:].astype(np.int16)
        opt_lags = fid.variables['lags'][:]
        fid.close()
    except:
        # - if no precomputed lag values, set all to 4 degrees:
        crs_lags = crs
        opt_lags = crs*0 - 4
    # Smooth lag variation over time:
    opt_lags[(opt_lags < mwo_lag_window_min) | (opt_lags > mwo_lag_window_max)] = np.nan
    # - interpolate to skip nans:
    opt_lags = np.interp(crs_lags, crs_lags[~np.isnan(opt_lags)], opt_lags[~np.isnan(opt_lags)])
    # - smooth to remove short-term fluctuations:
    opt_lags = np.round(savgol_filter(opt_lags, mwo_lag_width, mwo_lag_degree))
    # - remove bad flux values [table seems to use value 4900] and entries without a date:
    igd = (np.abs(b_spt) > 0) & (np.abs(b_spt) < 4800) & (date_spt != '')
    date_spt = date_spt[igd]
    time_spt = time_spt[igd]
    lat_spt = lat_spt[igd]
    lon_spt = lon_spt[igd]
    b_spt = -b_spt[igd]  # note: polarities in MWO file seem to be opposite.
    # - get datetime objects for spots:
    t_spots = []
    for k, d in enumerate(date_spt):
        x = time_spt[k].split(':')
        if x[0] == '':
            x = [0, 0, 0]
        dt1 = datetime.datetime.strptime(d, '%Y-%m-%d')
        dt1 = dt1 + datetime.timedelta(hours=int(x[0]), minutes=int(x[1]), seconds=int(x[2]))  # (add hours + deal with obsns after midnight)
        t_spots.append( dt1 )

    return t_spots, lat_spt, lon_spt, b_spt, opt_lags, crs_lags