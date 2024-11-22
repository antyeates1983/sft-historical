"""
    Python routines for reading polar crown filament data from Xu+2021.
    https://ui.adsabs.harvard.edu/abs/2021ApJ...909...86X/abstract

    A Yeates 2024-Aug
"""
import numpy as np
import datetime
from sunpy.coordinates.sun import carrington_rotation_time
import astropy.units as u
from paths import datapath_filaments
from _utils_ import toYearFraction

#--------------------------------------------------------------------------------------------------------------
def get_filament_data(filename='xu-pcfs.txt'):
    """
    Read data for locations of polemost polar crown filaments From Xu+2021.
    """
    pcf_data = np.loadtxt(datapath_filaments+'xu-pcfs.txt')
    pcf_lat = pcf_data[:,1]
    pcf_yr = np.zeros(len(pcf_lat))
    for k, cr in enumerate(pcf_data[:,0]):
        tpcf = carrington_rotation_time(cr, longitude=np.deg2rad(180)*u.rad)
        tpcf = datetime.datetime.strptime(tpcf.strftime('%Y %m %d %H'), '%Y %m %d %H')
        pcf_yr[k] = toYearFraction(tpcf)
    
    return pcf_yr, pcf_lat