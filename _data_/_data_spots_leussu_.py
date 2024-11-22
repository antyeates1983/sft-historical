"""
    Python tools for reading sunspot data from Leussu+2017.
    https://ui.adsabs.harvard.edu/abs/2017A%26A...599A.131L/abstract

    A Yeates 2024-Aug
"""
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from paths import datapath_leussu_spots

#--------------------------------------------------------------------------------------------------------------
def prep_leussu_spots(yrmin=1910, yrmax=1990):
    """
    Prepare Leussu sunspot data. [Used for labelling which cycle a region belongs to.]
    """
    # Determine Hale polarity map from Leussu+2016 spot data (who have assigned cycles already):
    data = np.loadtxt(datapath_leussu_spots+'leussu-spots.tsv', delimiter="\t", skiprows=54, usecols=[1,2,3])
    lspot_year = data[:,0]
    lspot_slat = np.sin(np.deg2rad(data[:,1]))
    lspot_cyc = data[:,2]
    lspot_slat = lspot_slat[(lspot_year > yrmin) & (lspot_year < yrmax)]
    lspot_cyc = lspot_cyc[(lspot_year > yrmin) & (lspot_year < yrmax)]
    lspot_year = lspot_year[(lspot_year > yrmin) & (lspot_year < yrmax)]
    # - make nearest-neighbour interpolator from (year, s) -> cycle number:
    lspot_x = np.stack((lspot_year, lspot_slat), axis=1)
    lcycle_interp = NearestNDInterpolator(lspot_x, lspot_cyc, rescale=True)

    return lcycle_interp