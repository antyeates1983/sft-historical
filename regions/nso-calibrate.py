"""
Calibrate and illustrate process of extraction magnetic regions from Ca K maps + sunspot measurements, using observed magnetograms as ground truth.

ARY 2024-Nov
"""
import numpy as np
from params import *
from _shared_ import get_threshold, get_plages, get_areas
from _shared_ import flux_calibration, get_polarities
import datetime
import sys
sys.path.append('../_data_')
from _data_nso_ import prep_cak_br_maps
from _data_spots_mwo_ import prep_mwo_spots
from _data_spots_leussu_ import prep_leussu_spots

# Start and end of comparison period:
t_start = datetime.datetime(1975, 3, 1, 12)
t_end = datetime.datetime(1985, 7, 31, 12)

# Manually identified bad maps (to omit from comparison):
bad_br = [1625,1631,1632,1635,1639,1640,1641,1642,1643,1644,1647,1648,1650,1658,1661,1663,1665,1713]
bad_ca = [1625,1654,1663,1664,1665,1677,1679,1692,1693,1699,1700,1704,1715,1725,1726,1732,1756,1758,1764]

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Prepare observational data:
    ncar, cr_start, cr_end, nlow, iboth, crs, map_br, map_cak = prep_cak_br_maps(ns, nph, t_start, t_end, bad_br, bad_ca, max_lat)
    t_spots, lat_spt, lon_spt, b_spt, opt_lags, crs_lags = prep_mwo_spots(crs, mwo_lag_window_min, mwo_lag_window_max, mwo_lag_width, mwo_lag_degree)
    lcycle_interp = prep_leussu_spots()

    # Get threshold in Ca-K corresponding to global parameter br_min:
    ca_opt = get_threshold(ncar, cr_start, cr_end, nlow, iboth, crs, map_br, map_cak)

    # Identify plages and compute areas + magnetogram fluxes:
    areas, fluxes = [], []
    for cr in crs[iboth]:
        print(cr)
        k = np.argmin(np.abs(crs - cr))
        cak, br = map_cak[k,:,:], map_br[k,:,:]
        regs, _ = get_plages(cr, ca_opt, cak, br, plot=False)
        areas1, fluxes1 = get_areas(regs, br)
        areas += areas1
        fluxes += fluxes1

    # Get parameters calibrating flux to plage area:
    flux_calibration(areas, fluxes)

    # Plot plages and polarity assignment:
    for cr in [1685]:
        k = np.argmin(np.abs(crs - cr))
        cak, br = map_cak[k,:,:], map_br[k,:,:]
        regs, _ = get_plages(cr, ca_opt, cak, br, plot=True)
        regs_pol_fill = get_polarities(regs, cr, ca_opt, t_spots, lat_spt, lon_spt, b_spt, opt_lags, crs_lags, lcycle_interp, plot=True)
