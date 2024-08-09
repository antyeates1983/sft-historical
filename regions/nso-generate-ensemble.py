"""
Generate ensemble of realizations of magnetic region dataset using sunspot data, with varying flux and certain fraction of polarities flipped for incomplete plages (i.e. those without sufficient sunspot measurements).

For each ensemble, output individual region files in required format for SFT code, along with image.

ARY 2024-Aug
"""
import os
import numpy as np
from params import *
from _shared_ import get_plages
from _shared_ import get_polarities, set_flux, output_region
import datetime
import sys
sys.path.append('../_data_')
from _data_nso_ import prep_cak_br_maps
from _data_spots_mwo_ import prep_mwo_spots
from _data_spots_leussu_ import prep_leussu_spots

# Time range:
t_start = datetime.datetime(1975, 3, 1, 12)
t_end = datetime.datetime(1985, 7, 31, 12)

# Number of realizations in ensemble:
n_rlztns = 10

# Threshold for plage regions:
ca_opt = 1.26614

# Parameters from area-to-flux calibration:
flux_fit_mean1 = 126.61
flux_fit_mean0 = -3160.47
flux_fit_sd1 = 33.1511198 
flux_fit_sd0 = -195.46127042

# Whether to omit any bad maps:
bad_ca = []

# Whether to use sunspot data:
use_spots = False

# Path for output data:
outdir = '/Users/bmjg46/Documents/stfc-historical/regions-nso/'

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    os.system('mkdir '+outdir)

    for irand in range(n_rlztns):
        if use_spots:
            outpath = outdir+'regions-spots%2.2i/' % irand
        else:
            outpath = outdir+'regions-hale%2.2i/' % irand
        os.system('mkdir '+outpath)

        # Seed random number generator repeatably:
        rng = np.random.default_rng(seed=irand)

        os.system('mkdir -p '+outpath.replace(' ', '\ ')+'/regions')
        os.system('mkdir -p '+outpath.replace(' ', '\ ')+'/im-regions')

        # Prepare observational data:
        ncar, cr_start, cr_end, nlow, iboth, crs, map_br, map_cak = prep_cak_br_maps(ns, nph, t_start, t_end, [], bad_ca, max_lat)
        lcycle_interp = prep_leussu_spots()
        if use_spots:
            t_spots, lat_spt, lon_spt, b_spt, opt_lags, crs_lags = prep_mwo_spots(crs, mwo_lag_window_min, mwo_lag_window_max, mwo_lag_width, mwo_lag_degree)

        # Identify plages and assign polarities:
        for cr in crs:
            print(cr)
            k = np.argmin(np.abs(crs - cr))
            cak = map_cak[k,:,:]
            regs, pregs = get_plages(cr, ca_opt, cak, '')
            if use_spots:
                regs_pol_fill, complete, pol_fracs = get_polarities(regs, cr, ca_opt, t_spots, lat_spt, lon_spt, b_spt, opt_lags, crs_lags, lcycle_interp, rng=rng)
            else:
                regs_pol_fill, complete, pol_fracs = get_polarities(regs, cr, ca_opt, [], [], [], [], [], [], lcycle_interp, rng=rng, hale=True)

            # Assign flux to each region, output and plot:
            for m, reg in enumerate(regs):
                reg_pol_fill = set_flux(reg, regs_pol_fill[m], complete[m], flux_fit_mean1, flux_fit_mean0, flux_fit_sd1, flux_fit_sd0, rng=rng)
                output_region(reg, reg_pol_fill, complete[m], pol_fracs[m], pregs[m], cr, outpath, plot=True)

        # Move manually-identified "bad" regions to a separate directory:
        os.system('mkdir -p '+outpath.replace(' ', '\ ')+'/bad-regions')
        os.system('mkdir -p '+outpath.replace(' ', '\ ')+'/bad-im-regions')
        with open('nso-bad-regions.txt', 'r') as f:
            data = f.read()
            bad_list = data.split('\n')
        for file in bad_list:
            if file != '':
                rfile = file[3:-3]+'unf'
                os.system('mv '+outpath+'regions/'+rfile+' '+outpath+'bad-regions/'+rfile)
                os.system('mv '+outpath+'im-regions/'+file+' '+outpath+'bad-im-regions/'+file)
