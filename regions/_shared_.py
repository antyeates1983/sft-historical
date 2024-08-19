"""
Calibrate and illustrate process of extraction magnetic regions from Ca K maps + sunspot measurements, using observed magnetograms as ground truth.

ARY 2024-Jul
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.io import FortranFile
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
import astropy.units as u
from sunpy.coordinates.sun import carrington_rotation_time
from params import *
sys.path.append('../_data_')
import _data_cak_mwo_
import _data_nso_
from _utils_ import toYearFraction
from paths import datapath_cak, datapath_nso

#--------------------------------------------------------------------------------------------------------------
def get_threshold(ncar, cr_start, cr_end, nlow, iboth, crs, map_br, map_cak, plot=True):
    """
    Get threshold in Ca-K corresponding to chosen level "br_min" in magnetograms.
    """

    # Compute average percentage of pixels over br_min:
    pfrac = np.zeros(ncar)
    for cr in range(cr_start, cr_end+1):
        k = cr - cr_start
        pfrac[k] = np.sum(np.abs(map_br[k,:,:]) > br_min)/nlow
    pfrac = np.array(pfrac)
    print('FRACTION OF PIXELS > %gG = %g' % (br_min, np.mean(pfrac[iboth])) )  # only include maps with good br

    # Determine corresponding threshold in Ca K:
    nthresh = 128
    ca_thresholds = np.linspace(0.8, 1.6, nthresh)
    over_pixels = np.zeros((nthresh, ncar))
    for cr in range(cr_start, cr_end+1):
        k = cr - cr_start
        for j in range(nthresh):
            over_pixels[j, k] = np.sum(map_cak[k,:,:] > ca_thresholds[j])/nlow
    cafrac = np.zeros(nthresh)
    for j in range(nthresh):
        cafrac[j] = np.mean(over_pixels[j, iboth], axis=1)
    pfrac0 =  np.mean(pfrac[iboth])
    imin = np.argmin(np.abs(cafrac - pfrac0))
    ca_opt = ca_thresholds[imin]
    print('Ca K THRESHOLD = %g' % ca_opt)

    if plot:
        # Plot cumulative distribution [for comparison with earlier papers]:
        fig = plt.figure(figsize=(5,4), tight_layout=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.subplot(211)
        plt.plot(ca_thresholds, cafrac, '-', color='k')
        plt.plot(ca_thresholds, ca_thresholds*0 + pfrac0, linestyle='--', color='tab:red')
        plt.plot([ca_opt, ca_opt], [0, 1], '--k')
        plt.xlim(0.8, 1.6)
        plt.ylim(0,1)
        plt.xlabel('Normalised Ca-K threshold')
        plt.ylabel('Fraction above threshold')
        plt.title('(a) Ca-K threshold = %g' % ca_opt)
        plt.text(0.82,0.07,'%g' % pfrac0, color='tab:red')

        plt.subplot(212)
        # fraction of pixels selected (only calculate when both maps are good):
        ncas = np.zeros(ncar) + np.nan
        nbrs = np.zeros(ncar) + np.nan
        for cr in range(cr_start, cr_end+1):
            k = cr - cr_start
            if np.isin(k,iboth):
                nca = np.sum(map_cak[k,:,:] > ca_opt)
                nbr = np.sum(np.abs(map_br[k,:,:]) > br_min)
                ncas[k] = nca
                nbrs[k] = nbr
        plt.plot(crs, ncas, '-', color='k', label='Ca-K')
        plt.plot(crs, nbrs, '-', color='tab:red', label='B_r')
        plt.xlim(crs[0], crs[-1])
        plt.xlabel('Carrington Rotation')
        plt.ylabel('Pixels above threshold')
        plt.legend()
        plt.title('(b)')

        plt.savefig('nso/cak-threshold.pdf')
        plt.close()

    return np.round(ca_opt, decimals=5)   # [rounded for consistency with earlier version]

#--------------------------------------------------------------------------------------------------------------
def get_plages(cr, ca_opt, cak, br, plot=False):
    """
    For given Carrington rotation, return array of plages.
    If plot==True then plot comparison with the synoptic magnetogram. 
    """
    # Find plages that exceed threshold:
    regs, _, pregs = _data_cak_mwo_.get_plages(_data_cak_mwo_.readmap, cr, ns, nph, cmin=ca_opt, plots=False, outpath='', datapath=datapath_cak, min_pix=npix_min, dilation_size=0)

    if plot:
        # Find all plages (for plotting):
        regs0, _, _ = _data_cak_mwo_.get_plages(_data_cak_mwo_.readmap, cr, ns, nph, cmin=ca_opt, plots=False, outpath='', datapath=datapath_cak, min_pix=0, dilation_size=0)

        # Limit to single cover:
        if len(regs0) > 0:
            regs0 = regs0[:,:,:nph] + regs0[:, :, nph:2*nph] + regs0[:, :, 2*nph:]
        if len(regs) > 0:
            regs1 = regs[:,:,:nph] + regs[:, :, nph:2*nph] + regs[:, :, 2*nph:]
        else:
            regs1 = []

        # Make map of colour-coded plages:
        balmap = br*0
        # - colour small plages:
        for reg in regs0:
            balmap[reg > 0]= 0.91
        # - colour large plages according to whether flux-balanced or not:
        for reg in regs1:
            fpos = np.sum(br[(reg > 0) & (br > 0)])
            fneg = np.sum(br[(reg > 0) & (br < 0)])
            if np.abs((fpos + fneg)/(fpos - fneg)) < max_unbalance:
                balmap[reg > 0]= 1
            else:
                balmap[reg > 0]= -1

        # Coordinate arrays for plotting:
        ds = 2.0/ns
        dph = 360.0/nph
        sc = np.linspace(-1+0.5*ds, 1-0.5*ds, ns)
        pc = np.linspace(0.5*dph, 360-0.5*dph, nph)

        fig = plt.figure(figsize=(10,6), tight_layout=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.subplot(321)
        plt.pcolormesh(pc, sc, cak, cmap='viridis', vmin=0.8, vmax=1.6, rasterized=True)
        plt.xlim(0, 360)
        plt.ylim(-np.sin(max_lat), np.sin(max_lat))
        plt.title('(a) CR%4.4i - Ca II K Normalized Intensity' % cr )
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(322)
        m = cak.copy()
        m[m < ca_opt] = 0
        plt.pcolormesh(pc, sc, m, cmap='viridis', vmin=0.8, vmax=1.6, rasterized=True)
        plt.xlim(0, 360)
        plt.ylim(-np.sin(max_lat), np.sin(max_lat))
        plt.title(r'(b) Ca $>$ %g' % np.round(ca_opt, decimals=3))
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(323)
        plt.pcolormesh(pc, sc, br, cmap='bwr', vmin=-50, vmax=50, rasterized=True)
        plt.xlim(0, 360)
        plt.ylim(-np.sin(max_lat), np.sin(max_lat))
        plt.title(r'(c) CR%4.4i - NSO $B_r$' % cr )
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(324)
        b = br.copy()
        b[np.abs(b) < br_min] = 0
        plt.pcolormesh(pc, sc, b, cmap='bwr', vmin=-50, vmax=50, rasterized=True)
        plt.xlim(0, 360)
        plt.ylim(-np.sin(max_lat), np.sin(max_lat))
        plt.title(r'(d) NSO $|B_r|>$ %g G' % br_min)
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(325)
        # Make overlap map:
        omap = (np.abs(b) > 0)*0.91
        omap[(b == 0) & (m > 0)] = 0.2
        omap[(np.abs(b) > 0) & (m > 0)] = 1
        plt.pcolormesh(pc, sc, omap, cmap='gist_stern_r', rasterized=True)
        plt.xlim(0, 360)
        plt.ylim(-np.sin(max_lat), np.sin(max_lat))
        plt.title(r'(e) Overlap')
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(326)
        # Show which plages are balanced before/after filtering:
        plt.pcolormesh(pc, sc, balmap, cmap='gist_stern_r', rasterized=True, vmin=-1, vmax=1)
        plt.pcolormesh(pc, sc, balmap, cmap='gist_stern_r', rasterized=True, vmin=-1, vmax=1)
        plt.xlim(0, 360)
        plt.ylim(-np.sin(max_lat), np.sin(max_lat))
        plt.title(r'(f) Flux Balance and Filtering')
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.savefig('nso/plages/plages_cr%4.4i.pdf' % cr)
        plt.close()

    return regs, pregs

#--------------------------------------------------------------------------------------------------------------
def get_areas(regs, br):
    """
    Return list of individual plage areas and observed fluxes, for given array of plages.
    """

    areas1 = []
    fluxes1 = []
    if len(regs) > 0:
        # - Limit to single cover:
        regs1 = regs[:,:,:nph] + regs[:, :, nph:2*nph] + regs[:, :, 2*nph:]
        # - Record area of plages and their unsigned fluxes from NSO:
        for reg in regs1:
            areas1.append( np.sum(reg > 0) )
            fluxes1.append( np.sum(np.abs(br[reg > 0])) )

    return areas1, fluxes1

#--------------------------------------------------------------------------------------------------------------
def flux_calibration(plage_area, plage_flux):
    """
    Scatterplot of observed magnetic flux against plage size.
    Compute linear fit parameters for mean and standard deviation.
    """

    plage_area = np.array(plage_area)
    plage_flux = np.array(plage_flux)

    fig = plt.figure(figsize=(5,3), tight_layout=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ds = 2.0/ns
    dph = 360.0/nph
    apix = ds*np.deg2rad(dph) * 6.96e10**2
    print('apix = %g' % apix)
    plt.scatter(plage_area, plage_flux * apix, 1, color='k')
    plt.xlabel('N Pixels')
    plt.ylabel('Unsigned Flux [Mx]')
    plt.ylim(0, 1e23)
    plt.xlim(0, 800)

    # Fit mean:
    popt, pcov = curve_fit(lambda x, a, b: a*x + b, plage_area, plage_flux)
    xs = np.linspace(0, 800, 2)
    plt.plot(xs, apix*(popt[0]*xs + popt[1]), 'k--', linewidth=1)
    plt.text(380, 0.85e23, (r'$\Phi=($%4.1f$N -$ %6.1f$)\mathrm{G}\,A_{\rm pix}$' % (popt[0], np.abs(popt[1]))))
    print('popt=', popt)
    print('flux = (%g * N + %g) * Ap' % (popt[0], popt[1]))

    # Compute residuals in bins:
    residuals = plage_flux - (popt[0]*plage_area + popt[1])
    bins = np.array([50, 75, 100, 125, 150,175, 200, 250, 300, 350, 400, 500])
    binc = 0.5*(bins[:-1] + bins[1:])
    nbins = len(bins)-1
    sdbin = np.zeros(nbins)
    for k in range(nbins):
        sdbin[k] = np.std(residuals[(plage_area >= bins[k]) & (plage_area < bins[k+1])])

    # Hence fit stddevs with straight line:
    popt_sd, pcov_sd = curve_fit(lambda x, a, b: a*x + b, binc[~np.isnan(sdbin)], sdbin[~np.isnan(sdbin)])
    print('popt_sd=', popt_sd)
    plt.plot(binc, apix*((popt[0]*binc + popt[1]) + sdbin), '+k', alpha=0.4)
    plt.plot(binc, apix*((popt[0]*binc + popt[1]) - sdbin), '+k', alpha=0.4)
    plt.plot(xs, apix*(popt[0]*xs + popt[1] + popt_sd[0]*xs + popt_sd[1]), 'k--', alpha=0.4, linewidth=1)
    plt.plot(xs, apix*(popt[0]*xs + popt[1] - popt_sd[0]*xs - popt_sd[1]), 'k--', alpha=0.4, linewidth=1)
    print('sd = (%g * N + %g) * Ap' % (popt_sd[0], popt_sd[1]))

    plt.xlim(50,800)
    plt.tight_layout()
    plt.savefig('nso/fluxfit-nso.pdf')
    plt.close()

#--------------------------------------------------------------------------------------------------------------
def get_polarities(regs, cr, ca_opt, t_spots, lat_spt, lon_spt, b_spt, opt_lags, crs_lags, lcycle_interp, rng=[], plot=False, hale=False):
    """
    For given Carrington rotation, return array of plages with polarities.
    Set rng to a random number generator to randomly flip polarities of some incomplete regions.
    Set plot=True to plot comparison with the synoptic magnetogram [only works if NSO magnetogram available]
    Set hale=True to treat all polarities as incomplete and ignore sunspot data.
    """
    
    # Coordinate arrays:
    ds = 2.0/ns
    dph = 360.0/nph
    sc = np.linspace(-1+0.5*ds, 1-0.5*ds, ns)
    pc3 = np.linspace(-360+0.5*dph, 720 - 0.5*dph, 3*nph)
    sc32, pc32 = np.meshgrid(sc, pc3, indexing='ij')

    if plot:
        brc, brl, brr = _data_nso_.readmap(cr, ns, nph, smooth=0, datapath=datapath_nso)
        br = np.concatenate((brl, brc, brr), axis=1)

    regs_pol = regs*0
    if not(hale):
        # Find all spots during a map and neighbours:
        # - central map:
        t_cm0 = carrington_rotation_time(cr, longitude=np.deg2rad(360)*u.rad)
        ispt0 = np.where(t_spots > t_cm0)[0][0]
        t_cm1 = carrington_rotation_time(cr+1, longitude=np.deg2rad(360)*u.rad)
        ispt1 = np.where(t_spots < t_cm1)[0][-1]
        ispt_cm = np.arange(ispt0, ispt1+1)
        # - map to right:
        t_cmr = carrington_rotation_time(cr-1, longitude=np.deg2rad(360)*u.rad)
        isptr = np.where(t_spots > t_cmr)[0][0]
        ispt_cmr = np.arange(isptr, ispt0)
        # - map to left:
        t_cml = carrington_rotation_time(cr+2, longitude=np.deg2rad(360)*u.rad)
        isptl = np.where(t_spots < t_cml)[0][-1]
        ispt_cml = np.arange(ispt1+1, isptl+1)
        # - smaller arrays:
        slat_spts = np.concatenate((np.sin(lat_spt[ispt_cm]), np.sin(lat_spt[ispt_cmr]), np.sin(lat_spt[ispt_cml])), axis=0)

        # Apply longitude correction:
        lon_spt_a = (lon_spt[ispt_cm] + np.deg2rad(opt_lags[crs_lags == cr]) + 2*np.pi) % (2*np.pi)
        lon_spt_b = (lon_spt[ispt_cmr] + np.deg2rad(opt_lags[crs_lags == cr]) + 2*np.pi) % (2*np.pi)
        lon_spt_c = (lon_spt[ispt_cml] + np.deg2rad(opt_lags[crs_lags == cr]) + 2*np.pi) % (2*np.pi)
        lon3_spts = np.concatenate((np.rad2deg(lon_spt_a), np.rad2deg(lon_spt_b) + 360, np.rad2deg(lon_spt_c)-360), axis=0)
        b_spts = np.concatenate((b_spt[ispt_cm], b_spt[ispt_cmr], b_spt[ispt_cml]), axis=0)
        pol_spts = b_spts/np.abs(b_spts)

        # Map of pixel polarities using sunspot observations:
        # - form KD tree:
        points = np.stack((np.rad2deg(np.arcsin(slat_spts)), lon3_spts), axis=1)
        tree = KDTree(points)
        # - map of nearest neighbours (within max distance):
        map_spts = np.zeros((ns, nph*3))
        for i in range(ns):
            for j in range(3*nph):
                dist, nbr = tree.query([np.rad2deg(np.arcsin(sc[i])), pc3[j]], k=mwo_spot_polarity_k, distance_upper_bound=mwo_spot_polarity_alpha)
                try:
                    map_spts[i,j] = np.mean(pol_spts[nbr])
                    map_spts[i,j] /= np.abs(map_spts[i,j])
                except:
                    pass

        # Assign polarities to regions:
        for j, reg in enumerate(regs):
            regs_pol[j][reg > 0] = map_spts[reg > 0]

    if plot:
        # Fraction of pixels with polarity determined:
        pol_frac = np.sum(np.abs(np.sum(regs_pol, axis=0)) > 0)/np.sum(np.sum(regs, axis=0) > 0)

        # Map of correct (+1) and incorrect (-1) polarities [before filling], compared to magnetogram:
        map_correct = br*0
        map_correct[np.sum(regs_pol, axis=0)*br > 0.5] = 1
        map_correct[np.sum(regs_pol, axis=0)*br < -0.5] = -1
        correct_frac = np.sum(map_correct > 0) / np.sum(np.abs(np.sum(regs, axis=0)) > 0)

    # Identify "complete" regions (minimum fraction of pixels have polarity determined):
    nregs = np.size(regs, axis=0)
    complete = np.zeros(nregs)
    pol_fracs = np.zeros(nregs)
    reg_cmplt = np.zeros((ns, 3*nph))
    regs_pol_fill = regs_pol.copy()
    for m, reg in enumerate(regs):
        pol_frac1 = np.sum(np.abs(regs_pol[m]) > 0)/np.sum(regs[m] > 0)
        pol_fracs[m] = pol_frac1
        fpos = np.sum(regs_pol[m] > 0)
        fneg = np.sum(regs_pol[m] < 0)
        if (fpos*fneg > 0) & (pol_frac1 > mwo_spot_complete_minfrac):
            reg_cmplt[reg > 0] = 1  # "complete" regions
            complete[m] = True
        else:
            reg_cmplt[reg > 0] = -1 # "incomplete" regions
            complete[m] = False
            # Throw away any polarity info and populate polarities using Hale's Law:-
            regs_pol_fill[m,:,:] = 0
            # - get correct polarity:
            sspot = np.mean(sc32[reg > 0])
            pspot = np.mean((pc32[reg > 0] + 360) % 360)
            tspot = carrington_rotation_time(cr, longitude=np.deg2rad(pspot)*u.rad)
            tspot = datetime.datetime.strptime(tspot.strftime('%Y %m %d %H'), '%Y %m %d %H')
            yspot = toYearFraction(tspot)
            hale_lpolarity_north = int(-1 + 2*(lcycle_interp(yspot, sspot) % 2))   # this is polarity of a leading NH spot
            # - populate east- and west-most pixels:
            regs_pol_fill[m][(pc32 == np.min(pc32[regs[m] > 0])) & (reg > 0)] = -hale_lpolarity_north
            regs_pol_fill[m][(pc32 == np.max(pc32[regs[m] > 0])) & (reg > 0)] = hale_lpolarity_north
            if (np.mean(sc32[reg > 0]) < 0):
                regs_pol_fill[m] *= -1
            if rng != []:
                # - randomly flip polarities of 5% of regions:
                prob = rng.random(1)
                if (prob < incomplete_flip_threshold):
                    regs_pol_fill[m] *= -1 
        # Fill remaining pixels using nearest-neighbour within region:-
        if complete[m] == 1:
            latrange = 90  # both latitude and longitude for n-n
        else:
            latrange = 0  # only longitude for n-n
        points = np.stack((sc32[np.abs(regs_pol_fill[m]) > 0]*latrange, pc32[np.abs(regs_pol_fill[m]) > 0]), axis=1)
        pols = regs_pol_fill[m][np.abs(regs_pol_fill[m]) > 0]
        tree = KDTree(points)
        for i in range(ns):
            for j in range(3*nph):
                if (reg[i,j] > 0) & (regs_pol_fill[m,i,j] == 0):
                    dist, nbr = tree.query([sc[i]*latrange, pc3[j]], k=1)
                    regs_pol_fill[m,i,j] = pols[nbr]
    
    # Set empty maps if there are no regions:
    if len(regs) == 0:
        regs = np.zeros((1, ns, nph*3))
        regs_pol = np.zeros((1, ns, nph*3))
        regs_pol_fill = np.zeros((1, ns, nph*3))

    if plot:
        # Map of correct (+1) and incorrect (-1) polarities [after filling], compared to magnetogram:
        map_correct_fill = br*0
        map_correct_fill[np.sum(regs_pol_fill, axis=0)*br > 0.5] = 1
        map_correct_fill[np.sum(regs_pol_fill, axis=0)*br < -0.5] = -1
        correct_frac_fill = np.sum(map_correct_fill > 0) / np.sum(np.sum(regs, axis=0) > 0)

        # Plot:
        fig = plt.figure(figsize=(10,6), tight_layout=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        plt.subplot(321)
        plt.pcolormesh(pc3, sc, map_spts, cmap='bwr', vmin=-5, vmax=5, rasterized=True)
        plt.scatter(lon3_spts, slat_spts, 0.5, pol_spts, cmap='bwr', vmin=-1, vmax=1)
        plt.plot([0, 0], [-1, 1], 'k--', linewidth=0.75)
        plt.plot([360, 360], [-1, 1], 'k--', linewidth=0.75)
        plt.ylim(-1,1)
        plt.xlim(-30,390)
        plt.title(('(a) CR%4.4i' % cr)+' - MWO Sunspot Polarity Map')
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(322)
        plt.pcolormesh(pc3, sc, np.sum(regs, axis=0), cmap='viridis', vmin=0.8, vmax=1.6, rasterized=True)
        plt.title('(b) Ca $>$ %g [%i plages]' % (np.round(ca_opt, decimals=3), len(regs)))
        plt.plot([0, 0], [-1, 1], 'w--', linewidth=0.75)
        plt.plot([360, 360], [-1, 1], 'w--', linewidth=0.75)
        plt.ylim(-1,1)
        plt.xlim(-30,390)
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(323)
        plt.pcolormesh(pc3, sc, np.sum(regs_pol, axis=0), cmap='bwr', vmin=-1, vmax=1, rasterized=True)
        binmap_cmplt = reg_cmplt > 0
        plt.contour(pc3, sc, binmap_cmplt, levels=[0.5], colors='c', linewidths=0.75)   
        binmap_incmplt = reg_cmplt < 0
        plt.contour(pc3, sc, binmap_incmplt, levels=[0.5], colors='m', linewidths=0.75)        
        plt.plot([0, 0], [-1, 1], 'k--', linewidth=0.75)
        plt.plot([360, 360], [-1, 1], 'k--', linewidth=0.75)
        plt.ylim(-1,1)
        plt.xlim(-30,390)
        plt.title((f'(c) Polarities from Spots ({pol_frac:.0%} of pixels, {correct_frac:.0%} correct)').replace('%', '\%'))
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(324)
        plt.pcolormesh(pc3, sc, br, cmap='bwr', vmin=-50, vmax=50, rasterized=True)
        plt.title(r'(d) NSO $B_r$')
        plt.plot([0, 0], [-1, 1], 'k--', linewidth=0.75)
        plt.plot([360, 360], [-1, 1], 'k--', linewidth=0.75)
        plt.xlim(-30,390)
        plt.ylim(-1,1)
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(325)
        plt.pcolormesh(pc3, sc, np.sum(regs_pol_fill, axis=0), cmap='bwr', vmin=-1, vmax=1, rasterized=True)
        plt.contour(pc3, sc, binmap_cmplt, levels=[0.5], colors='c', linewidths=0.75)   
        plt.contour(pc3, sc, binmap_incmplt, levels=[0.5], colors='m', linewidths=0.75)
        plt.plot([0, 0], [-1, 1], 'k--', linewidth=0.75)
        plt.plot([360, 360], [-1, 1], 'k--', linewidth=0.75)
        plt.ylim(-1,1)
        plt.xlim(-30,390)
        plt.title((f'(e) Final Polarities ({correct_frac_fill:.0%} correct)').replace('%', '\%'))
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.subplot(326)
        plt.pcolormesh(pc3, sc, br*(np.sum(regs, axis=0) > 0), cmap='bwr', vmin=-50, vmax=50, rasterized=True)
        plt.title(r'(f) NSO $B_r$ in Plages')
        plt.plot([0, 0], [-1, 1], 'k--', linewidth=0.75)
        plt.plot([360, 360], [-1, 1], 'k--', linewidth=0.75)
        plt.xlim(-30,390)
        plt.ylim(-1,1)
        plt.ylabel('Sine Latitude')
        plt.xlabel(r'Carrington Longitude [$^\circ$]')

        plt.tight_layout()
        plt.savefig('nso/polarities/polarities_cr%4.4i.pdf' % cr)

    return regs_pol_fill, complete, pol_fracs

#--------------------------------------------------------------------------------------------------------------
def set_flux(reg, reg_pol_fill, complete, flux_fit_mean1, flux_fit_mean0, flux_fit_sd1, flux_fit_sd0, rng=[]):
    """
    Assign flux to plage region.
    """

    # Assign flux based on plage area:
    plage_size = np.sum(reg > 0)
    if rng == []:
        # - deterministic:
        reg_pol_fill *= flux_fit_mean0/plage_size + flux_fit_mean1
    else:
        # - randomize based on distribution of standard deviations:
        reg_pol_fill *= rng.normal(flux_fit_mean1*plage_size + flux_fit_mean0, \
                                            flux_fit_sd1*plage_size + flux_fit_sd0, 1)/plage_size

    # Correct flux balance multiplicatively:
    r = reg_pol_fill
    fluxp = np.abs(np.sum(r[r > 0]))
    fluxn = np.abs(np.sum(r[r < 0]))
    fluxmn = 0.5*(fluxn + fluxp)
    reg_pol_fill[r < 0] *= fluxmn/fluxn
    reg_pol_fill[r > 0] *= fluxmn/fluxp

    # Reduce strength of Hale-filled regions to avoid overestimating dipole [with compromise of underestimating flux]:
    if complete == 0:
        reg_pol_fill /= incomplete_flux_scaling

    return reg_pol_fill

#--------------------------------------------------------------------------------------------------------------
def output_region(reg, reg_pol_fill, complete, pol_frac, preg, cr, outpath, plot=True):
    """
    Output magnetic region to Fortran unformatted file.
    Optionally, save plot of region.
    """

    pcen = preg % (2*np.pi)
    if(pcen == 0.0):
        pcen = 2*np.pi
    t_em = carrington_rotation_time(cr, longitude=pcen*u.rad)
    # - find region number (in case of multiple regions present at same time):
    ks = 1
    while (os.path.exists(outpath+'regions/r'+t_em.strftime('%Y%m%d.%H')+('_%4.4i.unf' % ks))):
        ks += 1
    # - save to file:
    fid = FortranFile(outpath+'regions/r'+t_em.strftime('%Y%m%d.%H')+('_%4.4i.unf' % ks), 'w')
    br1 = reg_pol_fill[:,:nph] + reg_pol_fill[:,nph:2*nph] + reg_pol_fill[:,2*nph:]
    plage = reg[:,:nph] + reg[:,nph:2*nph] + reg[:,2*nph:]
    fid.write_record(br1.astype(np.float64))
    fid.write_record(plage.astype(np.float64))
    fid.write_record(np.rad2deg(preg).astype(np.float64))
    fid.write_record(cr.astype(np.int16))
    fid.write_record(complete.astype(np.int8))
    fid.write_record(pol_frac.astype(np.float64))
    fid.close()

    if plot:
        ds = 2.0/ns
        dph = 360.0/nph
        sc = np.linspace(-1+0.5*ds, 1-0.5*ds, ns)
        pc = np.linspace(0.5*dph, 360-0.5*dph, nph)

        map_cak = _data_cak_mwo_.readmap(cr, ns, nph, datapath=datapath_cak)[0]

        plt.figure(figsize=(6,5))

        plt.subplot(211)
        plt.pcolormesh(pc, sc, map_cak, cmap='viridis', vmin=0.8, vmax=1.6)
        plt.ylim(-1,1)
        plt.xlim(0,360)
        plt.title('CR%4.4i ' % cr)

        plt.subplot(212)
        plt.pcolormesh(pc, sc, br1, cmap='bwr', vmin=-50, vmax=50)
        plt.title(('CR%4.4i ' % cr) + t_em.strftime('%Y%m%d.%H')+('_%4.4i' % ks))
        plt.ylim(-1,1)
        plt.xlim(0,360)

        plt.tight_layout()
        plt.savefig(outpath+'im-regions/im_r'+t_em.strftime('%Y%m%d.%H')+('_%4.4i.png' % ks), bbox_inches='tight')
        plt.close()
