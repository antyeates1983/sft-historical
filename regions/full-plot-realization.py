"""
Plot results for magnetic regions. Include butterfly diagrams for a single realization, and plot of fluxes for an ensemble of realizations.

ARY 2024-Nov
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import FortranFile
import astropy.units as u
from sunpy.coordinates.sun import carrington_rotation_time, carrington_rotation_number
import datetime
from params import ns, nph
sys.path.append('../_data_')
from _data_spots_leussu_ import prep_leussu_spots
from _data_ssa_mandal_ import prep_mandal_ssa
from _utils_ import toYearFraction

# Single realization to show in butterfly plots:
datadir1 = 'regions-full/regions-spots09/'

# Ensemble of realizations to show in flux plot:
# datadirs = ['regions-full/regions-spots%2.2i/' % j for j in range(0,20)]
datadirs = ['regions-full/regions-hale%2.2i/' % j for j in range(0,20)]

# Directory for plot:
plotdir = './'

#--------------------------------------------------------------------------------------------------------------
def get_regions(datadir):
    """
    Get time, sine-latitude, flux, leading flux and dipole moment data for all region in datadir.
    """

    rfiles = []
    for file in os.listdir(datadir+'regions/'):
        if file.startswith('r') and file.endswith('.unf'):
            rfiles.append(file)
    rfiles.sort()
    nreg = len(rfiles)

    ds = 2.0/ns
    dph = 2*np.pi/nph
    sc = np.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns)
    pc = np.linspace(0.5*dph, 2*np.pi - 0.5*dph, nph)
    sc2, pc2 = np.meshgrid(sc, pc, indexing='ij')

    t, slat = np.zeros(nreg), np.zeros(nreg)
    years, crs, cmps = np.zeros(nreg, dtype=int), np.zeros(nreg, dtype=int), np.zeros(nreg, dtype=int)
    flux, leadflux, dipole = np.zeros(nreg), np.zeros(nreg), np.zeros(nreg)

    for k, rfile in enumerate(rfiles):
        years[k] = int(rfile[1:5])
        t[k] = toYearFraction(datetime.datetime.strptime(rfile[1:12], '%Y%m%d.%H'))
        fid = FortranFile(datadir+'regions/'+rfile, 'r')
        br = fid.read_reals(dtype=np.float64).reshape((ns, nph))
        _ = fid.read_reals(dtype=np.float64).reshape((ns, nph))
        _ = fid.read_reals(dtype=np.float64) - 360
        crs[k] = fid.read_ints(dtype=np.int16)[0]
        cmps[k] = fid.read_ints(dtype=np.int8)[0]
        fid.close()
        # - centroid in sine-latitude:
        slat[k] = np.sum(sc2*np.abs(br))/np.sum(np.abs(br))
        # - unsigned flux:
        flux[k] = np.sum(np.abs(br)) * ds * dph * (6.96e10)**2
        # - dipole moment:
        dipole[k] = 0.75/np.pi * np.sum(br * sc2) * ds * dph 

        # Find signed flux of leading polarity:
        # - set unsigned flux:
        leadflux[k] = 0.5 * flux[k]
        # - positive and negative centroids in longitude:
        if ((np.min(pc2[np.abs(br)>0]) < 0.25*np.pi) & (np.max(pc2[np.abs(br)>0]) > 1.75*np.pi)):
            brc = np.roll(br, nph//2, axis=1)
            pp = pc[nph//2]
        else:
            brc = br
            pp = 0
        pcen = (np.sum(pc2[brc>0]*brc[brc>0])/np.sum(brc[brc > 0]) + pp) % (2*np.pi)
        ncen = (np.sum(pc2[brc<0]*brc[brc<0])/np.sum(brc[brc < 0]) + pp) % (2*np.pi)
        # - account for regions crossing edge:
        if (max([pcen, ncen] - min([pcen, ncen])) > np.pi):
            if (pcen > ncen):
                leadflux[k] *= -1
        else:
            if (ncen > pcen):
                leadflux[k] *= -1

    return t, years, slat, flux, leadflux, dipole, cmps
    
#--------------------------------------------------------------------------------------------------------------
def hale_joy_percentages(t, slat, leadflux, dipole, lcycle_interp):
    """
    Get percentage of regions obeying Hale's Law (expected leading polarity) and Joy's Law (expected sign of axial dipole strength), according to given solar cycles.
    """

    # Assign cycle to each region:
    cyc_ems = np.round(lcycle_interp(t, slat))

    hale_ems = leadflux*0 + 1
    hale_ems[(slat > 0) & ((cyc_ems % 2) == 0) & (leadflux > 0)] = -1
    hale_ems[(slat > 0) & ((cyc_ems % 2) == 1) & (leadflux < 0)] = -1
    hale_ems[(slat < 0) & ((cyc_ems % 2) == 0) & (leadflux < 0)] = -1
    hale_ems[(slat < 0) & ((cyc_ems % 2) == 1) & (leadflux > 0)] = -1
    hale_percent = np.sum(hale_ems > 0) / len(hale_ems)
    print('%i anti-Hale out of %i' % (np.sum(hale_ems < 0), len(hale_ems)))

    joy_ems = leadflux*0 + 1
    joy_ems[((cyc_ems % 2) == 0) & (dipole < 0)] = -1
    joy_ems[((cyc_ems % 2) == 1) & (dipole > 0)] = -1
    joy_percent = np.sum(joy_ems > 0) / len(joy_ems)
    print(joy_percent)

    return hale_percent, joy_percent

#--------------------------------------------------------------------------------------------------------------
def plot_butterfly_realization(gs, t, years, slat, flux, leadflux, dipole, lcycle_interp=''):
    """
    Make butterfly plots.
    """

    # Get year corresponding to start of magnetogram overlap period:
    t1 = carrington_rotation_time(1626, longitude=np.deg2rad(360)*u.rad)
    t1 = datetime.datetime.strptime(t1.strftime('%Y %m %d %H'), '%Y %m %d %H')
    t_olap_start = toYearFraction(t1)

    # Indices for sorting:
    lsrt = np.argsort(np.abs(leadflux))
    dsrt = np.argsort(np.abs(dipole))

    if lcycle_interp != '':
        # Make background shading for which cycle is which:
        ds = 2.0/ns
        t_cyc1 = np.linspace(1900, 2000, 1024)
        s_cyc1 = np.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns)
        t_cyc, s_cyc = np.meshgrid(t_cyc1, s_cyc1, indexing='ij')
        cyc = lcycle_interp(t_cyc, s_cyc)
        num_cyc = np.unique(cyc)
        tmid_cyc = num_cyc*0
        for i, c in enumerate(num_cyc):
            tmid_cyc[i] = np.mean(t_cyc[cyc == c])

        hale_percent, joy_percent = hale_joy_percentages(t, slat, leadflux, dipole, lcycle_interp)

    # (a) REGIONS COLOURED BY LEADING FLUX:
    # -------------------------------------
    ax = plt.subplot(gs[0,:-1])
    if lcycle_interp != '':
        plt.pcolormesh(t_cyc, s_cyc, cyc, cmap='PRGn', alpha=0.25, rasterized=True)
        for i, c in enumerate(num_cyc):
            if ((tmid_cyc[i]-3) > years[0]) & ((tmid_cyc[i]-3) < years[-1]):
                plt.text(tmid_cyc[i]-3, -0.9, 'SC%2.2i' % c)
    pm = plt.scatter(t[lsrt], slat[lsrt], 3, leadflux[lsrt], cmap='RdYlBu_r', vmin=-1e22, vmax=1e22, rasterized=True)
    plt.plot([t_olap_start, t_olap_start], [-1, 1], '--', linewidth=0.75, color='k')
    plt.xlabel('Year')
    plt.ylabel('Sine Latitude')
    plt.plot([np.min(t), np.max(t)], [0,0], color='k', linewidth=0.75)
    plt.xlim(years[0], years[-1])
    plt.ylim(-1, 1)
    if lcycle_interp != '':
        plt.title(f"(a) Leading Flux ({hale_percent:.1%} follow Hale's Law)".replace('%', '\%'))
    else:
        plt.title(f"(a) Leading Flux)")

    cbax = plt.subplot(gs[0,-1])
    cbar = plt.colorbar(pm, orientation="vertical", cax=cbax)
    cbar.set_label(r'$\pm\Phi$ [Mx]')

    # (b) REGIONS COLOURED BY AXIAL DIPOLE STRENGTH:
    # ----------------------------------------------
    ax2 = plt.subplot(gs[1,:-1])
    if lcycle_interp != '':
        plt.pcolormesh(t_cyc, s_cyc, cyc, cmap='PRGn', alpha=0.25, rasterized=True)
    pm2 = plt.scatter(t[dsrt], slat[dsrt], 3, dipole[dsrt], cmap='RdYlBu_r', vmin=-0.02, vmax=0.02, rasterized=True)
    plt.plot([t_olap_start, t_olap_start], [-1, 1], '--', linewidth=0.75, color='k')
    plt.xlabel('Year')
    plt.ylabel('Sine Latitude')
    plt.plot([np.min(t), np.max(t)], [0,0], color='k', linewidth=0.75)
    plt.xlim(years[0], years[-1])
    plt.ylim(-1, 1)
    if lcycle_interp != '':
        plt.title(f"(b) Axial Dipole Strength ({joy_percent:.1%} follow Joy's Law)".replace('%', '\%'))
    else:
        plt.title(f"(b) Axial Dipole Strength")        

    cbax = plt.subplot(gs[1,-1])
    cbar = plt.colorbar(pm2, orientation="vertical", cax=cbax)
    cbar.set_label(r'$b_{1,0}$ [G]')

#--------------------------------------------------------------------------------------------------------------
def get_flux_peryr(all_yrs, years, flux, cmps):
    """
    Get total annual fluxes in complete and incomplete plages.
    """
    flux_peryr_icmp = all_yrs*0.0
    for k, yrk in enumerate(years[cmps == 0]):
        flux_peryr_icmp[all_yrs == yrk] += flux[cmps == 0][k]
    flux_peryr_cmp = all_yrs*0.0
    for k, yrk in enumerate(years[cmps == 1]):
        flux_peryr_cmp[all_yrs == yrk] += flux[cmps == 1][k]

    return flux_peryr_cmp, flux_peryr_icmp

#--------------------------------------------------------------------------------------------------------------
def plot_flux_peryr(gs, all_yrs, flux_peryr_cmp, flux_peryr_icmp, t_ssam, areac_ssm):
    """
    Plot fluxes per year.
    """

    # Get year corresponding to start of magnetogram overlap period:
    t1 = carrington_rotation_time(1626, longitude=np.deg2rad(360)*u.rad)
    t1 = datetime.datetime.strptime(t1.strftime('%Y %m %d %H'), '%Y %m %d %H')
    t_olap_start = toYearFraction(t1)

    # Prep for second x-axis with CR numbers:
    crs1 = []
    for y in all_yrs:
        year = int(y)
        d = datetime.timedelta(days=(y - year)*365.0)
        day_one = datetime.datetime(year,1,1)
        date = d + day_one
        crs1.append(int(carrington_rotation_number(date)))
    all_crs = np.array(crs1)
    def yr2cr(x):
        return np.interp(x, all_yrs, all_crs)
    def cr2yr(x):
        return np.interp(x, all_crs, all_yrs)
    
    ax = plt.subplot(gs[2,:-1])
    for k in range(len(flux_peryr_cmp)):
        if k == 0:
            label0, label1, label2 = 'both', 'complete', 'incomplete'
        else:
            label0, label1, label2 = '', '', ''
        ax.plot(all_yrs, flux_peryr_cmp[k] + flux_peryr_icmp[k], '.-', linewidth=1, color='k', label=label0, alpha=0.25)
        ax.plot(all_yrs, flux_peryr_cmp[k], '.-', linewidth=1, color='c', label=label1, alpha=0.25)
        ax.plot(all_yrs, flux_peryr_icmp[k], '.-', linewidth=1, color='m', label=label2, alpha=0.25)
  
    ax.plot([t_olap_start, t_olap_start], [0, 6e24], '--', linewidth=0.75, color='k')

    ax.set_xlim(all_yrs[0], all_yrs[-1])
    ax.set_ylim(0,6e24)
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$\sum \Phi$ [Mx]')
    ax.set_title('(c) Annual Plage Magnetic Flux')
    ax.legend()

    ax2 = ax.twinx()
    ax2.fill_between(t_ssam, areac_ssam, facecolor='lightgray', label='Sunspot Area')
    ax2.set_ylabel(r'Sunspot Area [$\mu$Hem]', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 5000)

    secax = plt.gca().secondary_xaxis('bottom', functions=(yr2cr, cr2yr))
    secax.set_xticks([1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700])
    secax.set_xlabel('Carrington Rotation')
    secax.spines.bottom.set_position(("outward", 35))

    ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
    ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    lcycle_interp = prep_leussu_spots(yrmin=1900, yrmax=2000)
    t_ssam, areac_ssam = prep_mandal_ssa()
    
    fig = plt.figure(figsize=(10,7), tight_layout=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    gs = gridspec.GridSpec(3,12, height_ratios=[3,3,2])

    # Butterfly plots for single realization:
    t, years, slat, flux, leadflux, dipole, cmps = get_regions(datadir1)
    plot_butterfly_realization(gs, t, years, slat, flux, leadflux, dipole, lcycle_interp)

    # Plage fluxes per year for ensemble of realizations:
    n_rlztn = len(datadirs)
    all_yrs = np.unique(years)
    n_yrs = len(all_yrs)
    flux_peryr_cmp = []
    flux_peryr_icmp = []
    hale_percents = np.zeros(n_rlztn)
    joy_percents = np.zeros(n_rlztn)
    for k in range(n_rlztn):
        t, years, slat, flux, leadflux, dipole, cmps = get_regions(datadirs[k])
        flux_peryr_cmp1, flux_peryr_icmp1 = get_flux_peryr(all_yrs, years, flux, cmps)
        flux_peryr_cmp.append(flux_peryr_cmp1)
        flux_peryr_icmp.append(flux_peryr_icmp1)

        print(datadirs[k])
        hale_percents[k], joy_percents[k] = hale_joy_percentages(t, slat, leadflux, dipole, lcycle_interp)

    plot_flux_peryr(gs, all_yrs, flux_peryr_cmp, flux_peryr_icmp, t_ssam, areac_ssam)

    plt.tight_layout()
    # plt.savefig(plotdir+'/regions-bfly.pdf')

    sd_hale = np.std(hale_percents)
    sd_joy = np.std(joy_percents)
    mn_hale = np.mean(hale_percents)
    mn_joy = np.mean(joy_percents)
    print('Hale percentage: %6.3f +- %6.3f' % (mn_hale, sd_hale))
    print('Joy percentage: %6.3f +- %7.4f' % (mn_joy, sd_joy))