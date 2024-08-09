"""
    Plot polar field and axial dipole of optimal SFT run within each of several realizations of the plage ensemble. Highlight the overall optimum.

    Also plot a magnetic butterfly diagram for the overall optimum run, with filament locations overlaid.

    A Yeates - Aug 2024
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from _sft_ import sftrun, polar_fluxes
from run_optim_sft import steady_state, groundTruth_pf, ns, nph, datapath_polar
import sys
sys.path.append('../_data_/')
from _utils_ import toYearFraction
from _data_spots_leussu_ import prep_leussu_spots
from _data_filaments_xu_ import get_filament_data

# List of plage data realizations (directories):
# outpaths = [('/Users/bmjg46/Documents/stfc-historical/regions-full/regions-hale%2.2i/' % i) for i in range(20)]
# outpaths = [('/Users/bmjg46/Documents/stfc-historical/regions-full/regions-hale%2.2i/' % i) for i in range(10)]
outpaths = [('/Users/bmjg46/Documents/stfc-historical/regions-full/regions-spots%2.2i/' % i) for i in range(10)]

# Specific optimization run filename:
# [optim[n]_[m].pkl, where n is #parameters and m is #runs]
optfile1 = 'optim4_10000.pkl'  

# Filename for plot:
plotdir = 'full'
# plotfile_bfly, plotfile_params = 'opt-bfly-hale20.pdf', 'opt-params-hale20.pdf'
# plotfile_bfly, plotfile_params = 'opt-bfly-hale10.pdf', 'opt-params-hale10.pdf'
plotfile_bfly, plotfile_params = 'opt-bfly-spots10.pdf', 'opt-params-spots10.pdf'

#--------------------------------------------------------------------------------------------------------------
def get_sft_optim_ensemble():
    """
    Read in results from ensemble of SFT optimizations (for different plage realizations), and for optimum run in each case save polar fluxes, axial dipole strength, and butterfly diagram, as well as value of cost function.
    """

    n_rlztn = len(outpaths)

    ts, pfs_nor, pfs_sou, dips, bflys = [], [], [], [], []
    obj_pfs = np.zeros(n_rlztn)
    for k, outpath in enumerate(outpaths):
        # Restore outputs from pickle file:
        picklefile = open(outpath+optfile1,'rb')
        eta0s = pickle.load(picklefile)
        v0s = pickle.load(picklefile)
        p0s = pickle.load(picklefile)
        taus = pickle.load(picklefile)
        bqs = pickle.load(picklefile)
        b0s = pickle.load(picklefile)
        _ = pickle.load(picklefile)
        obj_pf_un = pickle.load(picklefile)
        ns = pickle.load(picklefile)
        nph = pickle.load(picklefile)
        t_start = pickle.load(picklefile)
        t_end = pickle.load(picklefile)
        _ = pickle.load(picklefile)
        _ = pickle.load(picklefile)
        _ = pickle.load(picklefile)
        _ = pickle.load(picklefile)
        _ = pickle.load(picklefile)
        _ = pickle.load(picklefile)
        tfull = pickle.load(picklefile)
        picklefile.close()

        # Get optimum run:
        obj_pfs[k] = np.min(obj_pf_un)

        # Redo optimum run and get time series for polar fluxes and dipole:
        koptp = np.argmin(obj_pf_un)
        hr_end = (t_end - t_start).days * 24 + (t_end - t_start).seconds // 3600
        ds = 2.0/ns
        dph = 2*np.pi/nph
        sc = np.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns)
        pc = np.linspace(0.5*dph, 2*np.pi - 0.5*dph, nph)
        # - Analytical initial field (without normalization as this can vary):
        sc2, pc2 = np.meshgrid(sc, pc, indexing='ij')
        br0 = steady_state(sc2, v0s[koptp], eta0s[koptp], p0s[koptp])

        # Compile the fortran code:
        # [if you update your Fortran installation you may need to `make clean` first]
        if k == 0:
            os.system('make -C fortran')

        datesp, _, dipolep, bflyp, _, _  = sftrun(0, ns, nph, outpath, t_start, hr_end, br0, eta0s[koptp], v0s[koptp], p0s[koptp], taus[koptp], bqs[koptp], b0=b0s[koptp], dims=1, codepath='./')
        tfull = [toYearFraction(d) for d in datesp]
        t = np.array(tfull)
        t -= t[0]
        ts.append(tfull)
        dips.append(dipolep)
        bflys.append(bflyp)

        pf_nor_sim, pf_sou_sim = polar_fluxes(t, sc, bflyp)
        pfs_nor.append(pf_nor_sim)
        pfs_sou.append(pf_sou_sim)

    return ts, pfs_nor, pfs_sou, dips, obj_pfs, bflys

#--------------------------------------------------------------------------------------------------------------
def plot_timeseries(gs, ts, pfs_nor, pfs_sou, dips, iopt, pf_nor_obs, pf_sou_obs, err_nor_obs, err_sou_obs,  lcycle_interp='', yfmax=6e22, ydmax=7, ncolor='tab:red', scolor='tab:blue', dcolor='k'):
    """
    Plot polar fluxes and axial dipole against time.
    """

    ax = plt.subplot(gs[0,:-1])

    if lcycle_interp != '':
        # Background shading of solar cycles:
        # [define cycle numbers from median of Leussu data, cutting out high latitudes]
        ds = 2.0/ns
        sc = np.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns)
        t2, sc2 = np.meshgrid(ts[0], sc[np.abs(sc) < 0.75], indexing='ij')
        cyc2 = lcycle_interp(t2, sc2)
        cyc = np.median(cyc2, axis=1)
        num_cyc = np.unique(cyc)
        tmid_cyc = num_cyc*0
        for i, c in enumerate(num_cyc):
            tmid_cyc[i] = np.mean(t2[cyc2 == c])
        cyc = np.stack((cyc, cyc), axis=1)
        ax.pcolormesh(ts[0], [-yfmax, yfmax], cyc.T, cmap='PRGn', alpha=0.25, rasterized=True, vmin=np.min(cyc2), vmax=np.max(cyc2))
        ax.set_ylim(-yfmax, yfmax)
        for i, c in enumerate(num_cyc):
            if ((tmid_cyc[i]) > ts[0][0]) & ((tmid_cyc[i]) < ts[0][-1]):
                ax.text(tmid_cyc[i]-1, -0.9*yfmax, 'SC%2.2i' % c)

    # - observed polar field:
    pf_nor_obs_min = pf_nor_obs[::50] - err_nor_obs[::50]
    pf_nor_obs_max = pf_nor_obs[::50] + err_nor_obs[::50]
    pf_sou_obs_min = pf_sou_obs[::50] - err_sou_obs[::50]
    pf_sou_obs_max = pf_sou_obs[::50] + err_sou_obs[::50]
    ax.fill_between(ts[iopt][::50], pf_nor_obs_min, pf_nor_obs_max, color=ncolor, alpha=0.15)
    ax.fill_between(ts[iopt][::50], pf_sou_obs_min, pf_sou_obs_max, color=scolor, alpha=0.15)

    for k, _ in enumerate(ts):
        ax.plot(ts[k], pfs_nor[k], ncolor, linewidth=1, alpha=0.25)
        ax.plot(ts[k], pfs_sou[k], scolor, linewidth=1, alpha=0.25)

    ax.plot(ts[iopt], pfs_nor[iopt], ncolor, label='North', linewidth=1)
    ax.plot(ts[iopt], pfs_sou[iopt], scolor, label='South', linewidth=1)
    ax.plot([ts[iopt][0], ts[iopt][-1]], [0, 0], 'k', linewidth=0.75)
    ax.set_xlabel('Year')
    ax.set_ylabel('Polar flux [Mx]')
    ax.legend(loc='upper left', ncols=2)
    ax.set_xlim(ts[iopt][0], ts[iopt][-1])
    ax.set_title('(a)')

    ax = plt.subplot(gs[1,:-1])

    if lcycle_interp != '':
        ax.pcolormesh(ts[0], [-ydmax, ydmax], cyc.T, cmap='PRGn', alpha=0.25, rasterized=True, vmin=np.min(cyc2), vmax=np.max(cyc2))
        ax.set_ylim(-ydmax, ydmax)

    ax.set_ylim(-ydmax, ydmax)

    for k, _ in enumerate(ts):
        ax.plot(ts[k], dips[k], dcolor, linewidth=1, alpha=0.15)

    ax.plot(ts[iopt], dips[iopt], color=dcolor, linewidth=1)
    ax.plot([ts[iopt][0], ts[iopt][-1]], [0, 0], 'k', linewidth=0.75)
    ax.set_xlabel('Year')
    ax.set_ylabel(r'$b_{1,0}$ [G]')
    ax.set_xlim(ts[iopt][0], ts[iopt][-1])
    ax.set_title('(b)')
    
#--------------------------------------------------------------------------------------------------------------
def plot_butterfly(gs, t, bfly, pcf_yr='', pcf_lat=''):
    """
    Plot butterfly diagram for single run.
    """

    ds = 2.0/ns
    sc = np.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns)
    
    ax = plt.subplot(gs[2,:-1])
    bfmax = 8
    pm = ax.pcolormesh(t, sc, bfly, cmap='bwr', vmin=-bfmax, vmax=bfmax, rasterized=True)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Year')
    ax.set_ylabel('Sine Latitude')

    ax.set_xlim(t[0], t[-1])
    ax.set_title('(c)')

    # - colorbar:
    cbax = plt.subplot(gs[2,-1])
    cbar = plt.colorbar(pm, orientation="vertical", cax=cbax)
    cbar.set_label(r'$\langle B_r\rangle$ [G]')

    # - polar crown filaments:
    try:
        ax.plot(pcf_yr, np.sin(np.deg2rad(pcf_lat)), '*k')
    except:
        pass

    plt.savefig(plotdir+'/'+plotfile_bfly)
    plt.close()

#--------------------------------------------------------------------------------------------------------------
def plot_sft_params(datadir, cmap='Reds'):
    """
    Plot cost-function values projected on each parameter for all runs in a single SFT realization.
    """

    # Restore outputs from pickle file:
    picklefile = open(datadir+optfile1,'rb')
    eta0s = pickle.load(picklefile)
    v0s = pickle.load(picklefile)
    p0s = pickle.load(picklefile)
    taus = pickle.load(picklefile)
    bqs = pickle.load(picklefile)
    b0s = pickle.load(picklefile)
    obj_pf_wt = pickle.load(picklefile)
    obj_pf_un = pickle.load(picklefile)
    picklefile.close()
    obj_pf = obj_pf_un

    # Compute Rm for each run:
    RSUN = 6.96e5
    rsundu = v0s*(1+p0s)**(0.5*(1+p0s))/p0s**(0.5*p0s)
    rm = RSUN*rsundu/eta0s #RSUN*v0s/eta0s
    lam = np.rad2deg(1/np.sqrt(rm))  # dynamo effectivity range

    koptp = np.argmin(obj_pf)
    ymin = 0.9*min(obj_pf)
    ymax = 1.1*max(obj_pf)
    cols = np.abs(rm - rm[koptp])**0.5

    # sort so lowest lambda is on top
    isort = np.argsort(-cols)
    eta0s = eta0s[isort]
    v0s = v0s[isort]
    p0s = p0s[isort]
    b0s = b0s[isort]
    taus = taus[isort]
    obj_pf = obj_pf[isort]
    cols = cols[isort]
    lam = lam[isort]
    rm = rm[isort]
    koptp = np.argmin(obj_pf)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, axs = plt.subplots(1, 6, sharey=True, figsize=(18,3))

    fig.subplots_adjust(wspace=0.05, hspace=0.4)

    axs[0].scatter(eta0s, obj_pf, 2, cols, cmap=cmap, rasterized=True)
    axs[0].plot([eta0s[koptp], eta0s[koptp]], [ymin, ymax], linestyle='--',linewidth=1, color='k')
    # axs[0].text(370,0.925*ymax,r'$\eta_0 =$ %5.1f $\,\mathrm{km}\,\mathrm{s}^{-1}$' % eta0s[koptp])
    axs[0].set_title(r'(a) $\eta_0 =$ %5.1f $\,\mathrm{km}\,\mathrm{s}^{-1}$' % eta0s[koptp])
    axs[0].set_ylim(ymin, ymax)
    axs[0].set_xlabel(r'$\eta_0$ [$\mathrm{km}^2\mathrm{s}^{-1}$]')
    axs[0].set_ylabel(r'Polar Flux Error [$\mathrm{Mx}$]')
    # axs[0].set_title('(a)')

    axs[1].scatter(v0s*1e3, obj_pf, 2, cols, cmap=cmap, rasterized=True)
    axs[1].plot([v0s[koptp]*1e3,v0s[koptp]*1e3], [ymin, ymax], linestyle='--',linewidth=1, color='k')
    # axs[1].text(5,0.925*ymax,r'$v_0 =$ %5.2f $\,\mathrm{m}\,\mathrm{s}^{-1}$' % (v0s[koptp]*1e3))
    axs[1].set_title(r'(b) $v_0 =$ %5.2f $\,\mathrm{m}\,\mathrm{s}^{-1}$' % (v0s[koptp]*1e3))
    axs[1].set_ylim(ymin, ymax)
    axs[1].set_xlabel(r'$v_0$ [$\mathrm{m}\,\mathrm{s}^{-1}$]')
    # axs[1].set_title('(b)')

    axs[2].scatter(p0s, obj_pf, 2, cols, cmap=cmap, rasterized=True)
    axs[2].plot([p0s[koptp], p0s[koptp]], [ymin, ymax], linestyle='--',linewidth=1, color='k')
    # axs[2].text(2.2,0.925*ymax,r'$p_0 =$ %4.2f' % p0s[koptp])
    axs[2].set_title(r'(c) $p_0 =$ %4.2f' % p0s[koptp])
    axs[2].set_ylim(ymin, ymax)
    axs[2].set_xlabel(r'$p_0$')
    # axs[2].set_title('(c)')

    axs[3].scatter(b0s, obj_pf, 2, cols, cmap=cmap, rasterized=True)
    axs[3].plot([b0s[koptp], b0s[koptp]], [ymin, ymax], linestyle='--',linewidth=1, color='k')
    # axs[3].text(-10,0.925*ymax,r'$B_0 =$ %4.2f$\,\mathrm{G}$' % b0s[koptp])
    axs[3].set_title(r'(d) $B_0 =$ %4.2f$\,\mathrm{G}$' % b0s[koptp])
    axs[3].set_ylim(ymin, ymax)
    axs[3].set_xlabel(r'$B_0$ [$\mathrm{G}$]')
    # axs[3].set_title('(d)')

    axs[4].scatter(rm, obj_pf, 2, cols, cmap=cmap, rasterized=True)
    axs[4].plot([rm[koptp], rm[koptp]], [ymin, ymax], linestyle='--',linewidth=1, color='k')
    # axs[4].text(110,0.925*ymax,r'$\mathrm{Rm}_0 =$ %4.2f' % rm[koptp])
    axs[4].set_title(r'(e) $\mathrm{Rm}_0 =$ %4.2f' % rm[koptp])
    axs[4].set_ylim(ymin, ymax)
    axs[4].set_xlabel(r'$\mathrm{Rm}_0$')
    # axs[4].set_title('(e)')

    axs[5].scatter(lam, obj_pf, 2, cols, cmap='Reds', rasterized=True)
    axs[5].plot([lam[koptp], lam[koptp]], [ymin, ymax], linestyle='--',linewidth=1, color='k')
    # axs[5].text(110,0.925*ymax,r'$\lambda_R =$ %4.2f' % lam[koptp])
    axs[5].set_title(r'(f) $\lambda_R =$ %4.2f' % lam[koptp])
    axs[5].set_ylim(ymin, ymax)
    axs[5].set_xlabel(r'$\lambda_R$')
    # axs[5].set_title('(f)')

    plt.savefig(plotdir+'/'+plotfile_params, bbox_inches='tight')
    plt.close()

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    os.system('mkdir '+plotdir)

    # Get time series of polar fluxes and axial dipole for optimum SFT models across plage ensemble:
    ts, pfs_nor, pfs_sou, dips, obj_pfs, bflys = get_sft_optim_ensemble()
    
    # Get overall optimum:
    iopt = np.argmin(obj_pfs)

    # Read ground truth data (observed polar fluxes from faculae):
    pf_nor_obs, pf_sou_obs, err_nor_obs, err_sou_obs = groundTruth_pf(ts[iopt], ns, nph, filename='f_PFlux_MWO_WSO_MDI_2.0.dat', datapath_polar=datapath_polar)

    # Read polar crown filament locations for overplotting:
    pcf_yr, pcf_lat = get_filament_data()

    # Read Leussu+ sunspot data for labelling solar cycles:
    lcycle_interp = prep_leussu_spots(yrmin=1900, yrmax=2000)

    fig = plt.figure(figsize=(10,7), tight_layout=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    gs = gridspec.GridSpec(3,12, height_ratios=[3,2,3])
    
    plot_timeseries(gs, ts, pfs_nor, pfs_sou, dips, iopt, pf_nor_obs, pf_sou_obs, err_nor_obs, err_sou_obs, lcycle_interp=lcycle_interp)
    plot_butterfly(gs, ts[iopt], bflys[iopt], pcf_yr=pcf_yr, pcf_lat=pcf_lat)

    # Read all parameter values and objective function for the overall optimum run:
    plot_sft_params(outpaths[iopt])