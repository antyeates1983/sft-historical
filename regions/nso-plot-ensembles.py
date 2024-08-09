"""
Plot results for ensemble of realizations of magnetic regions, for magnetogram overlap period.

ARY 2024-Aug
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
import datetime
import pickle
import os
from params import ns, nph, max_unbalance, max_lat
import sys
sys.path.append('../_data_')
from _data_nso_ import prep_cak_br_maps
from _utils_ import toYearFraction

# Parent directory containing individual realizations:
datapath = '/Users/bmjg46/Documents/stfc-historical/regions-nso/'

# Manually identified bad maps (to omit from comparison):
bad_br = [1625,1631,1632,1635,1639,1640,1641,1642,1643,1644,1647,1648,1650,1658,1661,1663,1665,1713]
bad_ca = [1625,1654,1663,1664,1665,1677,1679,1692,1693,1699,1700,1704,1715,1725,1726,1732,1756,1758,1764]

#--------------------------------------------------------------------------------------------------------------
def get_timerange(datadir):
    """
    Get first and last Carrington rotation numbers by looking at region file names.
    """

    flist = os.listdir(datadir+'/regions/')
    regfiles = []
    for f in flist:
        if f.endswith('.unf'):
            regfiles.append(f)
    regfiles.sort()
    t_start = datetime.datetime.strptime(regfiles[0][1:12], '%Y%m%d.%H')
    t_end = datetime.datetime.strptime(regfiles[-1][1:12], '%Y%m%d.%H')

    return t_start, t_end

#--------------------------------------------------------------------------------------------------------------
def get_correct_pix(datadir):
    """
    Return percentage of correct pixels with correct polarity for each rotation (omitting rotations with bad maps).
    """

    # Get list of plage regions:
    flist = os.listdir(datadir+'/regions/')
    regfiles = []
    for f in flist:
        if f.endswith('.unf'):
            regfiles.append(f)
    regfiles.sort()

    # Loop through each plage and make combined map of pixel polarities for each rotation:
    pols_plage = np.zeros((ncar, ns, nph))
    pols_plage_mwo = np.zeros((ncar, ns, nph))
    pols_nso = np.zeros((ncar, ns, nph))
    correct_pix = np.zeros(ncar)
    for regfile in regfiles:
        fid = FortranFile(datadir+'/regions/'+regfile, 'r')
        br_reg = fid.read_reals(dtype=np.float64).reshape((ns, nph))
        plage_reg = fid.read_reals(dtype=np.float64).reshape((ns, nph))
        pcen = fid.read_reals(dtype=np.float64) - 360
        cr = fid.read_ints(dtype=np.int16)
        complete = fid.read_ints(dtype=bool)
        fid.close()

        if np.isin(cr, bad_br) | np.isin(cr, bad_ca):
            continue

        # Polarities (reconstructed and observed):
        pols_plage[cr-cr_start,:,:] += np.sign(br_reg)
        if (complete[0]):
            pols_plage_mwo[cr-cr_start,:,:] += np.sign(br_reg)
        pols_nso[cr-cr_start,:,:] += np.sign(map_br[crs==cr] * plage_reg)

    # Compute fraction of correct pixel polarities:
    for k in range(ncar):
        correct_pix[k] = np.sum((pols_plage[k,:,:] * pols_nso[k,:,:]) > 0) / np.sum(pols_plage[k,:,:] != 0)
    
    return correct_pix

#--------------------------------------------------------------------------------------------------------------
def plot_polarity_stats(correct_pixs, color='k', alpha=0.25, label='', fig='new', finish=True):
    """
    Plot percentage of correct polarity pixels in each Carrington rotation, for multiple realizations.
    """

    if fig == 'new':
        fig = plt.figure(figsize=(5,2.5), tight_layout=True)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    means = np.zeros(len(correct_pixs))
    for k, f in enumerate(correct_pixs):
        means[k] = np.mean(f[~np.isnan(f)])
    mean = np.mean(means)

    for k, f in enumerate(correct_pixs):
        if k==0:
            label += (f' ({mean:.1%})').replace('%', '\%')
        else:
            label = ''
        plt.plot(crs, f*100, label=label , color=color, linewidth=1, alpha=alpha)
    
    # Plot overall mean:
    plt.plot(crs, crs*0 + mean*100, color=color, alpha=1, linewidth=1, linestyle='--')

    if finish:
        plt.xlabel('Carrington Rotation')
        plt.ylabel('Percentage of Pixels')
        plt.legend(ncols=3, loc='lower center')
        plt.xlim(cr_start, cr_end)
        plt.ylim(0,100)
        plt.savefig('nso/polarity-stats.pdf')

    return fig

#--------------------------------------------------------------------------------------------------------------
def get_cumulative_flux_dipole(datadir):
    """
    Return cumulative fluxes and dipole moments for reconstructed regions and same regions in NSO magnetograms.
    For single realization.
    """

    # Coordinate arrays:
    ds = 2.0/ns
    dph = 2*np.pi/nph
    sc = np.linspace(-1+0.5*ds, 1-0.5*ds, ns)
    pc = np.linspace(0.5*dph, 2*np.pi-0.5*dph, nph)
    sc2, pc2 = np.meshgrid(sc, pc, indexing='ij')

    # Get list of plage regions:
    flist = os.listdir(datadir+'/regions/')
    regfiles = []
    for f in flist:
        if f.endswith('.unf'):
            regfiles.append(f)
    regfiles.sort()

    all_crs = np.linspace(cr_start, cr_end, ncar)

    # Compute flux and dipole strengths for each region (if not already saved):
    try:
        fid = open(datadir+'/fluxes-dipoles.pkl', 'rb')
        crs = pickle.load(fid)
        slat_regs = pickle.load(fid)
        leadflux_regs = pickle.load(fid)
        fluxes_regs = pickle.load(fid)
        dipoles_regs = pickle.load(fid)
        slat_nsos = pickle.load(fid)
        leadflux_nsos = pickle.load(fid)
        fluxes_nso = pickle.load(fid)
        dipoles_nso = pickle.load(fid)
        cmps = pickle.load(fid)
        t_regs = pickle.load(fid)
        fid.close()
    except:
        slat_regs = []
        leadflux_regs = []
        fluxes_regs = []
        dipoles_regs = []
        slat_nsos = []
        leadflux_nsos = []
        fluxes_nso = []
        dipoles_nso = []
        cmps = []
        crs = []
        t_regs = []

        cr_prv = 0
        for regfile in regfiles:
            fid = FortranFile(datadir+'/regions/'+regfile, 'r')
            br_reg = fid.read_reals(dtype=np.float64).reshape((ns, nph))
            plage_reg = fid.read_reals(dtype=np.float64).reshape((ns, nph))
            pcen = fid.read_reals(dtype=np.float64) - 360
            cr = fid.read_ints(dtype=np.int16)
            complete = fid.read_ints(dtype=np.int8)[0]
            fid.close()

            t_reg = toYearFraction(datetime.datetime.strptime(regfile[1:12], '%Y%m%d.%H'))

            # Centroid in sine(latitude):
            slat_reg =  np.sum(sc2*np.abs(br_reg))/np.sum(np.abs(br_reg))

            # Flux and dipole moment of reconstructed region:
            flux_reg = np.sum(np.abs(br_reg)) * ds * dph
            dipole_reg = 0.75/np.pi * np.sum(sc2 * br_reg) * ds * dph


            # Signed flux of leading polarity:
            # - set unsigned flux
            leadflux_reg = 0.5 * flux_reg
            # - positive and negative centroids in longitude:
            if ((np.min(pc2[np.abs(br_reg)>0]) < 0.25*np.pi) & (np.max(pc2[np.abs(br_reg)>0]) > 1.75*np.pi)):
                brc = np.roll(br_reg, nph//2, axis=1)
                pp = pc[nph//2]
            else:
                brc = br_reg
                pp = 0
            poscen = (np.sum(pc2[brc>0]*brc[brc>0])/np.sum(brc[brc > 0]) + pp) % (2*np.pi)
            negcen = (np.sum(pc2[brc<0]*brc[brc<0])/np.sum(brc[brc < 0]) + pp) % (2*np.pi)
            # - account for regions crossing edge:
            if (max([poscen, negcen] - min([poscen, negcen])) > np.pi):
                if (poscen > negcen):
                    leadflux_reg *= -1
            else:
                if (negcen > poscen):
                    leadflux_reg *= -1

            # Read in new synoptic magnetogram (if necessary):
            if (cr != cr_prv):
                k = np.argmin(np.abs(all_crs - cr))
                br_map = map_br[k,:,:]
                try:
                    br_mapl = map_br[k+1,:,:]
                except:
                    br_mapl = np.zeros((180,360))
                try:
                    br_mapr = map_br[k-1,:,:]
                except:
                    br_mapr = np.zeros((180,360))                    
                cr_prv = cr
            # - if region straddles boundary, need to replace part of magnetogram with neighbour:
            minlon, maxlon = np.min(np.rad2deg(pc2[plage_reg > 0])), np.max(np.rad2deg(pc2[plage_reg > 0]))
            br_nso = br_map.copy()
            if (minlon < 90) & (maxlon > 270):
                if (pcen < 180): # replace RH half of map with neighbour:
                    br_nso[:,nph//2:] = br_mapl[:,nph//2:].copy()
                else: # replace LH half of map:
                    br_nso[:,:nph//2] = br_mapr[:,:nph//2].copy()
            br_nso[plage_reg == 0] = 0

            slat_nso =  np.sum(sc2*np.abs(br_nso))/np.sum(np.abs(br_nso))

            # Balance flux:
            fluxp = np.abs(np.sum(br_nso[br_nso > 0]))
            fluxn = np.abs(np.sum(br_nso[br_nso < 0]))
            fluxmn = 0.5*(fluxn + fluxp)
            # If too unbalanced then set to zero (e.g. 1666):
            unbalance = np.abs(fluxp - fluxn)/(fluxp + fluxn)
            if (unbalance < max_unbalance):
                br_nso[br_nso < 0] *= fluxmn/fluxn
                br_nso[br_nso > 0] *= fluxmn/fluxp
            else:
                br_nso[:] = 0
            flux_nso = np.sum(np.abs(br_nso)) * ds * dph
            dipole_nso = 0.75/np.pi * np.sum(sc2 * br_nso) * ds * dph

            # Signed flux of leading polarity:
            # - set unsigned flux
            leadflux_nso = 0.5 * flux_nso
            if (leadflux_nso > 0):
                # - positive and negative centroids in longitude:
                if ((np.min(pc2[np.abs(br_nso)>0]) < 0.25*np.pi) & (np.max(pc2[np.abs(br_nso)>0]) > 1.75*np.pi)):
                    brc = np.roll(br_nso, nph//2, axis=1)
                    pp = pc[nph//2]
                else:
                    brc = br_nso
                    pp = 0
                poscen = (np.sum(pc2[brc>0]*brc[brc>0])/np.sum(brc[brc > 0]) + pp) % (2*np.pi)
                negcen = (np.sum(pc2[brc<0]*brc[brc<0])/np.sum(brc[brc < 0]) + pp) % (2*np.pi)
                # - account for regions crossing edge:
                if (max([poscen, negcen] - min([poscen, negcen])) > np.pi):
                    if (poscen > negcen):
                        leadflux_nso *= -1
                else:
                    if (negcen > poscen):
                        leadflux_nso *= -1
            
            # Add this region to output as long as both maps are good and NSO flux is sufficiently balanced:
            if (~np.isin(cr, bad_br) & ~np.isin(cr, bad_ca) & (flux_nso != 0)):
                slat_regs.append(slat_reg)
                leadflux_regs.append(leadflux_reg)
                fluxes_regs.append(flux_reg)
                dipoles_regs.append(dipole_reg)
                slat_nsos.append(slat_nso)
                leadflux_nsos.append(leadflux_nso)
                fluxes_nso.append(flux_nso)
                dipoles_nso.append(dipole_nso)
                crs.append(cr)
                t_regs.append(t_reg)
                if (complete == 0):
                    cmps.append(0)
                else:
                    cmps.append(1)       
            
        # Convert lists to numpy arrays and save to pickle file:
        crs = np.array(crs)
        slat_regs = np.array(slat_regs)
        leadflux_regs = np.array(leadflux_regs)
        fluxes_regs = np.array(fluxes_regs)
        dipoles_regs = np.array(dipoles_regs)
        slat_nsos = np.array(slat_nsos)
        leadflux_nsos = np.array(leadflux_nsos)
        fluxes_nso = np.array(fluxes_nso)
        dipoles_nso = np.array(dipoles_nso)
        cmps = np.array(cmps)
        t_regs = np.array(t_regs)

        fid = open(datadir+'/fluxes-dipoles.pkl', 'wb')
        pickle.dump(crs, fid)
        pickle.dump(slat_regs, fid)
        pickle.dump(leadflux_regs, fid)
        pickle.dump(fluxes_regs, fid)
        pickle.dump(dipoles_regs, fid)
        pickle.dump(slat_nsos, fid)
        pickle.dump(leadflux_nsos, fid)
        pickle.dump(fluxes_nso, fid)
        pickle.dump(dipoles_nso, fid)
        pickle.dump(cmps, fid)
        pickle.dump(t_regs, fid)
        fid.close()

    # Sum values within each Carrington rotation:
    fluxes_percr = all_crs * 0
    dipoles_percr = all_crs * 0
    fluxes_nso_percr = all_crs * 0
    dipoles_nso_percr = all_crs * 0
    for k, crk in enumerate(crs):
        fluxes_percr[all_crs == crk] += fluxes_regs[k]
        dipoles_percr[all_crs == crk] += dipoles_regs[k]
        fluxes_nso_percr[all_crs == crk] += fluxes_nso[k]
        dipoles_nso_percr[all_crs == crk] += dipoles_nso[k]

    # Cumulative sums:
    cum_flux = np.cumsum(fluxes_percr) * 6.96e10**2
    cum_flux_nso = np.cumsum(fluxes_nso_percr) * 6.96e10**2
    cum_dipole = np.cumsum(dipoles_percr)
    cum_dipole_nso = np.cumsum(dipoles_nso_percr)

    return cum_flux, cum_flux_nso, cum_dipole, cum_dipole_nso

#--------------------------------------------------------------------------------------------------------------
def plot_cumulative_flux_dipole(cum_fluxes_spots, cum_dipoles_spots, cum_fluxes_nso, cum_dipoles_nso, cum_fluxes_hale, cum_dipoles_hale, spotcolor='tab:red', halecolor='tab:blue', alpha=0.25):
    """
    Plot cumulative fluxes and dipole moments for reconstructed regions, for multiple realizations.
    Compare runs with spots to runs without spots [if available].
    """

    all_crs = np.linspace(cr_start, cr_end, ncar)

    fig = plt.figure(figsize=(5,4), tight_layout=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.subplot(221)
    # - plot results with spots:
    plt.plot(all_crs, cum_fluxes_nso[0], '--k', label='NSO', linewidth=1)
    for k, f in enumerate(cum_fluxes_spots):
        plt.plot(all_crs, f, linewidth=0.75, color=spotcolor, alpha=alpha)
    plt.plot(all_crs, cum_fluxes_nso[0], '--k', linewidth=1)
    plt.xlim(cr_start, cr_end)
    plt.ylim(0, 1.2e25)
    plt.xlabel('Carrington Rotation')
    plt.ylabel(r'$\sum \Phi$ [Mx]')
    plt.title('(a)')
    plt.legend()

    plt.subplot(222)
    plt.plot(all_crs, cum_dipoles_nso[0], '--k', label='NSO', linewidth=1)
    for k, f in enumerate(cum_dipoles_spots):
        plt.plot(all_crs, f, linewidth=0.75, color=spotcolor, alpha=alpha)
    plt.plot(all_crs, cum_dipoles_nso[0], '--k', linewidth=1)
    plt.legend()
    plt.xlim(cr_start, cr_end)
    plt.ylim(-5.5,0.5)
    plt.xlabel('Carrington Rotation')
    plt.ylabel(r'$\sum b_{1,0}$ [G]')
    plt.plot(all_crs, all_crs*0, 'k', linewidth=0.75)
    plt.title('(b)')

    if cum_fluxes_hale != []:
        plt.subplot(223)
        # - plot results with no spots:
        plt.plot(all_crs, cum_fluxes_nso[0], '--k', label='NSO', linewidth=1)
        for k, f in enumerate(cum_fluxes_hale):
            plt.plot(all_crs, f, linewidth=0.75, color=halecolor, alpha=alpha)
        plt.plot(all_crs, cum_fluxes_nso[0], '--k', linewidth=1)
        plt.xlim(cr_start, cr_end)
        plt.ylim(0, 1.2e25)
        plt.xlabel('Carrington Rotation')
        plt.ylabel(r'$\sum \Phi$ [Mx]')
        plt.title('(c)')
        plt.legend()

        plt.subplot(224)
        plt.plot(all_crs, cum_dipoles_nso[0], '--k', label='NSO', linewidth=1)
        for k, f in enumerate(cum_dipoles_hale):
            plt.plot(all_crs, f, linewidth=0.75, color=halecolor, alpha=alpha)
        plt.plot(all_crs, cum_dipoles_nso[0], '--k', linewidth=1)
        plt.legend()
        plt.xlim(cr_start, cr_end)
        plt.ylim(-5.5,0.5)
        plt.xlabel('Carrington Rotation')
        plt.ylabel(r'$\sum b_{1,0}$ [G]')
        plt.plot(all_crs, all_crs*0, 'k', linewidth=0.75)
        plt.title('(d)')

    plt.tight_layout()
    plt.savefig('nso/fluxes-dipoles.pdf')
    plt.show()


#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    dirlist1 = os.listdir(datapath)
    dirlist = []
    for f in dirlist1:
        if f.startswith('regions-'):
            dirlist.append(f)
    dirlist.sort()

    # Preparation:
    t_start, t_end = get_timerange(datapath+dirlist[0])
    ncar, cr_start, cr_end, nlow, iboth, crs, map_br, map_cak = prep_cak_br_maps(ns, nph, t_start, t_end, bad_br, bad_ca, max_lat)

    # Loop through each realization:
    correct_pixs_spots = []
    correct_pixs_hale = []
    cum_fluxes_spots, cum_dipoles_spots, cum_fluxes_nso, cum_dipoles_nso = [], [], [], []
    cum_fluxes_hale, cum_dipoles_hale = [], []

    for datadir in dirlist:
        print(datadir)

        # Fraction of correct pixel polarities:
        correct_pix = get_correct_pix(datapath+datadir)
        if datadir[:13] == 'regions-spots':
            correct_pixs_spots.append(correct_pix)
        elif datadir[:12] == 'regions-hale':
            correct_pixs_hale.append(correct_pix)

        # Cumulative fluxes and dipole moments:
        cum_flux, cum_flux_nso, cum_dipole, cum_dipole_nso = get_cumulative_flux_dipole(datapath+datadir)
        if datadir[:13] == 'regions-spots':
            cum_fluxes_spots.append(cum_flux)
            cum_dipoles_spots.append(cum_dipole)
            cum_fluxes_nso.append(cum_flux_nso)
            cum_dipoles_nso.append(cum_dipole_nso)
        elif datadir[:12] == 'regions-hale':
            cum_fluxes_hale.append(cum_flux)
            cum_dipoles_hale.append(cum_dipole)

    # Generate figures:
    fig = plot_polarity_stats(correct_pixs_spots, color='tab:red', label='with spots', finish=False)
    _ = plot_polarity_stats(correct_pixs_hale, color='tab:blue', label='all Hale', fig=fig, finish=True)
    plot_cumulative_flux_dipole(cum_fluxes_spots, cum_dipoles_spots, cum_fluxes_nso, cum_dipoles_nso, cum_fluxes_hale, cum_dipoles_hale)