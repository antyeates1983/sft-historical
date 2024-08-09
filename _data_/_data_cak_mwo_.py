"""
    Python tools for reading digitised Ca-K maps from MWO.
    
    A Yeates - Aug 2024
"""
import numpy as np
from astropy.io import fits
from scipy.io import netcdf
from scipy.interpolate import interp2d
from scipy.ndimage.measurements import label
from scipy.ndimage import grey_dilation
import matplotlib.pyplot as plt
import os
from sunpy.coordinates.sun import carrington_rotation_number
from paths import datapath_cak
from _utils_ import plgndr

#--------------------------------------------------------------------------------------------------------------
def prep_cak_maps(ns, nph, t_start, t_end, bad_ca, max_lat):
    """
    Prepare Ca K maps for full dataset.
    """
    
    # Create directory for plots etc:
    os.system('mkdir full')

    # Carrington rotation numbers:
    cr_start = int(carrington_rotation_number(t_start))
    cr_end = int(carrington_rotation_number(t_end))
    ncar = cr_end - cr_start + 1

    # Coordinates:
    ds = 2.0/ns
    dph = 360.0/nph
    sc = np.linspace(-1+0.5*ds, 1-0.5*ds, ns)
    pc = np.linspace(0.5*dph, 360-0.5*dph, nph)
    sc2, pc2 = np.meshgrid(sc, pc, indexing='ij')

    # Read Ca K maps and save to pickle file for more rapid reloading:
    try:
        fid = netcdf.netcdf_file('full/cak.nc', 'r', mmap=False)
        crs = fid.variables['cr'][:]
        map_cak = fid.variables['map_cak'][:]
        fid.close()
    except:
        fid = netcdf.netcdf_file('full/cak.nc', 'w')
        fid.createDimension('tdim', ncar)
        fid.createDimension('sdim', ns)
        fid.createDimension('pdim', nph)
        tid = fid.createVariable('cr', 'd', ('tdim',))
        cakid = fid.createVariable('map_cak', 'd', ('tdim','sdim','pdim'))
        for cr in range(cr_start, cr_end+1):
            print(cr)
            k = cr-cr_start
            map_cak0, _, _, wtsm = readmap(cr, ns, nph, smooth=0, datapath=datapath_cak, weights=False)
            tid[k] = cr
            cakid[k,:,:] = map_cak0
        fid.close()
        crs = tid[:]
        map_cak = cakid[:,:,:]

    # Set maps to zero at high latitudes:
    for k in range(0, cr_end-cr_start+1):
        map_cak[k][np.abs(sc2) > np.sin(max_lat)] = 0
    nlow = np.sum(np.abs(sc2) <= np.sin(max_lat))

    # Get indices to "good" maps:
    ica = np.ones(ncar, dtype=np.int16)
    for cr in range(cr_start, cr_end+1):
        k = cr - cr_start
        if np.isin(cr, bad_ca):
            ica[k] = 0
    ica = np.where(ica > 0)

    return ncar, cr_start, cr_end, nlow, ica, crs, map_cak

#--------------------------------------------------------------------------------------------------------------
def readmap(rot, ns, nph, smooth=0, datapath='./', weights=False):
    """
        Read Ca-K map for Carrington rotation rot.
        Also reads in the neighbouring maps.
        
        ARGUMENTS:
            rot is the number of the required Carrington rotation (e.g. 2190)
            ns and nph define the required grid (default 180 and 360)
            smooth [optional] controls the strength of smoothing (default 0 is no smoothing)
            weights -- if True then also read in map of weights
    """
    
    # READ IN DATA AND STITCH TOGETHER 3 ROTATIONS:
    # ---------------------------------------------
    fname = 'cr%4.4i.fits' % rot
    try:
        fid = fits.open(datapath+fname)
        brm = fid[0].data[0,:,:]
        if weights:
            wtsm = fid[0].data[2,:,:]
        else:
            wtsm = []
        fid.close()
        brm[np.isnan(brm)] = 0
    except:
        print('! FAILED TO LOAD MAP FOR CR%4.4i' % rot)
        brm = np.zeros((ns, nph))
        wtsm = []
    fname = 'cr%4.4i.fits' % (rot+1)
    try:
        fid = fits.open(datapath+fname)
        brm_l = fid[0].data[0,:,:]
        fid.close()
        brm_l[np.isnan(brm_l)] = 0
    except:
        print('! FAILED TO LOAD MAP FOR CR%4.4i' % (rot+1))
        brm_l = np.zeros((ns, nph))
    fname = 'cr%4.4i.fits' % (rot-1)
    try:
        fid = fits.open(datapath+fname)
        brm_r = fid[0].data[0,:,:]
        fid.close()
        brm_r[np.isnan(brm_r)] = 0
    except:
        print('! FAILED TO LOAD MAP FOR CR%4.4i' % (rot-1))
        brm_r = np.zeros((ns, nph))

    nsm = np.size(brm, axis=0)
    npm = np.size(brm, axis=1)
    dsm = 2.0/nsm
    dpm = 2*np.pi/npm
    scm = np.linspace(-1 + 0.5*dsm, 1 - 0.5*dsm, nsm)
    pcm = np.linspace(0.5*dpm, 2*np.pi - 0.5*dpm, npm)

    # Stitch together:
    brm3 = np.concatenate((brm_l, brm, brm_r), axis=1)
    del(brm, brm_l, brm_r)

    # Coordinates of combined map (pretend it goes only once around Sun in longitude!):
    nsm = np.size(brm3, axis=0)
    npm = np.size(brm3, axis=1)
    dsm = 2.0/nsm
    dpm = 2*np.pi/npm
    scm = np.linspace(-1 + 0.5*dsm, 1 - 0.5*dsm, nsm)
    pcm = np.linspace(0.5*dpm, 2*np.pi - 0.5*dpm, npm)
    
    # (2) SMOOTH COMBINED MAP WITH SPHERICAL HARMONIC FILTER
    # ------------------------------------------------------
    if (smooth > 0):
       # Azimuthal dependence by FFT:
       brm3 = np.fft.fft(brm3, axis=1)

       # Choose suitable lmax based on smoothing filter coefficient:
       # -- such that exp[-smooth*lmax*(lmax+1)] < 0.05
       # -- purpose of this is to make sure high l's are suppressed, to avoid ringing
       lmax = 0.5*(-1 + np.sqrt(1-4*np.log(0.05)/smooth))
       print('lmax = %i' % lmax)

       # Compute Legendre polynomials on equal (s, ph) grid,
       # with spherical harmonic normalisation:
       lmax = 2*int((nph-1)/2)  # note - already lower resolution
       nm = 2*lmax+1  # only need to compute this many values
       plm = np.zeros((nsm, nm, lmax+1))
       for m in range(lmax+1):
           plm[:,m,:] = plgndr(m, scm, lmax)
       plm[:,nm-1:(nm-lmax-1):-1,:] = plm[:,1:lmax+1,:]
       
       # Compute spherical harmonic coefficients:
       blm = np.zeros((nm,lmax+1), dtype='complex')
       for l in range(lmax+1):
           blm[:lmax+1,l] = np.sum(plm[:,:lmax+1,l]*brm3[:,:lmax+1]*dsm, axis=0)
           blm[lmax+1:,l] = np.sum(plm[:,lmax+1:,l]*brm3[:,-lmax:]*dsm, axis=0)
           # Apply smoothing filter:
           blm[:,l] *= np.exp(-smooth*l*(l+1))

       # Invert transform:
       brm3[:,:] = 0.0
       for j in range(nsm):
           brm3[j,:lmax+1] = np.sum(blm[:lmax+1,:]*plm[j,:lmax+1,:], axis=1)
           brm3[j,-lmax:] = np.sum(blm[lmax+1:,:]*plm[j,lmax+1:,:], axis=1)

       brm3 = np.real(np.fft.ifft(brm3, axis=1))
                              
    # (3) INTERPOLATE CENTRAL MAP TO COMPUTATIONAL GRID
    # -------------------------------------------------
    # Form computational grid arrays:
    ds = 2.0/ns
    dph = 2*np.pi/nph
    sc = np.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns)
    pc1 = np.linspace( 0.5*dph, 2*np.pi - 0.5*dph, nph)
    pc = pc1/3 + 2*np.pi/3  # coordinate on the stitched grid

    # Interpolate to the computational grid:
    bri = interp2d(pcm, scm, brm3, kind='cubic', copy=True, bounds_error=False, fill_value=0)
    br = np.zeros((ns, nph))
    for i in range(ns):
       br[i,:] = bri(pc, sc[i]).flatten()
               
    # (4) INTERPOLATE LEFT AND RIGHT MAPS TO COMPUTATIONAL GRID
    # ---------------------------------------------------------
    brl = np.zeros((ns, nph))
    brr = np.zeros((ns, nph))
    for i in range(ns):
       brl[i,:] = bri(pc - 2*np.pi/3, sc[i]).flatten()
       brr[i,:] = bri(pc + 2*np.pi/3, sc[i]).flatten()

    del(brm3, bri)

    return br, brl, brr, wtsm


#--------------------------------------------------------------------------------------------------------------
def get_plages(readmap, rot, ns, nph, plots=False, outpath='./', datapath='./', cmin=1.094, max_lat=np.deg2rad(50), min_pix=1, dilation_size=0):
    """
        Determine plage regions by thresholding Ca-K synoptic map + neighbours.
        
        Returns arrays on s, phi grid (corrected for flux balance), and longitude centroid.
    """
    
    # Get map and neighbours smoothed and interpolated on dumfric grid.
    # Map itself is corrected for flux balance but neighbours are not.
    mapc, mapl, mapr, _ = readmap(rot, ns, nph, datapath=datapath)
    m3 = np.concatenate((mapl, mapc, mapr), axis=1)
    del(mapl, mapr)

    # Coordinates on triple map:
    np3 = np.size(m3, axis=1)
    dp3 = 6*np.pi/np3
    ds = 2./ns
    pc3 = np.linspace(0.5*dp3, 6*np.pi - 0.5*dp3, np3)
    sc3 = np.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns) 

    # Identify individual plage regions by thresholding:
    m3_d = (m3 > cmin)
    if dilation_size > 0:
        m3_d = grey_dilation(m3_d, size=dilation_size)
    labels, nregs = label(m3_d)

    # Remove regions with longitude centroid (area not flux) outside original map, 
    # too few pixels (< min_pix), or too near the pole (> max_lat).
    sc, pc = np.meshgrid(sc3, pc3, indexing='ij')
    regs = []
    pregs = []
    for i in range(1,nregs+1):
        if (np.sum(labels==i) > 0):
            pcen = np.mean(pc[labels==i])
            scen = np.mean(sc[labels==i])
            if ((pcen < 2*np.pi) | (pcen > 4*np.pi) | (abs(scen) > np.sin(max_lat)) | (np.sum(labels==i) < min_pix)):
                labels[labels==i] = 0
            else:
                # Create list of corrected Br arrays for good regions (ensure to include wrap-arounds).
                m1 = m3.copy()
                m1[labels!=i] = 0
                regs.append(m1)
                pregs.append(pcen)

    nregs = len(regs)
    regs = np.array(regs)

    if plots:
        plt.figure(figsize=(10,6))
        plt.subplot(311)
        plt.pcolormesh(np.rad2deg(pc3), sc3, m3, cmap='viridis', vmin=0.8, vmax=1.6)
        plt.plot([360, 360], [-1, 1], 'w--', linewidth=0.75)
        plt.plot([720, 720], [-1, 1], 'w--', linewidth=0.75)
        plt.title('CR%4.4i' % rot)

        plt.subplot(312)
        plt.pcolormesh(np.rad2deg(pc3), sc3, labels, cmap='nipy_spectral')
        # plt.pcolormesh(np.rad2deg(pc3), sc3, labels, cmap='viridis', vmin=0.8, vmax=1.6)
        plt.plot([360, 360], [-1, 1], 'w--', linewidth=0.75)
        plt.plot([720, 720], [-1, 1], 'w--', linewidth=0.75)

        plt.subplot(313)
        if nregs > 0:
            plt.pcolormesh(np.rad2deg(pc3), sc3, np.sum(regs,axis=0), cmap='viridis', vmin=0.8, vmax=1.6)
            # plt.pcolormesh(np.rad2deg(pc3[nph:2*nph]), sc3, np.sum(regs,axis=0), cmap='viridis', vmin=0.8, vmax=1.6)
        # plt.xlim(0, 1080)
        plt.ylim(-1, 1)

        plt.text(20, 0.6, 'THRESHOLD = %g' % cmin, color='w')
        plt.text(20, 0.3, 'MIN_PIX = %i' % min_pix, color='w')
        plt.text(20, 0, 'N REGIONS = %i' % nregs, color='w')

        plt.savefig('regions_%4.4i.png' % rot)
        plt.close()

    return regs, mapc, pregs