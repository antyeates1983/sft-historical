"""
    Python tools for reading NSO synoptic maps and mapping to (s,phi) grid.
    
    A Yeates 2024-Nov
"""
import numpy as np
import ftplib
from astropy.io import fits
from scipy.io import netcdf
from scipy.interpolate import RectBivariateSpline
from _utils_ import plgndr
from sunpy.coordinates.sun import carrington_rotation_number
import os
from paths import datapath_cak
from _data_cak_mwo_ import readmap as readmap_cak

#--------------------------------------------------------------------------------------------------------------
def prep_cak_br_maps(ns, nph, t_start, t_end, bad_br, bad_ca, max_lat, outpath='./nso/'):
    """
    Prepare Ca K and NSO maps.
    """
    
    # Create directory for plots etc:
    os.system('mkdir '+outpath)
    os.system('mkdir '+outpath+'plages')
    os.system('mkdir '+outpath+'polarities')

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

    # Read Ca K and NSO maps and save to pickle file for more rapid reloading:
    try:
        fid = netcdf.netcdf_file(outpath+'nso-v-cak.nc', 'r', mmap=False)
        crs = fid.variables['cr'][:]
        map_br = fid.variables['map_br'][:]
        map_cak = fid.variables['map_cak'][:]
        fid.close()
    except:
        fid = netcdf.netcdf_file(outpath+'nso-v-cak.nc', 'w')
        fid.createDimension('tdim', ncar)
        fid.createDimension('sdim', ns)
        fid.createDimension('pdim', nph)
        tid = fid.createVariable('cr', 'd', ('tdim',))
        brid = fid.createVariable('map_br', 'd', ('tdim','sdim','pdim'))
        cakid = fid.createVariable('map_cak', 'd', ('tdim','sdim','pdim'))
        for cr in range(cr_start, cr_end+1):
            print(cr)
            k = cr-cr_start
            map_br0, _, _ = readmap(cr, ns, nph, smooth=0)
            map_cak0, _, _, wtsm = readmap_cak(cr, ns, nph, smooth=0, datapath=datapath_cak, weights=False)
            tid[k] = cr
            brid[k,:,:] = map_br0
            cakid[k,:,:] = map_cak0
        fid.close()
        crs = tid[:]
        map_br = brid[:,:,:]
        map_cak = cakid[:,:,:]

    # Set maps to zero at high latitudes:
    for k in range(0, cr_end-cr_start+1):
        map_cak[k][np.abs(sc2) > np.sin(max_lat)] = 0
        map_br[k][np.abs(sc2) > np.sin(max_lat)] = 0
    nlow = np.sum(np.abs(sc2) <= np.sin(max_lat))

    # Get indices to "good" maps:
    ibr = np.ones(ncar, dtype=np.int16)
    ica = np.ones(ncar, dtype=np.int16)
    iboth = np.ones(ncar, dtype=np.int16)
    for cr in range(cr_start, cr_end+1):
        k = cr - cr_start
        if np.isin(cr, bad_br):
            ibr[k] = 0
            iboth[k] = 0
        if np.isin(cr, bad_ca):
            ica[k] = 0
            iboth[k] = 0
    ibr = np.where(ibr > 0)
    ica = np.where(ica > 0)
    iboth = np.where(iboth > 0)

    return ncar, cr_start, cr_end, nlow, iboth, crs, map_br, map_cak

#--------------------------------------------------------------------------------------------------------------
def get_solis_filename(rot, solisfiles):
    """
    Download a SOLIS map.
    """
    for f in solisfiles:
        if (('c%4.4i' % rot in f) & ('fits.gz' in f)):
            return 'ftp://solis.nso.edu/integral/kbv7g/'+f
    return ''

#--------------------------------------------------------------------------------------------------------------
def readmap(rot, ns, nph, smooth=0, datapath='./', smoothtype='new'):
    """
        Read in the synoptic map for Carrington rotation rot, corrects the flux and maps to the DuMFric grid.
        Also reads in the neighbouring maps, and puts them together for smoothing.
        
        ARGUMENTS:
            rot is the number of the required Carrington rotation (e.g. 2190)
            ns and nph define the required grid (e.g. 180 and 360)
            smooth [optional] controls the strength of smoothing (default 0 is no smoothing)
        
        [Important: the output maps are not corrected for flux balance - do this later if required.]       
        
        Searches NSO ftp site for KPVT or SOLIS synoptic maps.
    """
    ftp = ftplib.FTP('solis.nso.edu')
    ftp.login()
    ftp.cwd('integral/kbv7g')    
    solisfiles = ftp.nlst()

    # (1) READ IN DATA AND STITCH TOGETHER 3 ROTATIONS
    # ------------------------------------------------
    # Read in map and neighbours:
    try:
        try:
            brm = (fits.open('ftp://nispdata.nso.edu/kpvt/synoptic/mag/m%4.4if.fits' % rot))[0].data
            print('FOUND KPVT MAP FOR CR%4.4i' % rot)
        except:
            brm = (fits.open(get_solis_filename(rot, solisfiles)))[0].data
            print('FOUND SOLIS MAP FOR CR%4.4i' % rot)
    except:
        print('! FAILED TO LOAD MAP FOR CR%4.4i' % rot)
        brm = np.zeros((ns, nph))
    try:
        try:
            brm_l = (fits.open('ftp://nispdata.nso.edu/kpvt/synoptic/mag/m%4.4if.fits' % (rot+1)))[0].data
            print('FOUND KPVT MAP FOR CR%4.4i' % (rot+1))
        except:
            brm_l = (fits.open(get_solis_filename(rot+1, solisfiles)))[0].data
            print('FOUND SOLIS MAP FOR CR%4.4i' % (rot+1))
    except:
        print('! FAILED TO LOAD MAP FOR CR%4.4i' % (rot+1))
        brm_l = np.zeros((ns, nph))
    try:
        try:
            brm_r = (fits.open('ftp://nispdata.nso.edu/kpvt/synoptic/mag/m%4.4if.fits' % (rot-1)))[0].data
            print('FOUND KPVT MAP FOR CR%4.4i' % (rot-1))
        except:
            brm_r = (fits.open(get_solis_filename(rot-1, solisfiles)))[0].data
            print('FOUND SOLIS MAP FOR CR%4.4i' % (rot-1))
    except:
        print('! FAILED TO LOAD MAP FOR CR%4.4i' % (rot-1))
        brm_r = np.zeros((ns, nph))
    
    ftp.quit()

    nsm = np.size(brm, axis=0)
    npm = np.size(brm, axis=1)
    dsm = 2.0/nsm
    dpm = 2*np.pi/npm
    scm = np.linspace(-1 + 0.5*dsm, 1 - 0.5*dsm, nsm)  
    pcm = np.linspace(0.5*dpm, 2*np.pi - 0.5*dpm, npm)  
    
    # If necessary, rebin neighbouring map(s) to same resolution as central map:
    if (np.shape(brm_l)!=np.shape(brm)):
        nsl = np.size(brm_l, axis=0)
        npl = np.size(brm_l, axis=1)
        dsl = 2.0/nsl
        dpl = 2*np.pi/npl
        scl = np.linspace(-1 + 0.5*dsl, 1 - 0.5*dsl, nsl)  
        pcl = np.linspace(0.5*dpl, 2*np.pi - 0.5*dpl, npl)
        bri = RectBivariateSpline(pcl, scl, brm_l.T)
        brm_l = np.zeros((nsm, npm))
        for i in range(nsm):
            brm_l[i,:] = bri(pcm, scm[i]).T.flatten()
        del(bri)
    if (np.shape(brm_r)!=np.shape(brm)):
        nsr = np.size(brm_r, axis=0)
        npr = np.size(brm_r, axis=1)
        dsr = 2.0/nsr
        dpr = 2*np.pi/npr
        scr = np.linspace(-1 + 0.5*dsr, 1 - 0.5*dsr, nsr)  
        pcr = np.linspace(0.5*dpr, 2*np.pi - 0.5*dpr, npr)
        bri = RectBivariateSpline(pcr, scr, brm_r.T)
        brm_r = np.zeros((nsm, npm))
        for i in range(nsm):
            brm_r[i,:] = bri(pcm, scm[i]).T.flatten()
        del(bri)
        
    # Stitch together:
    brm3 = np.concatenate((brm_l, brm, brm_r), axis=1)
    del(brm, brm_l, brm_r)
    
    # Remove NaNs:
    brm3 = np.nan_to_num(brm3)
    
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
        if smoothtype == 'new':
            # NEW IMPLEMENTATION: computes discrete eigenfunctions instead of Plm.
            # Find eigenfunctions of discrete Laplacian:

            # Prepare tridiagonal matrix:
            Fp = sgm * 0  # Lp/Ls on p-ribs
            Fp[1:-1] = np.sqrt(1 - sgm[1:-1] ** 2) / (np.arcsin(scm[1:]) - np.arcsin(scm[:-1])) * dpm
            Vg = Fp / dsm / dpm
            Fs = ((np.arcsin(sgm[1:]) - np.arcsin(sgm[:-1])) / np.sqrt(1 - scm ** 2) / dpm)  # Ls/Lp on s-ribs
            Uc = Fs / dsm / dpm
            # - create off-diagonal part of the matrix:
            A = np.zeros((nsm, nsm))
            for j in range(nsm- 1):
                A[j, j + 1] = -Vg[j + 1]
                A[j + 1, j] = A[j, j + 1]
            # - term required for m-dependent part of matrix:
            mu = np.fft.fftfreq(npm)
            mu = 4 * np.sin(np.pi * mu) ** 2

            # FFT in phi of photospheric distribution at each latitude:
            brhat = np.fft.rfft(brm3, axis=1)

            # Loop over azimuthal modes (positive m):
            nm = npm//2 + 1
            blm = np.zeros((nsm, nm), dtype="complex")
            brhat1 = np.zeros((nsm, nm), dtype="complex")
            for m in range(nm):
                # - set diagonal terms of matrix:
                for j in range(nsm):
                    A[j, j] = Vg[j] + Vg[j + 1] + Uc[j] * mu[m]
                # - compute eigenvectors Q_{lm} and eigenvalues lam_{lm}:
                #   (note that A is symmetric so use special solver)
                lam, Q = la.eigh(A)
                # - find coefficients of eigenfunction expansion:
                for l in range(nsm):
                    blm[l,m] = np.dot(Q[:,l], brhat[:,m])
                    # - apply filter [the eigenvalues should be a numerical approx of lam = l*(l+1)]:
                    blm[l,m] *= np.exp(-smooth*lam[l])
                # - invert the latitudinal transform:
                brhat1[:,m] = np.dot(blm[:,m], Q.T)

            # Invert the FFT in longitude:
            brm3 = np.real(np.fft.irfft(brhat1, axis=1))

        else:
            # ORIGINAL IMPLEMENTATION: computing Plm using recurrence - suggest not to use since it is inaccurate for (moderately) large l.

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
                plm[:,m,:] = preptools.plgndr(m, scm, lmax)        
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
    bri = RectBivariateSpline(pcm, scm, brm3.T)
    br = np.zeros((ns, nph))
    for i in range(ns):
        br[i,:] = bri(pc, sc[i]).T.flatten()
                
    # (4) INTERPOLATE LEFT AND RIGHT MAPS TO COMPUTATIONAL GRID
    # ---------------------------------------------------------
    brl = np.zeros((ns, nph))
    brr = np.zeros((ns, nph))
    for i in range(ns):
        brl[i,:] = bri(pc - 2*np.pi/3, sc[i]).flatten()
        brr[i,:] = bri(pc + 2*np.pi/3, sc[i]).flatten()
 
    del(brm3, bri)
                
    return br, brl, brr

