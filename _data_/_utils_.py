"""
    Python helper routines.

    A Yeates 2024-Nov
"""
import numpy as np
import time
import datetime
import scipy.linalg as la

#--------------------------------------------------------------------------------------------------------------
def toYearFraction(date):
    """
    Convert datetime object to fractional year.
    """
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime.datetime(year=year, month=1, day=1)
    startOfNextYear = datetime.datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

#--------------------------------------------------------------------------------------------------------------
def correct_flux_multiplicative(f):
    """
        Correct the flux balance in the map f (assumes that cells have equal area).
    """
    
    # Compute positive and negative fluxes:
    ipos = f > 0
    ineg = f < 0
    fluxp = np.abs(np.sum(f[ipos]))
    fluxn = np.abs(np.sum(f[ineg]))
    
    # Rescale both polarities to mean:
    fluxmn = 0.5*(fluxn + fluxp)
    f1 = f.copy()
    f1[ineg] *= fluxmn/fluxn
    f1[ipos] *= fluxmn/fluxp
    
    return f1

#--------------------------------------------------------------------------------------------------------------
def sh_smooth(f, smooth):
    """
    Smooth array f [on dumfric grid, cell centres, no ghost cells] with spherical harmonic filter exp( -smooth*l*(l+1) ) in spectral space.

    This implementation uses discrete eigenfunctions (in latitude) instead of Plm.
    """

    nsm = np.size(f, axis=0)
    npm = np.size(f, axis=1)
    dsm = 2.0/nsm
    dpm = 2*np.pi/npm
    scm = np.linspace(-1 + 0.5*dsm, 1 - 0.5*dsm, nsm)
    sgm = np.linspace(-1, 1, nsm+1) 

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
    fhat = np.fft.rfft(f, axis=1)

    # Loop over azimuthal modes (positive m):
    nm = npm//2 + 1
    blm = np.zeros((nsm, nm), dtype="complex")
    fhat1 = np.zeros((nsm, nm), dtype="complex")
    for m in range(nm):
        # - set diagonal terms of matrix:
        for j in range(nsm):
            A[j, j] = Vg[j] + Vg[j + 1] + Uc[j] * mu[m]
        # - compute eigenvectors Q_{lm} and eigenvalues lam_{lm}:
        #   (note that A is symmetric so use special solver)
        lam, Q = la.eigh(A)
        # - find coefficients of eigenfunction expansion:
        for l in range(nsm):
            blm[l,m] = np.dot(Q[:,l], fhat[:,m])
            # - apply filter [the eigenvalues should be a numerical approx of lam = l*(l+1)]:
            blm[l,m] *= np.exp(-smooth*lam[l])
        # - invert the latitudinal transform:
        fhat1[:,m] = np.dot(blm[:,m], Q.T)

    # Invert the FFT in longitude:
    f_out = np.real(np.fft.irfft(fhat1, axis=1))
    
    return f_out

# #--------------------------------------------------------------------------------------------------------------
# def plgndr(m,x,lmax):
#     """
#         Evaluate associated Legendre polynomials P_lm(x) for given (positive)
#         m, from l=0,lmax, with spherical harmonic normalization included.
#         Only elements l=m:lmax are non-zero.
        
#         Similar to scipy.special.lpmv except that function only works for 
#         small l due to overflow, because it doesn't include the normalization.
#     """
    
#     nx = np.size(x)
#     plm = np.zeros((nx, lmax+1))
#     pmm = 1
#     if (m > 0):
#         somx2 = (1-x)*(1+x)
#         fact = 1.0
#         for i in range(1,m+1):
#             pmm *= somx2*fact/(fact+1)
#             fact += 2
    
#     pmm = np.sqrt((m + 0.5)*pmm)
#     pmm *= (-1)**m
#     plm[:,m] = pmm
#     if (m < lmax):
#         pmmp1 = x*np.sqrt(2*m + 3)*pmm
#         plm[:,m+1] = pmmp1
#         if (m < lmax-1):
#             for l in range(m+2,lmax+1):
#                 fact1 = np.sqrt(((l-1.0)**2 - m**2)/(4.0*(l-1.0)**2-1.0))
#                 fact = np.sqrt((4.0*l**2-1.0)/(l**2-m**2))
#                 pll = (x*pmmp1 - pmm*fact1)*fact
#                 pmm = pmmp1
#                 pmmp1 = pll
#                 plm[:,l] = pll
#     return plm
