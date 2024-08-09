"""
    Python helper routines.

    A Yeates - Oct 2022
"""
import numpy as np
import time
import datetime

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
def plgndr(m,x,lmax):
    """
        Evaluate associated Legendre polynomials P_lm(x) for given (positive)
        m, from l=0,lmax, with spherical harmonic normalization included.
        Only elements l=m:lmax are non-zero.
        
        Similar to scipy.special.lpmv except that function only works for 
        small l due to overflow, because it doesn't include the normalization.
    """
    
    nx = np.size(x)
    plm = np.zeros((nx, lmax+1))
    pmm = 1
    if (m > 0):
        somx2 = (1-x)*(1+x)
        fact = 1.0
        for i in range(1,m+1):
            pmm *= somx2*fact/(fact+1)
            fact += 2
    
    pmm = np.sqrt((m + 0.5)*pmm)
    pmm *= (-1)**m
    plm[:,m] = pmm
    if (m < lmax):
        pmmp1 = x*np.sqrt(2*m + 3)*pmm
        plm[:,m+1] = pmmp1
        if (m < lmax-1):
            for l in range(m+2,lmax+1):
                fact1 = np.sqrt(((l-1.0)**2 - m**2)/(4.0*(l-1.0)**2-1.0))
                fact = np.sqrt((4.0*l**2-1.0)/(l**2-m**2))
                pll = (x*pmmp1 - pmm*fact1)*fact
                pmm = pmmp1
                pmmp1 = pll
                plm[:,l] = pll
    return plm
