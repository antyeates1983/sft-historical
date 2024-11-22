"""
    Routine to run single simulation and return output to python.

    A Yeates 2024-Nov
"""
import numpy as np
import datetime
from scipy.io import FortranFile
import os

def sftrun(id, ns, nph, outpath, t_start, hr_end, br0, eta0, v0, p0, tau, bq, b0=1, dims=1, codepath='./', hale_factor=[]):
    """
    Run a single simulation with given parameters.
    Set dims=1 for 1D code.
    """
    RSUN = 6.96e10
    SDAY = 86400.0
    omA = np.deg2rad(0.18)/SDAY
    omB = -np.deg2rad(2.396)/SDAY
    omC = -np.deg2rad(1.787)/SDAY

    # Write initial data and parameters to fortran binary file:
    os.system('rm -f '+outpath+'br0%3.3i.unf' % id)
    os.system('rm -f '+outpath+'outs%3.3i.unf' % id)
    fid = FortranFile(outpath+'br0%3.3i.unf' % id, 'w')
    fid.write_record((b0*br0).T.astype(np.float64))
    fid.write_record(omA*1.0)
    fid.write_record(omB*1.0)
    fid.write_record(omC*1.0)
    fid.write_record(v0*1e5/RSUN)
    fid.write_record(p0*1.0)
    fid.write_record(eta0*1e10/RSUN**2)
    fid.write_record(bq*1.0)
    fid.write_record(tau*1.0)
    fid.close()
    # Run simulation:
    if (dims == 1):
        if (hale_factor == []):
            os.system(codepath+'fortran/sft1d '+outpath+(' %3.3i %i %i %i' % (id, ns, nph, hr_end)))
        else:
            os.system(codepath+'fortran/sft1d '+outpath+(' %3.3i %i %i %i %4.2f' % (id, ns, nph, hr_end, hale_factor)))
    else:
        os.system(codepath+'fortran/sft '+outpath+(' %3.3i %i %i %i' % (id, ns, nph, hr_end)))
    # Read data:
    fid = FortranFile(outpath+'outs%3.3i.unf' % id, 'r')
    nouts = fid.read_ints(dtype=np.int32)[0]
    hr_out = fid.read_ints(dtype=np.int32)
    uflux = fid.read_ints(dtype=np.float64)
    dipole = fid.read_reals(dtype=np.float64)
    bfly = fid.read_reals(dtype=np.float64).reshape((ns, nouts))
    ubfly = fid.read_reals(dtype=np.float64).reshape((ns, nouts))
    fid.close()
    dates = [t_start + datetime.timedelta(hours=int(hr)) for hr in hr_out]
    
    if (dims == 2):
        # Read final br snapshot:
        fid = FortranFile(outpath+'final_br%3.3i.unf' % id, 'r')
        br = fid.read_reals(dtype=np.float64).reshape((nph, ns)).T
        fid.close()
    else:
        br = np.zeros((ns, nph))
    
    # Remove temporary files:
    os.system('rm -f '+outpath+'br0%3.3i.unf' % id)
    os.system('rm -f '+outpath+'outs%3.3i.unf' % id)
    os.system('rm -f '+outpath+'final_br%3.3i.unf' % id)

    return dates, uflux, dipole, bfly, ubfly, br

def polar_fluxes(t, sc, bfly_sim, pcap=np.deg2rad(20)):
    """
    Compute polar fluxes from simulated butterfly diagram.
    pcap specifies polar cap boundary in radians.
    """

    RSUN = 6.96e10
    ds = sc[1]-sc[0]

    # Compute polar cap fluxes from simulation:
    sc2, t2 = np.meshgrid(sc, t, indexing='ij')

    f = bfly_sim.copy()
    f[sc2 < np.cos(pcap)] = 0
    pf_nor_sim = np.sum(f, axis=0)
    pf_nor_sim *= 2*np.pi * RSUN**2 * ds

    f = bfly_sim.copy()
    f[sc2 > -np.cos(pcap)] = 0
    pf_sou_sim = np.sum(f, axis=0)
    pf_sou_sim *= 2*np.pi * RSUN**2 * ds

    return pf_nor_sim, pf_sou_sim