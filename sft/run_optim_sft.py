"""
    Multiple SFT runs using SFT 1D. For full period.

    A Yeates - Aug 2024
"""
import multiprocessing
from itertools import repeat
import numpy as np
from scipy.interpolate import interp1d 
import datetime
from pyDOE import lhs
import pickle
import os
import time
from _sft_ import sftrun, polar_fluxes
import sys
sys.path.append('../_data_/')
from _utils_ import toYearFraction

# List of plage data realizations (directories):
outpaths = [('/Users/bmjg46/Documents/stfc-historical/regions-full/regions-hale%2.2i/' % i) for i in range(10)]

# Path to ground-truth polar field data:
datapath_polar = '/Users/bmjg46/Documents/data/mj-polar-faculae/'

# Plage data parameters:
t_start = datetime.datetime(1923, 10, 31, 12)
t_end = datetime.datetime(1985, 7, 31, 12)
ns = 180
nph = 360

# Number of SFT runs:
nsamp = 10000

# Parameters (set to list of [min, max] in order to vary):
eta0 = [200, 1000]
v0 = [5e-3, 30e-3]
p0 = [1,10]
tau = 0 # [1*365.25*86400, 20*365.25*86400]
bq = 0    
b0 = [-10, 0]

# Set this parameter to True to run the simulation but without regenerating the emerging region files:
reuse_regions = False

# Set this parameter to False to recalculate the ground truth arrays:
reuse_ground_truth = False

# How many timesteps to skip in order to reduce resolution when computing objective function:
obj_cadence = 10

# Whether to do calculation in parallel:
parallel = True

#--------------------------------------------------------------------------------------------------------------
def steady_state(sc2, v0, eta0, p0):
    """
    (Approx) steady-state Br for given SFT parameters.
    """
    
    RSUN = 6.96e5
    rsundu = v0*(1+p0)**(0.5*(1+p0))/p0**(0.5*p0)
    rm = RSUN*rsundu/eta0
    br = np.exp(-rm * (1-sc2**2)**(0.5*(1+p0)) / (1+p0))
    br[sc2 < 0] = -br[sc2 < 0]

    return br

#--------------------------------------------------------------------------------------------------------------
def objective_pf(pf_nor_sim, pf_sou_sim, pf_nor_obs, pf_sou_obs, err_nor_obs, err_sou_obs, skip=1):
    """
    Objective function to minimize in the optimization [polar field].
    Includes both north and south polar fluxes. 
    Computes both weighted [by uncertainties] and unweighted versions.
    """

    pnor = (pf_nor_sim[::skip] - pf_nor_obs[::skip])**2
    psou = (pf_sou_sim[::skip] - pf_sou_obs[::skip])**2
    obj_wt = np.sqrt(np.mean(pnor/err_nor_obs[::skip]**2 + psou/err_sou_obs[::skip]**2))
    obj_un = np.sqrt(np.mean(pnor + psou))

    return obj_wt, obj_un

#--------------------------------------------------------------------------------------------------------------
def groundTruth_pf(tfull, ns, nph, filename='f_PFlux_MWO_WSO_MDI_2.0.dat', datapath_polar='./'):
    """
    Generate ground truth arrays from observed polar faculae data.
    """

    # Read data and extract separate arrays:
    dat = np.genfromtxt(datapath_polar+filename, comments='%', usecols=(0, 1, 2, 9, 10, 11))
    year_n = dat[:,0]
    pf_n = dat[:,1]
    err_n = dat[:,2]
    year_s = dat[:,3]
    pf_s = dat[:,4]
    err_s = dat[:,5]

    # Interpolators for observed times:
    goodn = ~np.isnan(pf_n)
    goods = ~np.isnan(pf_s)
    pf_n_interpolator = interp1d(year_n[goodn], pf_n[goodn], fill_value=0, bounds_error=False, kind='linear')
    pf_s_interpolator = interp1d(year_s[goods], pf_s[goods], fill_value=0, bounds_error=False, kind='linear')
    err_n_interpolator = interp1d(year_n[goodn], err_n[goodn], fill_value=0, bounds_error=False, kind='linear')
    err_s_interpolator = interp1d(year_s[goods], err_s[goods], fill_value=0, bounds_error=False, kind='linear')

    # Interpolate to required times:
    pf_nor_obs = pf_n_interpolator(tfull)
    pf_sou_obs = pf_s_interpolator(tfull)
    err_nor_obs = err_n_interpolator(tfull)
    err_sou_obs = err_s_interpolator(tfull)
    return pf_nor_obs, pf_sou_obs, err_nor_obs, err_sou_obs

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for outpath in outpaths:

        # Create output directory if necessary:
        os.system('mkdir -p '+outpath.replace(' ', '\ '))
        # Start time of simulation:
        starttime = t_start.strftime('%Y%m%d.%H')
        # End time of simulation:
        endtime = t_end.strftime('%Y%m%d.%H')
        hr_end = (t_end - t_start).days * 24 + (t_end - t_start).seconds // 3600

        # Generate grid:
        # --------------
        ds = 2.0/ns
        dph = 2*np.pi/nph
        sc = np.linspace(-1 + 0.5*ds, 1 - 0.5*ds, ns)
        pc = np.linspace(0.5*dph, 2*np.pi - 0.5*dph, nph)
        sc2, pc2 = np.meshgrid(sc, pc, indexing='ij')

        # Generate region files:
        # ----------------------
        if not reuse_regions:
            os.system('rm -f '+outpath+'timefile.txt')
            # - Create timefile with list of the region files and their hour of emergence:
            rfiles = []
            for file in os.listdir(outpath+'/regions'):
                if file.startswith('r') and file.endswith('.unf'):
                    rfiles.append(file)
            rfiles.sort()
            timefile = open(outpath+'timefile.txt', 'w')
            if rfiles != []:
                for rfile in rfiles:
                    t_em = datetime.datetime.strptime(rfile[1:12], '%Y%m%d.%H')
                    ihour = (t_em - t_start).days * 24 + (t_em - t_start).seconds//3600
                    timefile.write(('%i\t' % ihour)+rfile+'\n')
            timefile.close()      

        # Compile the fortran code:
        # [if you update your Fortran installation you may need to `make clean` first]
        os.system('make -C fortran')

        # Generate random parameters:
        # ---------------------------
        # - Determine number of variable parameters:
        parnames = ['eta0', 'v0', 'p0', 'tau', 'bq', 'b0']
        npar = 0
        for param in parnames:
            if type(locals()[param])==list:  # is this parameter being varied?
                npar += 1
        # - Generate random parameters by Latin hypercube sampling:
        np.random.seed(1) 
        params = lhs(npar, samples=nsamp, criterion='center')  # on range [0,1]
        kc = 0
        for k, param in enumerate(parnames):
            if type(locals()[param])==list:
                pmin = locals()[param][0]
                pmax = locals()[param][1]
                locals()[param+'s'] = params[:,kc]*(pmax - pmin) + pmin
                kc += 1
            else:
                locals()[param+'s'] = np.zeros(nsamp) + locals()[param]

        # Do multiple SFT runs:
        # ---------------------
        obj_pf_wt = np.zeros(nsamp)
        obj_pf_un = np.zeros(nsamp)


        # - do first run (establishes number of timesteps):
        br0 = steady_state(sc2, v0s[0], eta0s[0], p0s[0])
    
        print('Starting run 0 -- ',b0s[0], eta0s[0], v0s[0], p0s[0], taus[0], bqs[0])
        dates, uflux, dipole, bfly, ubfly, br  = sftrun(0, ns, nph, outpath, t_start, hr_end, br0, eta0s[0], v0s[0], p0s[0], taus[0], bqs[0], b0=b0s[0], dims=1)
        tfull = [toYearFraction(d) for d in dates]
        t = np.array(tfull)
        t -= t[0]
        nt = len(dates)

        # - generate ground truth arrays of observables:
        if not reuse_ground_truth:
            os.system('rm -f '+outpath+'ground_truth.pkl')

        try:
            picklefile = open(outpath+'ground_truth.pkl','rb')
            pf_nor_obs = pickle.load(picklefile)
            pf_sou_obs = pickle.load(picklefile)
            err_nor_obs = pickle.load(picklefile)
            err_sou_obs = pickle.load(picklefile)
            picklefile.close()
        except:
            pf_nor_obs, pf_sou_obs, err_nor_obs, err_sou_obs = groundTruth_pf(tfull, ns, nph, datapath_polar=datapath_polar)
            picklefile = open(outpath+'ground_truth.pkl','wb')
            pickle.dump(pf_nor_obs, picklefile)
            pickle.dump(pf_sou_obs, picklefile)
            pickle.dump(err_nor_obs, picklefile)
            pickle.dump(err_sou_obs, picklefile)
            picklefile.close()

        # - compute objective function for first run:
        pf_nor_sim, pf_sou_sim = [], []
        pf_nor_sim1, pf_sou_sim1 = polar_fluxes(t, sc, bfly)

        pf_nor_sim.append(pf_nor_sim1[::obj_cadence])
        pf_sou_sim.append(pf_sou_sim1[::obj_cadence])

        obj_pf_wt[0], obj_pf_un[0] = objective_pf(pf_nor_sim1, pf_sou_sim1, pf_nor_obs, pf_sou_obs, err_nor_obs, err_sou_obs, skip=obj_cadence)
        # - do remaining runs:
        tic = time.time()
        if parallel:
            ncpu = os.cpu_count()-1
            br0s = np.zeros((ncpu, ns, nph))
            print('Starting %i remaining runs with %i cpus' % (nsamp-1, ncpu))
            tic = time.time()
            pool = multiprocessing.Pool(ncpu)
            for k in range(1, nsamp, ncpu):
                print('Starting wave from %i to %i' % (k, k+ncpu))
                # - initial conditions:
                for i in range(ncpu):
                    try:
                        br0s[i,:,:] = steady_state(sc2, v0s[k+i], eta0s[k+i], p0s[k+i])
                    except:
                        pass
                _, uflux, dipole, bfly, ubfly, _  = zip(*pool.starmap(sftrun, zip(range(k,k+ncpu), repeat(ns), repeat(nph), repeat(outpath), repeat(t_start), repeat(hr_end), br0s, eta0s[k:k+ncpu], v0s[k:k+ncpu], p0s[k:k+ncpu], taus[k:k+ncpu], bqs[k:k+ncpu], b0s[k:k+ncpu], repeat(1))))
                for j in range(ncpu):
                    try:
                        pf_nor_sim1, pf_sou_sim1 = polar_fluxes(t, sc, bfly[j])
                        pf_nor_sim.append(pf_nor_sim1[::obj_cadence])
                        pf_sou_sim.append(pf_sou_sim1[::obj_cadence])
                        obj_pf_wt[k+j], obj_pf_un[k+j] = objective_pf(pf_nor_sim1, pf_sou_sim1, pf_nor_obs, pf_sou_obs, err_nor_obs, err_sou_obs, skip=obj_cadence)
                    except:
                        break
        else:
            for k in range(1, nsamp):
                br0 = steady_state(sc2, v0s[k], eta0s[k], p0s[k])
                print('Starting run %i --' % k, b0s[k], eta0s[k], v0s[k], p0s[k], taus[k], bqs[k])
                _, uflux, dipole, bfly, ubfly, _  = sftrun(k, ns, nph, outpath, t_start, hr_end, br0, eta0s[k], v0s[k], p0s[k], taus[k], bqs[k], b0=b0s[k], dims=1)
                # - compute objective function:
                pf_nor_sim1, pf_sou_sim1 = polar_fluxes(t, sc, bfly)
                pf_nor_sim.append(pf_nor_sim1[::obj_cadence])
                pf_sou_sim.append(pf_sou_sim1[::obj_cadence])
                obj_pf_wt[k], obj_pf_un[k] = objective_pf(pf_nor_sim1, pf_sou_sim1, pf_nor_obs, pf_sou_obs, err_nor_obs, err_sou_obs, skip=obj_cadence)

        toc = time.time()
        print('Elapsed time = %g' % (toc-tic))

        # Save the outputs to pickle file:
        # --------------------------------
        picklefile = open(outpath+('optim%i_%i.pkl' % (npar, nsamp)),'wb')
        pickle.dump(eta0s, picklefile)
        pickle.dump(v0s, picklefile)
        pickle.dump(p0s, picklefile)
        pickle.dump(taus, picklefile)
        pickle.dump(bqs, picklefile)
        pickle.dump(b0s, picklefile)
        pickle.dump(obj_pf_wt, picklefile)
        pickle.dump(obj_pf_un, picklefile)
        pickle.dump(ns, picklefile)
        pickle.dump(nph, picklefile)
        pickle.dump(t_start, picklefile)
        pickle.dump(t_end, picklefile)
        pickle.dump(eta0, picklefile)
        pickle.dump(v0, picklefile)
        pickle.dump(p0, picklefile)
        pickle.dump(tau, picklefile)
        pickle.dump(bq, picklefile)
        pickle.dump(b0, picklefile)
        pickle.dump(tfull, picklefile)
        picklefile.close()

        pf_nor_sim = np.array(pf_nor_sim)
        pf_sou_sim = np.array(pf_sou_sim)
        picklefile = open(outpath+('optim%i_%i_pf_sim.pkl' % (npar, nsamp)),'wb')
        pickle.dump(pf_nor_sim, picklefile)
        pickle.dump(pf_sou_sim, picklefile)
        pickle.dump(tfull, picklefile)
        picklefile.close()       
