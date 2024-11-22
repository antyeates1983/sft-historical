"""
    Python tools for reading sunspot area data from Mandal+2020.
    https://ui.adsabs.harvard.edu/abs/2020A%26A...640A..78M/abstract
    
    https://cdsarc.u-strasbg.fr/viz-bin/cat/J/A+A/640/A78#/browse
    
    A Yeates 2024-Nov
"""
import numpy as np
import datetime
from paths import datapath_mandal_ssa
from _utils_ import toYearFraction

#--------------------------------------------------------------------------------------------------------------
def prep_mandal_ssa():
    """
    Prepare sunspot area data from Mandal+2020 https://cdsarc.u-strasbg.fr/viz-bin/cat/J/A+A/640/A78#/browse
    """
    ssadat = np.loadtxt(datapath_mandal_ssa+'J_A+A_640_A78_catalog1.dat.gz.txt', delimiter='|', dtype=str, skiprows=9371) # need to start on 1 Jan in a year

    datestr_ssa = ssadat[:,0]
    areac_ssa = ssadat[:,3]
    # - convert to dates:
    dates = []
    for d in datestr_ssa:
        dates.append( datetime.datetime.strptime(d, '%Y/%m/%d') )
    # - monthly average:
    t_ssam = []
    areac_ssam = []
    dnxt = 0
    for yr in range(dates[0].year, dates[-1].year):
        for mon in range(1,13):
            tot = 0
            nn = 0.0
            while (dates[dnxt].year==yr) & (dates[dnxt].month == mon):
                tot += float(areac_ssa[dnxt])
                nn += 1
                dnxt += 1
            areac_ssam.append(tot/nn)
            t_ssam.append(toYearFraction(dates[dnxt-15]))
    areac_ssam = np.array(areac_ssam)
    t_ssam = np.array(t_ssam)

    return t_ssam, areac_ssam