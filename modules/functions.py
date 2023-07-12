# license etc


# IMPORT PACKAGES

import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.integrate import simpson
import random
from joblib import Parallel, delayed
import multiprocessing



# FIXED PARAMETERS
pi = np.pi
mp = 1.6726e-24 # grams
mu = 1.22 # mean molecular weight
pc = 3.086e18 # cm
msun = 1.99e33 # grams


# RADIATIVE TRANSFER PARAMETERS
hden = np.arange(0, 6.75, 0.25)        # cm^-3
logG0 = np.arange(0, 6.2, 0.25)[::-1]  # 6-13.6 eV
logFX = np.arange(-1, 4.2, 0.25)[::-1] # 1-100 keVA
hnames = ['h' + ('%03d' % (h*1e2)).replace('.', '') for h in hden]
gnames = ['g' + ('%.2f' % g).replace('.', '') for g in logG0]
xnames = ['x' + ('%.2f' % x).replace('.', '') for x in logFX]





def Mmol_r(r, Mmol_tot, rCO):
    '''
    Molecular mass at each radius from the exponential profile
    (Equation 5 from Esposito+23)
    '''
    Mmol = Mmol_tot * (1 - (np.exp(-r/rCO) * (r/rCO + 1)))
    return Mmol


def Vmol_r(r, rCO):
    '''
    Molecular volume at each radius in cm^-3
    (Equation 6 from Esposito+23)
    '''
    if r <= 1.5*rCO/17:
        Vmol = (4/3) * pi * r**3 * pc**3         # cm^-3
    else:
        Vmol = 2 * pi * (rCO/17) * r**2 * pc**3  # cm^-3
    return Vmol


def sersic(r, I_e, r_eff, n):
    '''
    Sersic profile value for a given radius r
    approximation for n>0.36 (Ciotti & Bertin 1999)
    '''
    bn = 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2)
    bn += 131/(1148175*n**3) - 2194697/(30690717750*n**4)
    I_r = I_e * np.exp(-bn*((r/r_eff)**(1/n) - 1))
    return I_r


def gmc_distribution(a, Mmin=1e3, Mmax=1e6):
    '''
    Given an index a>1, returns an array with the PMF
    '''
    k = 1 / (Mmax**(1-a) - Mmin**(1-a))
    mass_bins = np.arange(np.log10(Mmin), np.log10(Mmax), 0.2)
    masses = np.arange(np.log10(Mmin) + 0.1, np.log10(Mmax), 0.2)
    mass_distr = []
    for logx in mass_bins:
        N = k * ((10**(logx + 0.2))**(1-a) - (10**logx)**(1-a))
        mass_distr.append(N)
    return np.array(mass_distr)


def cluster_factor(a, gmcs, Mmin=1e3, Mmax=1e6):
    '''
    This functions clusters GMCs to speed-up the GMC filling process
    It has a precision of two decimal points
    e.g. a PL index of 1.659 will be rounded to 1.66
    '''
    a = np.round(a, 2)
    gmcMasses = np.array([gmc.M for gmc in gmcs])
    mass_distr = gmc_distribution(a, Mmin, Mmax) / np.array([gmc.M for gmc in gmcs])
    slope, CF = 0., 0
    while np.round(slope, 2) != -a:
        CF += 1
        gmcNums = (CF * mass_distr / mass_distr[-1]).astype('int')
        slope = linregress(np.log10(gmcMasses), np.log10(gmcNums))[0]
    return CF


def gmc_fill_single_ring(i, ring, gmcs, mass_distr, cluster_params, verbose=False):
    gmcCounts = ['N_%s' % gmc.name for gmc in gmcs]
    # add a cluster of GMCs alltogether to speed-up
    clusterNums, clusterMass, clusterVolume = cluster_params
    if (ring['M_r'] > clusterMass) and (ring['V_r'] > clusterVolume):
        Ncluster = int(min([ring['M_r'] / clusterMass, ring['V_r'] / clusterVolume]))
        ring['M_plugged'] += Ncluster * clusterMass
        ring['V_plugged'] += Ncluster * clusterVolume
        ring[gmcCounts] += Ncluster * clusterNums
    else:
        Ncluster = 0
    # add single GMCs extracted one by one
    if (ring['M_r'] > gmcs[0].M) and (ring['V_r'] > gmcs[0].V):
        while all([ring['M_plugged'] < ring['M_r'], ring['V_plugged'] < ring['V_r']]):
            GMCrandom = random.choices(population=gmcs, weights=mass_distr, k=1)
            Mrandom = GMCrandom[0].M  # Msun
            Vrandom = GMCrandom[0].V  # cm^3
            ring['M_plugged'] += Mrandom
            ring['V_plugged'] += Vrandom
            ring['N_%s' % GMCrandom[0].name] += 1
        ring['M_plugged'] -= Mrandom
        ring['V_plugged'] -= Vrandom
        ring['N_%s' % GMCrandom[0].name] -= 1
    else:
        try:
            Ncluster = Ncluster
        except NameError:
            Ncluster = np.nan
    if verbose:
        print('===',  i, '='*10)
        print('Plugged GMCs = %1d' % sum(ring[gmcCounts]))
        print('Number of GMC-clusters = %1d' % Ncluster)
    return ring


def gmc_fill(outfolder, Mmol_tot, r25, logR_step, gmcs, a, n_cores=None, verbose=False):
    '''
    Calculates the galaxy volume, then it fills volume and mass with GMCs
    '''
    rCO = 0.17e3 * r25 # pc
    radii = 10**np.arange(0, np.log10(2*rCO), logR_step) # pc
    mass = [Mmol_r(r, Mmol_tot, rCO) for r in radii]     # Msun
    volume = [Vmol_r(r, rCO) for r in radii]             # cm^-3
    # setting a radially-quantized DataFrame for the galaxy
    quanta = pd.DataFrame({
        'r': radii,                              # radius in pc = external radius of ring
        'V_r': np.diff(volume, prepend=0),       # ring volume in cm^3
        'M_r': np.diff(mass, prepend=0)})        # ring molecular mass in Msun
    # add GMCs to quanta DataFrame
    gmcCounts = ['N_%s' % gmc.name for gmc in gmcs]
    quanta[gmcCounts] = 0
    quanta[gmcCounts] = quanta[gmcCounts].astype(int)
    # setting a GMC cluster
    CF = cluster_factor(a, gmcs)
    mass_distr = gmc_distribution(a) / np.array([gmc.M for gmc in gmcs])
    clusterNums = (CF * mass_distr / mass_distr[-1]).astype('int')
    clusterMass = sum(clusterNums * np.array([gmc.M for gmc in gmcs]))
    clusterVolume = sum(clusterNums * np.array([gmc.V for gmc in gmcs]))
    cluster_params = (clusterNums, clusterMass, clusterVolume)
    # plug the GMCs powerlaw-randomly
    quanta['M_plugged'] = 0
    quanta['V_plugged'] = 0
    # parallelize the rings fill-up
    num_cores = n_cores if n_cores is not None else multiprocessing.cpu_count()
    gmc_rings = Parallel(n_jobs=num_cores)(delayed(gmc_fill_single_ring)(
        i, quanta.loc[i], gmcs, mass_distr, cluster_params, verbose=verbose) for i in quanta.index)
    for i in range(len(gmc_rings)):
        quanta.loc[i] = gmc_rings[i]
    # add n(r) [cm^-3] and N_H(r) [cm^-2]
    quanta['n_r'] = (quanta.M_r.cumsum()*msun/(mu*mp))/quanta.V_r.cumsum()
    quanta['NH_r'] = [simpson(quanta['n_r'].iloc[:i], 
        x=pc*quanta.r.iloc[:i]) for i in range(1, len(quanta.index)+1)]
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    quanta.to_csv('%sGMCs_Nradii%1d_a%1d.csv' % (outfolder, len(quanta), int(1e2*a)))
    return quanta


def baseline_sled(quanda, gmcs, logLX, FUVparams, flatNH=None, G0floor=False, Jmax=13):
    quanta = quanda.copy()
    # XDR (NH is in logLX_1_100, but we compute it only if logNH>22)
    if flatNH:
        if flatNH > 22:
            logLX_1_100 = logLX - np.log10(0.256) - 0.9 * (flatNH-22)
        else:
            logLX_1_100 = logLX - np.log10(0.256)
    else:
        NH_r = quanta['NH_r'].to_numpy()
        logNHr22 = np.array([np.log10(x)-22 if x > 0 else 0 for x in NH_r])
        logLX_1_100 = logLX - np.log10(0.256) - 0.9 * logNHr22
    quanta['logFX_r'] = logLX_1_100 - np.log10(4*pi * (pc*quanta['r'])**2)
    logFX_min = min(logFX) - 0.125
    # PDR
    Ie, Re, n = FUVparams
    G0_r = np.array([sersic(x, Ie, Re, n) for x in quanta['r']/1e3]) / 1.6e-3
    quanta['logG0_r'] = np.log10(G0_r)
    if G0floor:
        quanta.loc[(quanta['logG0_r'] < min(logG0)), 'logG0_r'] = min(logG0)
    # jump at the first massive radial bin but stop before flux too low
    iMassive = quanta[quanta['M_plugged'] > 0].index
    iPDR = [i for i in iMassive if i not in quanta[quanta['logG0_r'] < (min(logG0) - 0.125)].index]
    iXDR = [i for i in iMassive if i not in quanta[quanta['logFX_r'] < (min(logFX) - 0.125)].index]
    qPDR, qXDR = quanta.loc[iPDR], quanta.loc[iXDR]
    qPDR['QlogG0'] = [gnames[np.abs(logG0 - quanta.loc[i, 'logG0_r']).argmin()] for i in iPDR]
    qXDR['QlogFX'] = [xnames[np.abs(logFX - quanta.loc[i, 'logFX_r']).argmin()] for i in iXDR]
    basePDR, baseXDR = np.zeros(Jmax), np.zeros(Jmax)
    gmcCounts = ['N_%s' % gmc.name for gmc in gmcs]
    for i, gmc in enumerate(gmcs):
        pdr_gmc, xdr_gmc = gmc.pdr_xdr()
        Npdr = qPDR.groupby('QlogG0')[gmcCounts[i]].sum()
        Nxdr = qXDR.groupby('QlogFX')[gmcCounts[i]].sum()
        basePDR += (Npdr * pdr_gmc[pdr_gmc.columns[:Jmax]].loc[Npdr.index].T).T.sum().to_numpy()
        baseXDR += (Nxdr * xdr_gmc[xdr_gmc.columns[:Jmax]].loc[Nxdr.index].T).T.sum().to_numpy()
    return basePDR[:Jmax], baseXDR[:Jmax], basePDR[:Jmax] + baseXDR[:Jmax]


def chi_sled(obSLED, fitSLED, chi_thresh=None, Nfree=0):
    '''
    Calculates chi-square between observed and expected CO SLEDs
    obSLED is a list of SLEDs: data, lower err, upper err, upper limits (0 or 1)
    chi_thresh=0.15 means ob_err = 0.15 * data if the error is lower than that
    '''
    chi = 0
    for j in range(len(obSLED[0])):
        if not np.isnan(obSLED[0][j]):
            if obSLED[3][j] != 1:
                ob_err = np.mean([obSLED[1][j], obSLED[2][j]])
                if chi_thresh and (ob_err < chi_thresh * obSLED[0][j]):
                    ob_err = chi_thresh * obSLED[0][j]
                chi += ((obSLED[0][j] - fitSLED[j]) / ob_err)**2
            else:
                chi += ((0 - fitSLED[j]) / obSLED[0][j])**2
    non_detections = sum(np.isnan(np.array(obSLED[0], dtype=float)))
    non_detections += sum(obSLED[3] == 1)
    dof = len(obSLED[0]) - non_detections - Nfree
    if dof <= 0:
        red_chi = 1e9
    else:
        red_chi = chi / dof
    return chi, red_chi


def COfit(obSLED, quanta, gmcs, logLX, FUVparams, G0floor, Jmax=13, chi_thresh=None):
    '''
    Fit the observed CO SLED with the baseline model
    It will return the best-fit (alphaCO, logNH, red_chi)
    '''
    logNH_values = np.arange(22., 25.01, 0.1)
    alphaCO_values = np.logspace(-2, 1, 25) * 4.3
    minimalia = [np.nan, np.nan, 1e9]  # initialize minimalia[2]=redchi to a large value
    for logNH_temp in logNH_values:
        fitSLED = baseline_sled(quanta, gmcs, logLX, FUVparams,
                                flatNH=logNH_temp, G0floor=G0floor, Jmax=Jmax)[2]
        for norm_temp in alphaCO_values/4.3:
            redchi_temp = chi_sled(obSLED, norm_temp*fitSLED, chi_thresh, Nfree=2)[1]
            if redchi_temp <= minimalia[2]:
                minimalia = [logNH_temp, norm_temp, redchi_temp]
    logNHflat, alphaCO, redchi = minimalia[0], minimalia[1] * 4.3, minimalia[2]
    return (logNHflat, alphaCO, redchi)

