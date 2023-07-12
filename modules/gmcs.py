# license etc


# IMPORT PACKAGES

import numpy as np
import pandas as pd


# FIXED PARAMETERS
pi = np.pi
mp = 1.6726e-24 # grams
mu = 1.22 # mean molecular weight
pc = 3.086e18 # cm



### GMC class

class GMC:
    def __init__(self, name, M, R=None, rho0=None, Mach=None, Temp=None, Nclumps=None):
        self.name = name
        self.M = M
        self.R = R if R is not None else 0
        self.V = (4/3) * pi * (self.R * pc)**3
        self.rho0 = rho0 if rho0 is not None else 0
        self.Mach = Mach if Mach is not None else 0
        self.Temp = Temp if Temp is not None else 0
        self.n0 = rho0 / (mu * mp) if rho0 is not None else 0
        self.Nclumps = Nclumps if Nclumps is not None else 0
        # if the clumps have been extracted and the PDR/XDR files are ready:
        if Nclumps is not None:
            try:
                self.PDR = pd.read_csv('./data/GMC_%s_PDR.csv' % self.name, index_col=0)
                self.XDR = pd.read_csv('./data/GMC_%s_XDR.csv' % self.name, index_col=0)
            except FileNotFoundError:
                print('Missing PDR and/or XDR files for GMC %s' % self.name)
    
    def pdr_xdr(self):
        try:
            pdr = pd.read_csv('./data/GMC_%s_PDR.csv' % self.name, index_col=0)
            xdr = pd.read_csv('./data/GMC_%s_XDR.csv' % self.name, index_col=0)
            return pdr, xdr
        except FileNotFoundError:
            print('PDR and XDR files for GMC %s are not ready!' % gmc.name)
    
        



### GMC lists

# import the GMC csv with all the GMC parameters
e23csv = pd.read_csv('/media/phd/hub/gmchub/data/GMC_e23.csv', index_col=0)

# generate the GMC list
e23list = []
for name in e23csv.columns[:-1].to_list():
    gmc = e23csv[name]
    e23list.append(
        GMC(
            name = name,
            M = 10**gmc.loc['logM'], # Msun
            R = 10**gmc.loc['logR'], # pc
            rho0 = mu * mp * 10**gmc.loc['logn0'], # cm^-3
            Mach = gmc.loc['Mach'],
            Temp = gmc.loc['Temp'],
            Nclumps = gmc.loc['Nclumps']
            )
        )

