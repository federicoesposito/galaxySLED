# galaxySLED: a code to reproduce and fit a galaxy CO SLED
# Copyright (C) 2024  Federico Esposito
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



# IMPORT PACKAGES
import pkgutil
from io import BytesIO
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
                pdrfile = 'resources/GMC_%s_PDR.csv' % self.name
                xdrfile = 'resources/GMC_%s_XDR.csv' % self.name
                self.PDR = pd.read_csv(BytesIO(pkgutil.get_data(__name__, pdrfile)), index_col=0)
                self.XDR = pd.read_csv(BytesIO(pkgutil.get_data(__name__, xdrfile)), index_col=0)
            except FileNotFoundError:
                print('Missing PDR and/or XDR files for GMC %s' % self.name)
    
    def pdr_xdr(self):
        try:
            pdrfile = 'resources/GMC_%s_PDR.csv' % self.name
            xdrfile = 'resources/GMC_%s_XDR.csv' % self.name
            pdr = pd.read_csv(BytesIO(pkgutil.get_data(__name__, pdrfile)), index_col=0)
            xdr = pd.read_csv(BytesIO(pkgutil.get_data(__name__, xdrfile)), index_col=0)
            return pdr, xdr
        except FileNotFoundError:
            print('PDR and XDR files for GMC %s are not ready!' % gmc.name)
    
        



### GMC lists

# import the GMC csv with all the GMC parameters
e24csv = pd.read_csv(BytesIO(pkgutil.get_data(__name__, 'resources/GMC_e24.csv')), index_col=0)
f24csv = pd.read_csv(BytesIO(pkgutil.get_data(__name__, 'resources/GMC_f24.csv')), index_col=0)
f24z65csv = pd.read_csv(BytesIO(pkgutil.get_data(__name__, 'resources/GMC_f24z65.csv')), index_col=0)

# generate the e24 GMC list
e24list = []
for name in e24csv.columns[:-1].to_list():
    gmc = e24csv[name]
    e24list.append(
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

# generate the f24 GMC list
f24list = []
for name in f24csv.columns[:-1].to_list():
    gmc = f24csv[name]
    f24list.append(
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

# generate the f24z65 GMC list
f24z65list = []
for name in f24z65csv.columns[:-1].to_list():
    gmc = f24z65csv[name]
    f24z65list.append(
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
