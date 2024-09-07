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


'''
This script contains the same first blocks of galaxySLED_notebook.ipynb
It is intended to test the gmc_fill function in a quick way
It is not intended for science use!
'''

# import galaxySLED
import galaxysled as gs

# import libraries to run this notebook (installed by default with galaxySLED)
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('\n--- You are currently using this python version:')
print(sys.version)
print('--- You are currently using python from this virtual environment:')
print(os.path.basename(sys.executable.replace('/bin/python','')), '\n')


# INPUT: galaxy data
Mmol_tot = 3.33e+10           # Msun
r25 = 13.76                   # kpc
logLX = 43.19                 # erg/s (2-10 keV)
FUVparams = 1.10, 1.35, 0.70  # erg/s/cm^2, kpc, -
SFR = 35                      # Msun/yr

# script parameters (logR_step=1 for quick tests, you should use the commented value of 0.05 or similar)
outfolder = os.getcwd() + '/ngc7469/'
rCO = 0.17e3*r25           # pc
logR_end = np.log10(2*rCO) # 2*rCO [pc]
logR_step = 1              # 0.05
a = 1.64                   # GMC mass PL distribution index

# GMCs data
gmcs = gs.e24list

# additional parameters
G0floor = True
Jmax = 13
n_cores = None
verbose = True


# RUN the gmc_fill function
galdf = gs.gmc_fill(
	outfolder, Mmol_tot, r25, logR_end, logR_step, a, gmcs, n_cores, verbose)


# final print
print('\n------> Test ended correctly\n')

