# galaxySLED
A new physically-motivated model for estimating the molecular line emission in active galaxies

## What do you need it for
We developed a code to estimate the CO emission in AGN-host galaxies. The underlying model is described in Esposito et al. (submitted), and it has been tested with the data presented in Esposito et al. 2022.

## Download the code and the data
- Select the drop-down menu "Code" and click "Download ZIP"
- Extract it in your favourite location

## Set up the environment
I suggest you to create a new environment.
If you have `conda`, just type in a terminal `conda create --name myenv`.
Once created, activate it with the command `conda activate myenv`.
Now you can install all the necessary packages within this environment.

#### Necessary Python packages
- `numpy`
- `pandas`
- `scipy`
- `joblib`
- `multiprocessing`
- `matplotlib` (this is needed only in the notebook)

## Run the python notebook `gmc_notebook.ipynb` to learn how to use the code
The notebook contains a walkthrough and a real-galaxy example for calculating the CO SLED (with plots)

## What are the contents of the code directories
The `data` directory contains the PDR and XDR emission for different Giant Molecular Clouds (GMCs). At the moment there is only one model of 15 GMCs (which is the one described in Esposito et al., subm.). There is one file, `GMC_e23.csv`, which contains the description of each GMC (as their masses, radii, etc). The other files, 2 for each GMC, contain the PDR and XDR estimated emission: every column is a CO line, where `CO4` means the CO(4-3) line, and every row is a different incident flux.

The `modules` directory contains the Python modules with the functions that run the different parts of the code. The `gmcs.py` module contains the definition of the GMC class and the list of the available built-in GMCs. The `functions.py` module contains all the useful functions.

The `ngc7469` directory contains a single file, which can be reproduced by the notebook. Every row is a different galactocentric radius, and the columns are the radial profiles of mass, volume, number of GMCs, column density, etc.
