# galaxySLED: a code to reproduce and fit a galaxy CO SLED
A new physically-motivated model for estimating the molecular line emission in active galaxies

## What do you need it for
We developed a code to estimate the CO emission in AGN-host galaxies. The underlying model is described in Esposito et al. (submitted), and it has been tested with the data presented in Esposito et al. 2022.

#### What you have to input
To produce the baseline CO SLED you need to feed the code with:
- The molecular gas mass of your object
- The optical radius
- The intrinsic nuclear X-ray luminosity
- The three Sersic parameters for the FUV flux

With these input data, the code will produce the expected CO SLED of your object: this is called the "Baseline model".
<img src="./ngc7469/ngc7469_baseline.png" alt="Baseline CO SLED of NGC 7469" width="500"/>

If you already have the luminosity of some CO lines, you can make the baseline model fit the observed data.
Input the CO luminosities with their errors, it works with upper limits as well.
The output will be:
- The best-fit CO SLED, up to CO(30-29)
- The best-fit CO-to-H2 conversion factor $\alpha_{CO}$
- The best-fit X-ray attenuation column density $N_H$

<img src="./ngc7469/ngc7469_bestfit.png" alt="Best-fit CO SLED of NGC 7469" width="500"/>

## Download the code and set up the environment
I suggest you to create a new environment.
If you have `conda`, just type in a terminal `conda create --name myenv`.
Once created, activate it with the command `conda activate myenv`.
Now you can install all the necessary packages within this environment.

#### Necessary Python packages
- `numpy`
- `pandas`
- `scipy`
- `joblib` ([link](https://joblib.readthedocs.io/en/latest/installing.html))
- `multiprocessing`
- `matplotlib` (this is needed only in the notebook)

## Run the python notebook `galaxySLED_notebook.ipynb` to learn how to use the code
The notebook contains a walkthrough and a real-galaxy example for calculating the CO SLED (with plots)

## What are the contents of the code directories
The `data` directory contains the PDR and XDR emission for different Giant Molecular Clouds (GMCs). At the moment there is only one model of 15 GMCs (which is the one described in Esposito et al., subm.). There is one file, `GMC_e23.csv`, which contains the description of each GMC (as their masses, radii, etc). The other files, 2 for each GMC, contain the PDR and XDR estimated emission: every column is a CO line, where `CO4` means the CO(4-3) line, and every row is a different incident flux.

The `modules` directory contains the Python modules with the functions that run the different parts of the code. The `gmcs.py` module contains the definition of the GMC class and the list of the available built-in GMCs. The `functions.py` module contains all the useful functions.

The `ngc7469` directory contains a single file, which can be reproduced by the notebook. Every row is a different galactocentric radius, and the columns are the radial profiles of mass, volume, number of GMCs, column density, etc.
