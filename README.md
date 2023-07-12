# galaxySLED
A new physically-motivated model for estimating the molecular line emission in active galaxies

## What do you need it for
We developed a code to estimate the CO emission in AGN-host galaxies. The underlying model is described in Esposito et al. (submitted), and it has been tested with the data presented in Esposito et al. 2022.

## Download the code and the data
- Select Code and click "Download ZIP"
- Extract it in your favourite location

## Set up the environment
I suggest you to create a new environment. 
If you have `conda`, just type in a terminal `conda create --name myenv`.
Once created, activate it with the command `conda activate myenv`.
Now you can install all the necessary packages within this environment.

#### Necessary Python packages
- numpy
- pandas
- scipy
- joblib
- multiprocessing
- matplotlib (this is needed only in the notebook)

## Run the python notebook `gmc_notebook.ipynb` to learn how to use the code
The notebook contains a walkthrough and a real-galaxy example for calculating the CO SLED (with plots)
