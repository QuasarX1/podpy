# podpy-refocused

[![PyPI - Version](https://img.shields.io/pypi/v/podpy-refocused.svg)](https://pypi.org/project/podpy-refocused)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/podpy-refocused.svg)](https://pypi.org/project/podpy-refocused)

Updaded version of the podpy code, forked from [turnerm/podpy](https://github.com/turnerm/podpy) at commit f6cb93cf4d30a7927fef9e282fe76555d4957488.

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)
- [Original README](#original-readme)

## Installation

### Production

Are you installing on an HPC faculity? Add any required `module load x` commands to the `modules.sh` file.  
Alternitivley, add a single line, sourcing one of the pre-existing options for current facuilities.

This software comes with a BASH installation script for use as a standalone peice of software:
```console
./install.sh
```
and then, to make it avalible as a commandline utility, add the following to your .bashrc file:
```
source <repo location>/load.sh
```
(don't forget to run `source ~/.bashrc` after adding the above!)

Alternitivley, to install into your own Python environment:
```console
pip install ./src/podpy-refocused
```

### Development/Testing

This project is developed using the [Hatch](https://hatch.pypa.io/latest/) build system. With Hatch installed, run (from the root of this project):
```console
hatch env create default
```
To use the newly created environment, run (from the root of this project):
```console
hatch shell default
```

## License

`podpy-refocused` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license (see [LICENSE.md](LICENSE.md)).

## Usage

To ensure the software has been installed correctly, try running the following from a scratch directory (it will create four image files):
```console
pod-run-example <repo location>/example_data/spectrum
```
This will produce the following files:
- h1.png
- c4.png
- o6.png
- example-tau-tau.png

## Original-README

__This is the content of the original project's README file, updated for markdown format:__  

This repository contains an implentation of the pixel optical depth (POD) method as described in `Turner et al. 2014, MNRAS, 445, 794`, which builds upon the description in `Aguirre et al. 2002, ApJ, 576, 1`.  
If you use the code, please cite these two works, which can be found here:  

http://adsabs.harvard.edu/abs/2014MNRAS.445..794T  
http://adsabs.harvard.edu/abs/2002ApJ...576....1A  
If you have any questions, comments, feature reqeusts etc. please contact the author (Monica Turner) at turnerm@mit.edu.

The example QSO spectrum of J010311+131617 was obtained from the Keck Observatory Database of Ionized Absorbers toward QSOs (KODIAQ; O'Meara et al. 2015, AJ, 150, 111; Lehner et al. 2014, ApJ 788, 119), which was funded through NASA ADAP grant NNX10AE84G. This research has made use of the Keck Observatory Archive (KOA), which is operated by the W. M. Keck Observatory and the NASA Exoplanet Science Institute (NExScI), under contract with the National Aeronautics and Space Administration. We also acknowledge the PI of the dataset, C.Steidel.  

### INSTRUCTIONS
------------

To install the package, make sure you are in the top directory that contains setup.py and type:

pip install . 

You may need root permissions. If you do not have them or do not want to use them, then try:

pip install --user .

To run the example, open up the python or ipython and run "example/example.py". This script is best run interactively as otherwise the figures may not hold.

### REQUIREMENTS
------------
numpy 
scipy 
matplotlib
astropy

### CONTENTS
--------
example/example.py: A working example of applying the code to a z=2.710 QSO. Best run interactively (from ipython type 'run example.py') as otherwise the figures may not hold. 
example/spectrum/J010311+131617_f.fits: Spectrum flux array 
example/spectrum/J010311+131617_e.fits: Spectrum error array
example/spectrum/J010311+131617_dla.mask: Spectrum DLA mask
example/spectrum/J010311+131617_badreg.mask: Spectrum bad region mask

podpy/Spectrum.py: A class that reads in a spectrum and formats it for input into Pod.py
podpy/Pod.py: The main class that recovers optical depths	
podpy/TauBinned.py: A class for creating objects that have the percentile of one ion as binned by another
podpy/universe.py: Physical constants
