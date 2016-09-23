# PyLATO
Noncollinear Magnetic Tight Binding code

Created on Sunday April 12, 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

Currently this programme works with Python 2.7.5.

# Setup step
An third party library, PyDQED https://github.com/ReactionMechanismGenerator/PyDQED, is used in this code and this has to be setup.
Follow the instructions in the README.rst file in the pydqed folder to do a local installation (for a linux machine just type `make` from within that folder).


# Running the code
This is the main program for computing the eigenvalues and eigenfunctions for a noncollinear tight binding model chosen in the JobDef.json file. The input file can be given a different name, but it must be specified when running this programme. To run the programme from the commandline type:

./TB.py specificationfile.json
or
python TB.py specificationfile.json

where "specificationfile.json" can be any name as long as it's a json file.


# Set up a simulation
To alter the system being simulated you can do one of the following:
  * Alter settings in the specification file. See the example, JobDef.json.
  * Alter the parameters of the canonical, exponential or hydrocarbon tight binding model: change the corresponding json file in the models folder.
  * Change the tight binding model being used: Add a json and a py file to the models folder, following the example of the files currently present.


# Units
Units used are:
  * Length -- Angstroms
  * Energy -- eV
