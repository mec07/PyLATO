# PyLATO
Noncollinear Magnetic Tight Binding code

Created on Sunday April 12, 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

Currently this programme works with Python 2.7.5.

# Setup
First you will want to clone this repository from github and then you will want to create a python virtual environment (to find out more about virtual environments please refer to: http://docs.python-guide.org/en/latest/dev/virtualenvs/).
The following instructions will work for Linux and OSX operating systems.
It is of course possible to run this programme from a Windows machine but you will have to perform the Windows equivalent instructions.

To clone this repository type in a terminal window:
```git clone git@github.com:mec07/PyLATO.git```

For a python virtual environment you will need to install the `virtualenv` package. Type in a terminal window:

```pip install virtualenv```

When you are inside the PyLATO folder you can start the virtual environment and install all the packages type in a terminal window:

```virtualenv venv
pip install -r requirements.txt
source venv/bin/activate```

To exit from the virtual environment just type in a terminal window:

```deactivate```

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
