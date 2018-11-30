# PyLATO
Noncollinear Magnetic Tight Binding code

Created on Sunday April 12, 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

Currently this programme works with Python 3.

# Setup
First you will want to clone this repository from github and then you will want to create a python virtual environment (to find out more about virtual environments please refer to: http://docs.python-guide.org/en/latest/dev/virtualenvs/).
The following instructions will work for Linux and OSX operating systems.
It is of course possible to run this programme from a Windows machine but you will have to perform the Windows equivalent instructions.

To clone this repository type in a terminal window:

```
git clone git@github.com:mec07/PyLATO.git
```

For a python 3 virtual environment you will need to install the `virtualenv` package. Type in a terminal window:

```
pip install virtualenv
```

(on some computers you will have to specify python 3 pip by using `pip3` instead of just `pip` --- you can check your pip version using: `pip --version`).
When you are inside the PyLATO folder you can start the virtual environment and install all the packages type in a terminal window:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements-test.txt
```

(on some computers you will to specify python 3 by using `python3` instead of `python` --- you can check your python version using: `python --version`).
To exit from the virtual environment just type in a terminal window:

```
deactivate
```


# Tests
After you have setup the repository, and have activated the virtual environment you can run the tests.
Some tests have been added recently. To run the tests type in a terminal window:

```
pytest
```

To run the behave test suite, type the following into the terminal:

```
behave
```

You should see the tests get executed and see that all of the tests have passed.
You can also run them both using the make command:

```
make test
```


# Running the code
The default job definition file that PyLATO uses is `JobDef.json`, so if you just run

```
pylato/main.py
```

that is what it will look for.
You can specify an input file by including it in the command to run PyLATO on the commandline:

```
pylato/main.py specificationfile.json
```

where "specificationfile.json" can be any name as long as it's a json file.


# Set up a simulation
To alter the system being simulated you can do one of the following:
  * Alter settings in the specification file. See the example, `JobDef.json`.
  * Alter the parameters of the canonical, exponential or hydrocarbon tight binding model: change the corresponding json file in the models folder.
  * Change the tight binding model being used: Add a json and a py file to the models folder, following the example of the files currently present.


# Units
Units used are:
  * Length -- Angstroms
  * Energy -- eV
