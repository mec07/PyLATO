"""
Created on Sunday April 19, 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

This module carries out the needed initialisation tasks
"""
#
# Import modules
import os, sys, importlib
import TBH
import TBelec
import TBIO
import commentjson
from Verbosity import *

class InitJob:
    """Set up the job, build the initial geometry, hamiltonian, and electronic structure."""
    def __init__(self, jobfile):
        """Initialise the job."""

        # Set up variables that define the job in a dictionary
        with open(jobfile, 'r') as inputfile:
            self.Def = commentjson.loads(inputfile.read())

        # Fetch the model name path from the Job file
        modelname = self.Def['model']
        modelpath = os.path.join("models", modelname + ".py")

        # Catch invalid model path
        if os.path.exists(modelpath) == False:
            print "ERROR: Unable to open tight binding model at %s. ", modelpath
            sys.exit()

        # Import the module responsible for the tight binding model
        model_module = importlib.import_module("models." + modelname)

        # Initialise the model class
        self.Model = model_module.MatrixElements(os.path.join("models", modelname + ".json"))

        # Initialise the geometry
        self.init_geom(self.Def['gy_file'])

        # Initialise the Hamiltonian class
        self.Hamilton = TBH.Hamiltonian(self)

        # Initialise the electron module
        self.Electron = TBelec.Electronic(self)

    def init_geom(self, filepath):
		"""initialise the geometry."""

		# Read in the geometry from file
		NAtom, Pos, AtomType = TBIO.ReadGeom(filepath)

		# Write out the geometry
		TBIO.WriteXYZ(self, NAtom, '', AtomType, Pos)

		# Transfer geometry to the JobClass
		self.NAtom    = NAtom
		self.Pos      = Pos

		self.AtomType = AtomType
		self.NOrb     = [self.Model.atomic[self.AtomType[a]]['NOrbitals'] for a in range(self.NAtom)]

		verboseprint(self.Def['verbose'], "Atom positions:")
		verboseprint(self.Def['verbose'], self.Pos)