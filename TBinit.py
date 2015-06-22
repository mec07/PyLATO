"""
Created on Sunday April 19, 2015

@author: Andrew Horsfield and Marc Coury

This module carries out the needed initialisation tasks
"""
#
# Import modules
import os, sys, importlib
import TBgeom
import TBH
import TBelec
import commentjson


class InitJob:
    """Set up the job, build the initial geometry, hamiltonian, and electronic structure."""
    def __init__(self, jobfile):
        """Initialise the job."""

        # Set up variables that define the job in a dictionary
        with open(jobfile, 'r') as inputfile:
            self.Def = commentjson.loads(inputfile.read())

        #with open(atomicfile, 'r') as afile:
        #    self.Atomic = commentjson.loads(afile.read())

        # Model Import
        #
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
        TBgeom.init(self)
        #
        # Initialise the Hamiltonian class
        self.Hamilton = TBH.Hamiltonian(self)
        #
        # Initialise the electron module
        self.Electron = TBelec.Electronic(self)