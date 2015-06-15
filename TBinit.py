"""
Created on Sunday April 19, 2015

@author: Andrew Horsfield and Marc Coury

This module carries out the needed initialisation tasks
"""
#
# Import modules
import TBgeom
import TBH
import TBelec
import commentjson


class InitJob:
    """Sets up the job, builds the initial geometry, hamiltonian, and electronic structure."""
    def __init__(self, jobfile, atomicfile):
        """Initialise the job."""
        # Set up variables that define the job in a dictionary
        with open(jobfile,'r') as inputfile:
            self.Def = commentjson.loads(inputfile.read())

        with open(atomicfile,'r') as afile:
            self.Atomic = commentjson.loads(afile.read())
        # Initialise the geometry
        TBgeom.init(self)
        #
        # Initialise the Hamiltonian class
        self.Hamilton = TBH.Hamiltonian(self)
        #
        # Initialise the electron module
        self.Electron = TBelec.Electronic(self)