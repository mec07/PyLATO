"""
Created on Thursday, June 18, 2015

@author: Andrew Horsfield and Marc Coury

This is a simple exponential model for an sp system.

"""
#
# Import the modules that will be needed
import os, sys
import numpy as np
import math
import commentjson
from Verbosity import *

class MatrixElements:
	"""Constructor for the tightbinding model

	This class constructs the model, and stores model-related data.
	It also offers methods to build the radial functions and pairpotentials.

	MatrixElements always needs to have the named methods helements,
	slements (for overlap), and pairpotentials (for forces and energy).
	The TBH assumes the model to have a MatrixElements class with these
	methods.

	This model has:
		NO tabularisation
		NO pairpotentials
		NO overlap

	Attributes:
		atomic: atomic model data, such as species, on-site energy, orbitals, etc
		data: imported json hamiltonian dictionary
		function_grid: list of all interatomic hamiltonian functions

	Methods:
		helements: compute hopping integral for supplied atomic species

	"""
    def __init__(self, modelpath):
        """Import model data, initialise orbital pair index matrix."""

        # Catch invalid model path
        if os.path.exists(modelpath) == False:
            print "ERROR: Unable to open tight binding model file:", modelpath
            sys.exit()

        # Import the tight binding model parameters
        with open(modelpath, 'r') as modelfile:
            modeldata = commentjson.loads(modelfile.read())

        self.atomic = modeldata['species']
        self.data = modeldata['hamiltonian']

		# Allocate space for five integrals. This model includes up to l=1 orbitals,
		# hence five integrals (ss_sigma, sp_sigma, ps_sigma, pp_sigma, pp_pi) are 
		# evaluated simultaneously.
        self.v = np.zeros(5, dtype='double')

        # Generate pair of indices for each pair of shells, showing which values
        # of v to use
        v_bgn = np.zeros((2, 2), dtype='double')
        v_end = np.zeros((2, 2), dtype='double')
        #
        # ss
        v_bgn[0, 0] = 0
        v_end[0, 0] = 1
        #
        # sp
        v_bgn[0, 1] = 1
        v_end[0, 1] = 2
        #
        # ps
        v_bgn[1, 0] = 2
        v_end[1, 0] = 3
        #
        # pp
        v_bgn[1, 1] = 3
        v_end[1, 1] = 5

        self.v_bgn = v_bgn
        self.v_end = v_end

    # Functions used to evaluate tight binding models
    def helements(self, r, atomicspecies_1, atomicspecies_2):
        """Simple exponential model for an sp system."""

        # Pick out the right coefficients for the bond
        coeffs = self.data[atomicspecies_1][atomicspecies_2]

        # Compute the hopping integrals for the reference geometry
        self.v[0] = coeffs['vsss'] * math.exp( -coeffs['kss'] * (r-coeffs['r0']) )
        self.v[1] = coeffs['vsps'] * math.exp( -coeffs['ksp'] * (r-coeffs['r0']) )
        self.v[2] = coeffs['vpss'] * math.exp( -coeffs['kps'] * (r-coeffs['r0']) )
        self.v[3] = coeffs['vpps'] * math.exp( -coeffs['kpp'] * (r-coeffs['r0']) )
        self.v[4] = coeffs['vppp'] * math.exp( -coeffs['kpp'] * (r-coeffs['r0']) )

        # Return integrals and indices
        return self.v, self.v_bgn, self.v_end