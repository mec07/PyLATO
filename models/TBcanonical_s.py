#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Created on Friday, June 26, 2015

@author: Marc Coury and Max Boleininger

This is a simple canonical model for an s system.

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

    This model supports:
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
        """Import model data, initialise orbital pair index matrix.

        Arguments:
            modelpath: path to the model json file
        """

        # Catch invalid model path
        if os.path.exists(modelpath) == False:
            print "ERROR: Unable to open tight binding model file:", modelpath
            sys.exit()

        # Import the tight binding model parameters
        with open(modelpath, 'r') as modelfile:
            modeldata = commentjson.loads(modelfile.read())

        # Store imported data as attributes
        self.atomic = modeldata['species']
        self.data = modeldata['hamiltonian']

        # The model includes l=1 orbitals, hence space for 2 integrals 
        # (in order: pp_sigma and pp_pi) is made.
        self.v = np.zeros(2, dtype="double")

        # Generate pair of indices for each pair of shells, showing which values
        # of v to use. This follows from the Slater-Koster table.
        # 
        # Example: two interacting shells with max l of 0 (eg H-H).
        # l=0, l=0: ss_sigma, slice v from 0 to 1: v[0]
        #
        # This needs to be expanded for d and f orbitals, and is model-independent.
        self.v_bgn = np.array([[0]])
        self.v_end = np.array([[1]])

    # Functions used to evaluate tight binding models
    def helements(self, r, atomicspecies_1, atomicspecies_2):
        """Simple canonical model for a s system."""

        # Pick out the right coefficients for the bond
        coeffs = self.data[atomicspecies_1][atomicspecies_2]

        # Compute the hopping integrals for the reference geometry
        self.v[0] = coeffs['vsss'] * math.exp( -coeffs['kss'] * (r-coeffs['r0']) )

        # Return integrals and indices
        return self.v, self.v_bgn, self.v_end