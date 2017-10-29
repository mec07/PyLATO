#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Created on Thursday, June 18, 2015

@author: Andrew Horsfield and Marc Coury

This is a simple exponential model for an sp system.

"""
#
# Import the modules that will be needed
import os
import sys
import numpy as np
import math
import commentjson
from verbosity import *


class MatrixElements:
	"""Constructor for the tightbinding model

	This class constructs the model, and stores model-related data.
	It also offers methods to build the radial functions and pairpotentials.

	MatrixElements always needs to have the named methods helements,
	slements (for overlap), and pairpotentials (for forces and energy).
	The Hamiltonian assumes the model to have a MatrixElements class with these
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

		# The model includes up to l=1 orbitals, hence space for five integrals 
		# (in order: ss_sigma, sp_sigma, ps_sigma, pp_sigma, pp_pi) is made.
		self.v = np.zeros(5, dtype="double")

		# Generate pair of indices for each pair of shells, showing which values
		# of v to use. This follows from the Slater-Koster table.
		# 
		# Example: two interacting shells with max l of 1 (eg C-C).
		# l=0, l=0: ss_sigma, slice v from 0 to 1: v[0] (ss_sigma)
		# l=0, l=1: sp_sigma, slice v from 1 to 2: v[1] (sp_sigma)
		# l=1, l=0: ps_sigma, slice v from 2 to 3: v[2] (ps_sigma)
		# l=1, l=1: pp_sigma and pp_pi, slice v from 3 to 5: v[3] and v[4]
		# (pp_sigma and pp_pi)
		#
		# This needs to be expanded for d and f orbitals, and is model-independent.
		self.v_bgn = np.array([[0, 1], [2, 3]])
		self.v_end = np.array([[1, 2], [3, 5]])

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
