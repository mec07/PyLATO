#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Created on Thursday, June 18, 2015

@author: Marc Coury and Max Boleininger

Orthogonal tight-binding model for hydrocarbons

"""
#
# Import the modules that will be needed
import os, sys
import numpy as np
from scipy.interpolate import UnivariateSpline
import math
import commentjson

class GoodWin:
	"""Orthogonal tight-binding model for hydrocarbons

	The model follows the publication from:
	Computational materials synthesis. I. A tight-binding scheme for hydrocarbons
	A. P. Horsfield, P. D. Godwin, D. G. Pettifor, and A. P. Sutton
	PHYSICAL REVIEW B VOLUME 54, NUMBER 22 1 DECEMBER 1996-II

	Attributes:
		c0, c1, c2, c3: polynomial tail coefficients

	Methods:
		radial: evaluate the radial function
	"""

	def __init__(self, v0 = 0, r0 = 0, rc = 0, rcut = 0, r1 = 0, n = 0, nc = 0):
		"""Initialise the radial function from the supplied parameters.

		Keyword arguments:
			v0, r0, rc, rcut, r1, n, nc: a description can be found in the paper
		"""

		self.v0   = v0
		self.r0   = r0
		self.rc   = rc
		self.rcut = rcut
		self.r1   = r1
		self.n    = n
		self.nc   = nc

		dr = rcut - r1
		self.c0 = self.radial(r1)
		self.c1 = -n * (1 + nc * np.power(r1/rc, nc)) * self.radial(r1)/r1
		self.c2 = -2*self.c1 / dr - 3*self.c0 / dr/dr
		self.c3 = self.c1 / dr/dr + 2*self.c0 / dr/dr/dr

	def radial(self, r):
		"""Evaluate the radial function as a function of distance r

		Arguments:
			r: distance
		"""

		if r <= self.r1:
			return self.v0 * np.power(self.r0/r, self.n) * np.exp(self.n * 
				(-np.power(r/self.rc, self.nc) + np.power(self.r0/self.rc, self.nc)))
		if self.rcut > r > self.r1:
			x = r - self.r1
			return self.c0 + x*(self.c1 + x*(self.c2 + x*self.c3))
		else:
			return 0.0

class MatrixElements:
	"""Constructor for the tightbinding model

	This class constructs the model, and stores model-related data.
	It also offers methods to build the radial functions and pairpotentials.

	MatrixElements always needs to have the named methods helements,
	slements (for overlap), and pairpotentials (for forces and energy).
	The TBH assumes the model to have a MatrixElements class with these
	methods.

	This model supports:
		tabularisation (optional)
		pairpotentials
		NO overlap

	Attributes:
		atomic: atomic model data, such as species, on-site energy, orbitals, etc
		data: imported json hamiltonian dictionary
		pairpotentials: imported json pairpotentials dictionary
		tabularisation: imported json tabularisation dictionary
		function_grid: list of all interatomic hamiltonian functions
		pairpotential_grid: list of all interatomic pairpotential functions 

	Methods:
		helements: compute hopping integral for supplied atomic species
		pairpotential: compute pairpotential for supplied atomic species

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
		self.pairpotentials = modeldata['pairpotentials']
		self.embedded = modeldata['embedding']
		self.tabularisation = modeldata['tabularisation']

		# Allocate space for five integrals. This model includes up to l=1 orbitals,
		# hence five integrals (ss_sigma, sp_sigma, ps_sigma, pp_sigma, pp_pi) are 
		# evaluated simultaneously.
		self.v = np.zeros(5, dtype='double')

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

		# Interaction grid for a diatomic model. The values are placeholders.
		#
		# Species 1 (hydrogen) is max l=0, species 2 (carbon) has max l=1.
		# The grid stores the radial functions required to compute the integrals
		# between species 1 and species 2.

		shells = [[1, 3], [3, 5]]

		function_grid = [
						[[0], [0, 0, 0]], 
						[[0, 0, 0], [0, 0, 0, 0, 0]]
						]

		# Loop over interactions, assinging radial functions to the grid
		for i, species1 in enumerate(shells):
			for j, species2 in enumerate(species1):
				for radial in range(species2):
					function_grid[i][j][radial] = GoodWin(**self.data[i][j][radial]).radial

		# Interaction grid for pairpotential function
		pairpotential_grid = [[0, 0], [0, 0]]

		# Loop over interactions, assinging pairpotential functions to the grid
		for i, species1 in enumerate(shells):
			for j, species2 in enumerate(species1):
				pairpotential_grid[i][j] = GoodWin(**self.pairpotentials[i][j]).radial

		# Pair potential embedded function
		embed = lambda x: x*(self.embedded['a1'] + x*(self.embedded['a2'] + x*(self.embedded['a3'] + x*self.embedded['a4'])))

		# Embed pairpotential functions
		for i, species1 in enumerate(shells):
			for j, species2 in enumerate(species1):
				pairpotential_grid[i][j] = lambda x: embed(pairpotential_grid[i][j](x))	

		# Optionally interpolate radial functions
		if self.tabularisation["enable"] == 1:
			# Range of radii for the interpolating function
			rvalues = np.arange(0.5, 2.6, self.tabularisation["resolution"], dtype="double")
			interp_settings = {"k": 3, "s": 0, "ext": "zeros"}

			# Loop over interactions, interpolating radial functions
			for i, species1 in enumerate(shells):
				for j, species2 in enumerate(species1):
					for radial in range(species2):
						yvalues = [function_grid[i][j][radial](r) for r in rvalues]
						function_grid[i][j][radial] = UnivariateSpline(rvalues, yvalues, **interp_settings)

		self.function_grid = function_grid
		self.pairpotential_grid = pairpotential_grid

	def helements(self, r, atomicspecies_1, atomicspecies_2):
		"""Orthogonal model for hydrocarbons."""

		# Pick out the right functions for the bond
		funcs = self.function_grid[atomicspecies_1][atomicspecies_2]

		# Compute the hopping integrals for the reference geometry
		for i, radial_function in enumerate(funcs):
			self.v[i] = radial_function(r)

		# Return integrals and indices
		return self.v, self.v_bgn, self.v_end

	def pairpotential(self, r, atomicspecies_1, atomicspecies_2):
		"""Embedded pairpotentials for hydrocarbons."""

		radial_function = self.pairpotential_grid[atomicspecies_1][atomicspecies_2]
		return radial_function(r)

if __name__ == "__main__":
	# Print out some matrix elements if run as a script.

	model = MatrixElements("TBhydrocarbons.json")

	xvals = np.arange(0.5, 3, 0.01)
	yvals = np.array([model.pairpotential_grid[0][1](x) for x in xvals])

	for i,j in enumerate(xvals):
		print xvals[i], yvals[i]



