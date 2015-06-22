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

	This model has:
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

		# HH parameters
		hh_sss = GoodWin(**self.data[0][0]).radial

		# HC parameters
		hc_sss = GoodWin(**self.data[1][0]).radial
		hc_sps = GoodWin(**self.data[1][1]).radial
		hc_pss = GoodWin(**self.data[1][2]).radial

		# CH parameters
		ch_sss = GoodWin(**self.data[2][0]).radial
		ch_sps = GoodWin(**self.data[2][1]).radial
		ch_pss = GoodWin(**self.data[2][2]).radial

		# CC parameters
		cc_sss = GoodWin(**self.data[3][0]).radial
		cc_sps = GoodWin(**self.data[3][1]).radial
		cc_pss = GoodWin(**self.data[3][2]).radial
		cc_pps = GoodWin(**self.data[3][3]).radial
		cc_ppp = GoodWin(**self.data[3][4]).radial

		# Pair potentials
		hh_pap = GoodWin(**self.pairpotentials[0]).radial
		hc_pap = GoodWin(**self.pairpotentials[1]).radial
		ch_pap = GoodWin(**self.pairpotentials[2]).radial
		cc_pap = GoodWin(**self.pairpotentials[3]).radial

		# Pair potential embedded function
		embed = lambda x: x*(self.embedded['a1'] + x*(self.embedded['a2'] + x*(self.embedded['a3'] + x*self.embedded['a4'])))

		# Embed pair potentials
		hh_pap2 = lambda x: embed(hh_pap(x))
		hc_pap2 = lambda x: embed(hc_pap(x))
		ch_pap2 = lambda x: embed(ch_pap(x))
		cc_pap2 = lambda x: embed(cc_pap(x))		

		if self.tabularisation['enable'] == 1:
			# Range of radii for the interpolating function.
			rvalues = np.arange(0.5, 2.6, self.tabularisation['resolution'], dtype='double')

			interp_settings = {"k": 3, "s": 0, "ext": "zeros"}
			hh_sss = UnivariateSpline(rvalues, [hh_sss(r) for r in rvalues], **interp_settings)
			
			hc_sss = UnivariateSpline(rvalues, [hc_sss(r) for r in rvalues], **interp_settings)
			hc_sps = UnivariateSpline(rvalues, [hc_sps(r) for r in rvalues], **interp_settings)
			hc_pss = UnivariateSpline(rvalues, [hc_pss(r) for r in rvalues], **interp_settings)
			
			ch_sss = UnivariateSpline(rvalues, [ch_sss(r) for r in rvalues], **interp_settings)
			ch_sps = UnivariateSpline(rvalues, [ch_sps(r) for r in rvalues], **interp_settings)
			ch_pss = UnivariateSpline(rvalues, [ch_pss(r) for r in rvalues], **interp_settings)
			
			cc_sss = UnivariateSpline(rvalues, [cc_sss(r) for r in rvalues], **interp_settings)
			cc_sps = UnivariateSpline(rvalues, [cc_sps(r) for r in rvalues], **interp_settings)
			cc_pss = UnivariateSpline(rvalues, [cc_pss(r) for r in rvalues], **interp_settings)
			cc_pps = UnivariateSpline(rvalues, [cc_pps(r) for r in rvalues], **interp_settings)
			cc_ppp = UnivariateSpline(rvalues, [cc_ppp(r) for r in rvalues], **interp_settings)

		# Store the radial functions in the function grid. index 0 is hydrogen, index 1 is carbon
		# Hence:  0,0 = hh; 0,1 = hc; 1,0 = ch; 1,1 = cc
		self.function_grid = [[[hh_sss], [hc_sss, hc_sps]], [[ch_sss, ch_sps], [cc_sss, cc_sps, cc_pss, cc_pps, cc_ppp]]]
		self.pairpotential_grid = [[hh_pap2, hc_pap2], [ch_pap2, cc_pap2]]

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
	yvals = np.array([model.pairpotential_grid[1][1](x) for x in xvals])

	for i,j in enumerate(xvals):
		print xvals[i], yvals[i]



