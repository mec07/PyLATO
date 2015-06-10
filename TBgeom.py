"""
Created on Monday April 20, 2015

@author: Andrew Horsfield

This module manages the geometry of the system (atomic coordinates etc)
"""
#
# Import modules
#
# import numpy as np
# import math as m

import TBIO


def init(JobDef):
	"""initialise the geometry."""
	global NAtom, Pos, AtomType
	#
	# Read in the geometry from file
	NAtom, Pos, AtomType = TBIO.ReadGeom(JobDef['gy_file'])
	#
	# Write out the geometry
	TBIO.WriteXYZ(NAtom, 'Hello', AtomType, Pos)