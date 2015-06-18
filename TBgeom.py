"""
Created on Monday April 20, 2015

@author: Andrew Horsfield

This module manages the geometry of the system (atomic coordinates etc)
"""
#
# Import modules

import TBIO

def init(JobClass):
	"""initialise the geometry."""
	#
	# Read in the geometry from file
	NAtom, Pos, AtomType = TBIO.ReadGeom(JobClass.Def['gy_file'])
	#
	# Write out the geometry
	TBIO.WriteXYZ(JobClass, NAtom, 'Hello', AtomType, Pos)

	# Transfer geometry to the JobClass
	JobClass.NAtom    = NAtom
	JobClass.Pos      = Pos
	JobClass.AtomType = AtomType