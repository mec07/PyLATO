"""
Created on Monday April 20, 2015

@author: Andrew Horsfield

This module manages the geometry of the system (atomic coordinates etc)
"""
#
# Import modules

import TBIO
import numpy as np
from Verbosity import *


def init(JobClass):
	"""initialise the geometry."""	
	#
	# Read in the geometry from file
	NAtom, Pos, AtomType = TBIO.ReadGeom(JobClass.Def['gy_file'])
	#
	# Write out the geometry
	TBIO.WriteXYZ(JobClass, NAtom, 'Hello', AtomType, Pos)
	#

	# Transfer geometry to the JobClass
	JobClass.NAtom    = NAtom
	JobClass.Pos      = Pos
	JobClass.AtomType = AtomType
	JobClass.NOrb     = [JobClass.Atomic[str(JobClass.AtomType[a])]['NOrbitals'] for a in range(JobClass.NAtom)]

	verboseprint(JobClass.Def['verbose'],"Atom positions:")
	verboseprint(JobClass.Def['verbose'],JobClass.Pos)