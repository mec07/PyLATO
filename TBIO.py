"""
Created on Sunday April 19, 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

This module contains functions that perform input and output operations
"""
#
# Import modules
import TBH
import numpy as np
from Verbosity import *

def DOS(e, nbin):
    """Produce a total density of states (NEEDS TO BE UPDATED TO PLOT THE DOS)"""
    hist, bin_edges = np.histogram(e, nbin)
    return hist, binedges


def PsiSpin(JobClass):
    """Write out the spin vector for each eigenstate."""
    spin = np.zeros( 3, dtype='double')
    rho  = np.zeros((2, 2), dtype='complex')
    verboseprint(JobClass.Def['verbose'],'state   sx     sy     sz')

    jH = JobClass.Hamilton
    
    for n in range(0, jH.HSOsize):
        #
        # Build the spin density matrix
        rho[0, 0] = np.vdot(JobClass.psi[0        :jH.H0size,  n], JobClass.psi[        0:jH.H0size, n])
        rho[0, 1] = np.vdot(JobClass.psi[0        :jH.H0size,  n], JobClass.psi[jH.H0size:jH.HSOsize,n])
        rho[1, 0] = np.vdot(JobClass.psi[jH.H0size:jH.HSOsize, n], JobClass.psi[        0:jH.H0size, n])
        rho[1, 1] = np.vdot(JobClass.psi[jH.H0size:jH.HSOsize, n], JobClass.psi[jH.H0size:jH.HSOsize,n])
        #
        # Build the spin expectation values
        spin[0] = (rho[0, 1] + rho[1, 0]).real
        spin[1] = (rho[0, 1] - rho[1, 0]).imag
        spin[2] = (rho[0, 0] - rho[1, 1]).real
        #
        # Write out the spins
        verboseprint(JobClass.Def['verbose'],'{0:4d}  {1:5.2f}  {2:5.2f}  {3:5.2f}'.format(n, spin[0], spin[1], spin[2]))


def WriteXYZ(JobClass, NAtom, Comment, AtomType, Pos):
    """
    Writes coordinates out to an XYZ file.

    Note that the file is opened for append, so new frames are added to the end of existing frames.

    NAtom is the number of atoms
    Comment is a string holding a one line comment
    AtomType is a list holding the index for the atom type for each atom
    Pos is a 3xNAtom double precision NumPy array holding the atomic positions

    """
    f_xyz = open('geometry.xyz', 'a')
    f_xyz.write('{0:d}\n'.format(NAtom))
    f_xyz.write('{0}\n'.format(Comment))
    for i in range(0, NAtom):
        f_xyz.write('{0} {1:11.6f} {2:11.6f} {3:11.6f}\n'.format(JobClass.Atomic[str(AtomType[i])]['ChemSymb'], Pos[i,0], Pos[i,1], Pos[i,2]))
    f_xyz.close()


def ReadGeom(filename):
    """
    This function reads in the atomic geometry.

    The geometry file has the structure:

    Number of atoms
    Type, x, y, z
    Type, x, y, z
    Type, x, y, z
    Type, x, y, z
    Type, x, y, z
    Type, x, y, z

    etc
    """

    # Open the file
    f_geom = open(filename, 'r')
    #
    # Read in the number of atoms
    NAtom = int(f_geom.readline())
    #
    # Allocate space for the atom data
    Pos = np.zeros((NAtom, 3), dtype='double')
    AtomType = np.zeros(NAtom, dtype='int')
    #
    # For each atom read in the data
    for i in range(0, NAtom):
        in_line = f_geom.readline()
        in_line = in_line.strip()
        AtomType[i], Pos[i, 0], Pos[i, 1], Pos[i, 2] = in_line.split(',')
    #
    # Close the file
    f_geom.close()
    #
    # Return the geometry
    return NAtom, Pos, AtomType