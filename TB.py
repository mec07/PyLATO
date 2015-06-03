"""
Created on Sunday April 12, 2015

@author: Andrew Horsfield

This is the main program for computing the eigenvalues and eigenfunctions for a
tight binding model of a helical molecule that includes spin-orbit coupling

Units used are:
  Length -- Angstroms
  Energy -- eV
"""
#
# Import the modules to be used here
import TBH
import TBIO
import TBinit
import TBelec
import TBgeom
import numpy as np
import math as m
#
# Initialise the program
JobDef = TBinit.init()
#
# Build the non-self-consistent Hamiltonian, with spin-orbit terms included
TBH.BuildHSO(JobDef)
#
# Allocate memory for the eigenvalues and eigenvectors
e = np.zeros(TBH.HSOsize, dtype='double')
psi = np.zeros((TBH.HSOsize, TBH.HSOsize), dtype='complex')
#
# Make the Fock matrix self-consistent
SCFerror = 1.0e+99
while SCFerror > JobDef['scf_tol']:
    #
    # Build the fock matrix
    TBH.BuildFock(JobDef)
    #
    # Diagonalise the Fock matrix
    e, psi = np.linalg.eigh(TBH.Fock)
    #
    # Occupy the orbitals according to the Fermi distribution
    TBelec.Occupy(e, JobDef['el_kT'], 1.0e-15)
    #
    # Build the density matrix
    TBelec.DensityMatrix(psi)
    #
    # Compute the net charge on each site
    q = TBelec.ChargePerSite ()
    #
    # Compute the net spin on each site
    s = TBelec.SpinPerSite()
    #
    # If self-consistency is not asked for, then force the loop to stop
    if JobDef['scf_on'] == 0:
        SCFerror = 0.0
    #
    # Otherwise, compute the error in the charges, and update the charges and spin
    else:
        #
        # Compute the SCF error
        SCFerror = m.sqrt(np.vdot(q-TBH.q, q-TBH.q)/TBgeom.NAtom)
        print('SCF error = ', SCFerror)
        #
        # Update the input charges and spins
        TBH.q = TBH.q + JobDef['scf_mix']*(q-TBH.q)
        TBH.s = TBH.s + JobDef['scf_mix']*(s-TBH.s)
print(e)
print(q)
print(s.T)
#
# Write out the spins for each orbital
TBIO.PsiSpin (e, psi)
