#!/usr/bin/env python
"""
Created on Sunday April 12, 2015

@author: Andrew Horsfield and Marc Coury

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
from Verbosity import *
import numpy as np
import math
import sys

def main():
    """Initialise the program."""
    JobDef = TBinit.init()
    #
    # Build the non-self-consistent Hamiltonian (incl hopping and spin-orbit)
    TBH.BuildHSO(JobDef)
    #
    # Allocate memory for the eigenvalues and eigenvectors
    e = np.zeros(TBH.HSOsize, dtype='double')
    psi = np.zeros((TBH.HSOsize, TBH.HSOsize), dtype='complex')
    #
    # Make the Fock matrix self-consistent
    SCFerror = 1.0e+99
    # flag to indicate if self-consistency has been obtained.
    SCFflag = False
    # If self-consistency is not asked for, then only do one iteration
    if JobDef['scf_on'] == 0:
        max_loops = 1
    else:
        max_loops = JobDef['scf_max_loops']

    for ii in range(max_loops):
        #
        # Build the fock matrix (adds the density matrix dependent terms)
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
        q = TBelec.ChargePerSite()
        #
        # Compute the net spin on each site
        s = TBelec.SpinPerSite()
        #
        # Compute the error in the charges, and update the charges and spin
        SCFerror = math.sqrt(np.vdot(q-TBH.q, q-TBH.q)/TBgeom.NAtom)
        verboseprint(JobDef['verbose'], 'SCF loop = ', ii+1, '; SCF error = ', SCFerror)
        # Check if the SCF error is still larger than the tolerance
        if SCFerror > JobDef['scf_tol']:
            #
            # Update the input charges and spins
            TBH.q = TBH.q + JobDef['scf_mix']*(q-TBH.q)
            TBH.s = TBH.s + JobDef['scf_mix']*(s-TBH.s)
        # If SCF error is smaller than or equal to the tolerance then leave loop
        else:
            SCFflag = True
            break

    # if self-consistency is required then print out number of SCF loops taken
    if JobDef['scf_on'] == 1:
        verboseprint(JobDef['verbose'], "Number of SCF loops: ", ii+1)
        # if self-consistency is not obtained the throw an error and exit.
        if SCFflag is False:
            print "ERROR: Self-consistency not obtained within maximum number of cycles: ", max_loops
            sys.exit()


    verboseprint(JobDef['verbose'], "Energy eigenvalues: ")
    verboseprint(JobDef['verbose'], e)
    verboseprint(JobDef['extraverbose'], q)
    verboseprint(JobDef['extraverbose'], s.T)
    #
    # Write out the spins for each orbital
    TBIO.PsiSpin(JobDef['extraverbose'], e, psi)


if __name__ == "__main__":
    # Execute the main code if run as a script.
    main()
