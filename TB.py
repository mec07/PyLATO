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
import TBIO
import TBinit
from Verbosity import *
import numpy as np
import math
import sys

def main():
    """Initialise the program."""

    Job = TBinit.InitJob("JobDef.json")
    #
    # Build the non-self-consistent Hamiltonian (incl hopping and spin-orbit)
    Job.Hamilton.buildHSO()
    #
    # Allocate memory for the eigenvalues and eigenvectors
    Job.e   = np.zeros( Job.Hamilton.HSOsize, dtype='double')
    Job.psi = np.zeros((Job.Hamilton.HSOsize, Job.Hamilton.HSOsize), dtype='complex')

    #
    # Make the Fock matrix self-consistent
    SCFerror = 1.0e+99
    # flag to indicate if self-consistency has been obtained.
    SCFflag = False

    # If self-consistency is not asked for, then only do one iteration
    if Job.Def['scf_on'] == 0:
        max_loops = 1
    else:
        max_loops = Job.Def['scf_max_loops']

    for ii in range(max_loops):
        #
        # Build the fock matrix (adds the density matrix dependent terms)
        Job.Hamilton.buildfock()
        #
        # Diagonalise the Fock matrix
        Job.e, Job.psi = np.linalg.eigh(Job.Hamilton.fock)
        #
        # Occupy the orbitals according to the Fermi distribution
        Job.Electron.occupy(Job.e, Job.Def['el_kT'], 1.0e-15)
        #
        # Build the density matrix
        Job.Electron.densitymatrix()
        #
        # Compute the net charge on each site
        q = Job.Electron.chargepersite()
        #
        # Compute the net spin on each site
        s = Job.Electron.spinpersite()
        #
        # Compute the error in the charges, and update the charges and spin
        SCFerror = math.sqrt(np.vdot(q-Job.Hamilton.q, q-Job.Hamilton.q) / Job.NAtom)
        verboseprint(Job.Def['verbose'], 'SCF loop = ', ii+1, '; SCF error = ', SCFerror)
        # Check if the SCF error is still larger than the tolerance
        if SCFerror > Job.Def['scf_tol']:
            #
            # Update the input charges and spins
            Job.Hamilton.q = Job.Hamilton.q + Job.Def['scf_mix'] * (q-Job.Hamilton.q)
            Job.Hamilton.s = Job.Hamilton.s + Job.Def['scf_mix'] * (s-Job.Hamilton.s)
        # If SCF error is smaller than or equal to the tolerance then leave loop
        else:
            SCFflag = True
            break

    # if self-consistency is required then print out number of SCF loops taken
    if Job.Def['scf_on'] == 1:
        verboseprint(Job.Def['verbose'], "Number of SCF loops: ", ii+1)
        # if self-consistency is not obtained the throw an error and exit.
        if SCFflag is False:
            print "ERROR: Self-consistency not obtained within maximum number of cycles: ", max_loops
            sys.exit()


    verboseprint(Job.Def['verbose'], "Energy eigenvalues: ")
    verboseprint(Job.Def['verbose'], Job.e)
    verboseprint(Job.Def['extraverbose'], Job.Hamilton.q)
    verboseprint(Job.Def['extraverbose'], (Job.Hamilton.s).T)
    #
    # Write out the spins for each orbital
    TBIO.PsiSpin(Job)


if __name__ == "__main__":
    # Execute the main code if run as a script.
    main()
