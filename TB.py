#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Created on Sunday April 12, 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

This is the main program for computing the eigenvalues and eigenfunctions for a
noncollinear tight binding model chosen in the JobDef.json file. The input file
can be given a different name, but it must be specified when running this
programme. To run the programme from the commandline type:

./TB.py specificationfile.json

where "specificationfile.json" can be any name as long as it's a json file.

Currently this programme works with Python 2.7.5.

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
import os, sys

def main():
    """Initialise the program."""

    if len(sys.argv) > 1:
        jobpath = sys.argv[1]
        if os.path.exists(jobpath) == True:
            Job = TBinit.InitJob(jobpath)
        else:
            print "ERROR: Unable to find job file:", jobpath
            sys.exit()
    else:
        print "No Job file specified. Proceeding with default JobDef.json."
        if os.path.exists("JobDef.json") == True:
            Job = TBinit.InitJob("JobDef.json")
        else:
            print "ERROR: Unable to find default job file: JobDef.json"
            sys.exit()

    # Check to see if my Hamiltonians are being used and hence if the pulay mixing is required
    if Job.Def['Hamiltonian'] in ('scase','pcase','dcase','vectorS'):
        # myHami is a flag
        myHami = True
    #
    # Build the non-self-consistent Hamiltonian (incl hopping and spin-orbit)
    Job.Hamilton.buildHSO()
    #
    # Allocate memory for the eigenvalues and eigenvectors
    Job.e   = np.zeros( Job.Hamilton.HSOsize, dtype='double')
    Job.psi = np.zeros((Job.Hamilton.HSOsize, Job.Hamilton.HSOsize), dtype='complex')

    
    # Initial step to solve H0 and initialise the Mulliken chareges

    # Diagonalise the HSO matrix
    Job.e, Job.psi = np.linalg.eigh(Job.Hamilton.HSO)
    # Occupy the orbitals according to the Fermi distribution
    Job.Electron.occupy(Job.e, Job.Def['el_kT'], Job.Def['mu_tol'], Job.Def['mu_max_loops'])
    #
    # Build the density matrix
    Job.Electron.densitymatrix()
    # if it's one of my Hamiltonians then we need to initialise rhotot
    if myHami:
        Job.Electron.rhotot = Job.Electron.rho
    #
    # Compute the net charge on each site
    Job.Hamilton.q = Job.Electron.chargepersite()

    # Compute the net spin on each site
    Job.Hamilton.s = Job.Electron.spinpersite()

    # Make the Fock matrix self-consistent
    SCFerror = 1.0e+99
    # flag to indicate if self-consistency has been obtained.
    SCFflag = False

    # If self-consistency is not asked for, then only do one iteration
    #if Job.Def['scf_on'] == 0:
    #   max_loops = 1
    #else:

    if Job.Def["scf_on"] == 1:
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
            Job.Electron.occupy(Job.e, Job.Def['el_kT'], Job.Def['mu_tol'], Job.Def['mu_max_loops'])
            #
            # Build the density matrix
            Job.Electron.densitymatrix()

            if myHami:
                # Compare the difference between the new and the old on-site density matrix elements
                SCFerror = Job.Electron.SCFerror()
                verboseprint(Job.Def['verbose'], 'SCF loop = ', ii+1, '; SCF error = ', SCFerror)
                # Check if the SCF error is still larger than the tolerance
                if SCFerror > Job.Def['scf_tol']:
                    # Update the density matrix by Pulay mixing
                    Job.Electron.pulay()
                else:
                    SCFflag = True
                    break

            else:
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
                verboseprint(Job.Def['extraverbose'], 'SCF charges = ', Job.Hamilton.q)
                #
                # IF SCF is on check if the SCF error is still larger than the tolerance.
                if SCFerror > Job.Def['scf_tol']:
                    #
                    # Update the input charges and spins
                    Job.Hamilton.q = Job.Hamilton.q + Job.Def['scf_mix'] * (q-Job.Hamilton.q)
                    Job.Hamilton.s = Job.Hamilton.s + Job.Def['scf_mix'] * (s-Job.Hamilton.s)
                #
                # If SCF error is smaller than or equal to the tolerance then leave loop
                else:
                    SCFflag = True
                    break

        # Print out number of SCF loops taken
        verboseprint(Job.Def['verbose'], "Number of SCF loops: ", ii+1)
        # if self-consistency is not ofbtained the throw an error and exit.
        if SCFflag is False:
            print "ERROR: Self-consistency not obtained within maximum number of cycles: ", max_loops
            sys.exit()


    verboseprint(Job.Def['verbose'], "Energy eigenvalues: ")
    verboseprint(Job.Def['verbose'], Job.e)
    if Job.Def['Hamiltonian'] == "standard":
        verboseprint(Job.Def['extraverbose'], "Mulliken charges: ", Job.Hamilton.q)
        verboseprint(Job.Def['extraverbose'], (Job.Hamilton.s).T)
    #
    # Write out the spins for each orbital
    TBIO.PsiSpin(Job)


    ############################
    # DEBUGGING NEW FUNCTIONS: #
    ############################
    TBIO.WriteOrbitalOccupations(Job,"occupations.txt")
    Job.Electron.magnetic_correlation(0,1)

if __name__ == "__main__":
    # Execute the main code if run as a script.
    main()
