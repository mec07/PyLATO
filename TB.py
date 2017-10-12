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
from TBSelfConsistency import PerformSelfConsistency
from Verbosity import verboseprint
import numpy as np
import os
import sys


def initialisation():
    if len(sys.argv) > 1:
        jobpath = sys.argv[1]
        if os.path.exists(jobpath):
            Job = TBinit.InitJob(jobpath)
        else:
            print("ERROR: Unable to find job file:")
            print(jobpath)
            sys.exit()
    else:
        if os.path.exists("JobDef.json"):
            print("No Job file specified. Proceeding with default JobDef.json.")
            Job = TBinit.InitJob("JobDef.json")
        else:
            print("ERROR: Unable to find default job file: JobDef.json")
            sys.exit()

    return Job


def main():
    """Initialise the program."""
    Job = initialisation()

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
    #
    # Compute the net charge on each site
    Job.Hamilton.q = Job.Electron.chargepersite()

    # Compute the net spin on each site
    Job.Hamilton.s = Job.Electron.spinpersite()

    success = True
    if Job.Def["scf_on"] == 1:
        success = PerformSelfConsistency(Job)

    verboseprint(Job.Def['verbose'], "Energy eigenvalues: ")
    verboseprint(Job.Def['verbose'], Job.e)
    if Job.Def['Hamiltonian'] == "collinear":
        verboseprint(Job.Def['extraverbose'], "Mulliken charges: ", Job.Hamilton.q)
        verboseprint(Job.Def['extraverbose'], (Job.Hamilton.s).T)

    # Write out information about the simulation if it is specified in the job definition
    if success:
        print("\n\n\nSuccessfully completed calculation!")
        TBIO.WriteSimulationResults(Job)


if __name__ == "__main__":
    # Execute the main code if run as a script.
    main()
