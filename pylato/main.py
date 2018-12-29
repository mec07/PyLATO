#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sunday April 12, 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

This is the main program for computing the eigenvalues and eigenfunctions for a
noncollinear tight binding model chosen in the JobDef.json file. The input file
can be given a different name, but it must be specified when running this
programme. To run the programme from the commandline type:

pylato/pylato.py specificationfile.json

where "specificationfile.json" can be any name as long as it's a json file.

Currently this programme works with Python 2.7.5.

Units used are:
  Length -- Angstroms
  Energy -- eV
"""
#
# Import the modules to be used here
import numpy as np
import os
import sys

from pylato.init_job import InitJob
from pylato.pylato_IO import WriteSimulationResults
from pylato.self_consistency import PerformSelfConsistency
from pylato.verbosity import verboseprint


def main():
    """Initialise the program."""
    Job = InitJob(get_job_file())
    Job = execute_job(Job)

    print_results(Job)


def get_job_file():
    # Default job file
    jobpath = "JobDef.json"

    if len(sys.argv) > 1:
        jobpath = sys.argv[1]

    return jobpath


def execute_job(Job):
    #
    # Build the non-self-consistent Hamiltonian (incl hopping and spin-orbit)
    Job.Hamilton.buildHSO(Job)
    #
    # Build the fock matrix
    Job.Hamilton.buildFock(Job)
    #
    # Allocate memory for the eigenvalues and eigenvectors
    Job.e = np.zeros(Job.Hamilton.HSOsize, dtype='double')
    Job.psi = np.zeros((Job.Hamilton.HSOsize, Job.Hamilton.HSOsize), dtype='complex')

    # Initial step to solve Fock and initialise the Mulliken chareges

    # Diagonalise the fock matrix
    Job.e, Job.psi = np.linalg.eigh(Job.Hamilton.fock)
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

    if Job.Def["scf_on"] == 1:
        # This raises SelfConsistencyError if self-consistency is not obtained
        PerformSelfConsistency(Job)

    return Job


def print_results(Job):
    verboseprint(Job.Def['verbose'], "Energy eigenvalues: ")
    verboseprint(Job.Def['verbose'], Job.e)
    if Job.Def['Hamiltonian'] == "collinear":
        verboseprint(Job.Def['extraverbose'], "Mulliken charges: ", Job.Hamilton.q)
        verboseprint(Job.Def['extraverbose'], (Job.Hamilton.s).T)

    # Write out information about the simulation if it is specified in the job definition
    print("\n\n\nSuccessfully completed calculation!")
    WriteSimulationResults(Job)


if __name__ == "__main__":
    # Execute the main code if run as a script.
    main()
