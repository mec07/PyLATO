"""
Created on Thursday, October 12, 2017

@author: Marc Coury

This module performs the self consistency iterations.
"""

from verbosity import verboseprint
import numpy as np
import math


def PerformSelfConsistency(Job):
    # Make the Fock matrix self-consistent
    SCFerror = 1.0e+99
    # flag to indicate if self-consistency has been obtained.
    SCFflag = False

    # Check to see if a noncollinear Hamiltonian is being used and hence if the linear mixing is required
    isNoncollinearHami = False
    if Job.Def['Hamiltonian'] in ('scase', 'pcase', 'dcase', 'noncollinear'):
        isNoncollinearHami = True

    # if it's a noncollinear Hamiltonian we need to initialise rhotot
    if isNoncollinearHami:
        Job.Electron.rhotot = Job.Electron.rho

    max_loops = Job.Def['scf_max_loops']
    for ii in range(max_loops):
        #
        # Build the fock matrix (adds the density matrix dependent terms)
        Job.Hamilton.buildFock(Job)
        #
        # Diagonalise the Fock matrix
        Job.e, Job.psi = np.linalg.eigh(Job.Hamilton.fock)
        #
        # Occupy the orbitals according to the Fermi distribution
        Job.Electron.occupy(Job.e, Job.Def['el_kT'], Job.Def['mu_tol'], Job.Def['mu_max_loops'])
        #
        # Build the density matrix
        Job.Electron.densitymatrix()

        if isNoncollinearHami:
            # Compare the difference between the new and the old on-site density matrix elements
            SCFerror = Job.Electron.SCFerror()
            verboseprint(Job.Def['verbose'], 'SCF loop = ', ii+1, '; SCF error = ', SCFerror)
            # Check if the SCF error is still larger than the tolerance
            if SCFerror > Job.Def['scf_tol']:
                # Update the density matrix by linear mixing
                # Job.Electron.linear_mixing()
                Job.Electron.GR_Pulay(ii+1)
            else:
                SCFflag = True
                break

            if Job.Def['McWeeny'] == 1:
                Job.Electron.McWeeny()

            verboseprint(Job.Def['extraverbose'], "number of electrons = "+str(Job.Electron.electronspersite().sum()))
            verboseprint(Job.Def['extraverbose'], "output rho idempotency error is: ", Job.Electron.idempotency_error(Job.Electron.rho))
            verboseprint(Job.Def['extraverbose'], "input rho idempotency error is: ", Job.Electron.idempotency_error(Job.Electron.rhotot))
            verboseprint(Job.Def['extraverbose'], "SCF charges = ", Job.Hamilton.q)
            verboseprint(Job.Def['extraverbose'], "Magnetic moments = ", Job.Electron.spinpersite().T)

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

            # if the McWeeny flag is on then purify the density matrix
            if Job.Def['McWeeny'] == 1:
                Job.Electron.McWeeny()

    # Print out number of SCF loops taken
    verboseprint(Job.Def['verbose'], "Number of SCF loops: ", ii+1)
    # if self-consistency is not ofbtained the throw an error and exit.
    if SCFflag is False:
        print("ERROR: Self-consistency not obtained within maximum number of cycles: {}".format(max_loops))

    return SCFflag
