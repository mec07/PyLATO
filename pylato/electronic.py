"""
Created on Thursday 16 April 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

This module contains functions that are needed once the molecular orbitals are
populated by electrons.
"""
#
# Import the modules that will be needed
import numpy as np
import math
import os
import sys
import time
import random

from pylato.fermi import fermi_0, fermi_non0
from pylato.hamiltonian import map_atomic_to_index
from pylato.verbosity import verboseprint

dir_path = os.path.dirname(os.path.realpath(__file__))


class Electronic:
    """Initialise and build the density matrix."""
    def __init__(self, JobClass):
        """Compute the total number of electrons in the system, allocate space for occupancies"""

        # Save job reference as an attribute for internal use.
        self.Job = JobClass

        # Set up the core charges, and count the number of electrons
        self.zcore = np.zeros(self.Job.NAtom, dtype='double')

        for a in range(0, self.Job.NAtom):
            self.zcore[a] = self.Job.Model.atomic[self.Job.AtomType[a]]['NElectrons']
        self.NElectrons = np.sum(self.zcore)

        #
        # Allocate memory for the level occupancies and density matrix
        self.occ = np.zeros( self.Job.Hamilton.HSOsize, dtype='double')
        self.rho = np.matrix(np.zeros((self.Job.Hamilton.HSOsize, self.Job.Hamilton.HSOsize), dtype='complex'))
        self.rhotot = np.matrix(np.zeros((self.Job.Hamilton.HSOsize, self.Job.Hamilton.HSOsize), dtype='complex'))
        # setup for the Pulay mixing
        self.inputrho = np.zeros((self.Job.Def['num_rho'], self.Job.Hamilton.HSOsize, self.Job.Hamilton.HSOsize), dtype='complex')
        self.outputrho = np.zeros((self.Job.Def['num_rho'], self.Job.Hamilton.HSOsize, self.Job.Hamilton.HSOsize), dtype='complex')
        self.residue = np.zeros((self.Job.Def['num_rho'], self.Job.Hamilton.HSOsize, self.Job.Hamilton.HSOsize), dtype='complex')

        if self.Job.Def['el_kT'] == 0.0:
            self.fermi = fermi_0
        else:
            self.fermi = fermi_non0

        if self.Job.Def.get('optimisation_routine') == 1:
            self.optimisation_routine = self.optimisation_routine1
        elif self.Job.Def.get('optimisation_routine') == 2:
            self.optimisation_routine = self.optimisation_routine2
        else:
            print("WARNING: No optimisation routine selected. Using optimisation_routine1.")
            self.optimisation_routine = self.optimisation_routine1

    def occupy(self, s, kT, n_tol, max_loops):
        """Populate the eigenstates using the Fermi function.
        This function uses binary section."""
        #
        # Find the lower bound to the chemical potential
        mu_l = self.Job.e[0]
        while np.sum(self.fermi(self.Job.e, mu_l, kT)) > self.NElectrons:
            mu_l -= 10.0*kT
        #
        # Find the upper bound to the chemical potential
        mu_u = self.Job.e[-1]
        while np.sum(self.fermi(self.Job.e, mu_u, kT)) < self.NElectrons:
            mu_u += 10.0*kT
        #
        # Find the correct chemical potential using binary section
        mu = 0.5*(mu_l + mu_u)
        n = np.sum(self.fermi(self.Job.e, mu, kT))
        count = 0
        while math.fabs(self.NElectrons-n) > n_tol*self.NElectrons:
            count+=1
            if count>max_loops:
                print("ERROR: The chemical potential could not be found. The error became "+str(math.fabs(self.NElectrons-n)))
                sys.exit()
            if n > self.NElectrons:
                mu_u = mu
            elif n < self.NElectrons:
                mu_l = mu
            mu = 0.5*(mu_l + mu_u)
            n = np.sum(self.fermi(self.Job.e, mu, kT))
        self.occ = self.fermi(self.Job.e, mu, kT)

    def densitymatrix(self):
        """Build the density matrix."""
        self.rho = np.matrix(self.Job.psi)*np.diag(self.occ)*np.matrix(self.Job.psi).H

    def constructDensityMatrixFromOccupation(self, occupation):
        """Build the density matrix using stored eigenvectors and a provided occupation vector."""
        self.rhotot = np.matrix(self.Job.psi)*np.diag(occupation)*np.matrix(self.Job.psi).H

    def constructDensityMatrixFromEigenvaluesAndEigenvectors(self, eigenvalues, eigenvectors):
        """Build the density matrix using provided eigenvectors and eigenvalues."""
        self.occupy(eigenvalues, self.Job.Def['el_kT'], self.Job.Def['mu_tol'], self.Job.Def['mu_max_loops'])
        self.rho = np.matrix(eigenvectors)*np.diag(self.occ)*np.matrix(eigenvectors).H

    def SCFerror(self):
        """
        Calculate the self-consistent field error. We do this by comparing the
        on-site elements of the new density matrix (self.rho) with the old 
        density matrix, self.rhotot. It is normalised by dividing by the total
        number of electrons.

        """
        return sum(abs(
                self.rho[map_atomic_to_index(atom1, orbital1, spin1, self.Job.NAtom, self.Job.NOrb), map_atomic_to_index(atom1, orbital2, spin2, self.Job.NAtom, self.Job.NOrb)]
           - self.rhotot[map_atomic_to_index(atom1, orbital1, spin1, self.Job.NAtom, self.Job.NOrb), map_atomic_to_index(atom1, orbital2, spin2, self.Job.NAtom, self.Job.NOrb)])
                for atom1 in range(self.Job.NAtom) for orbital1 in range(self.Job.NOrb[atom1]) for spin1 in range(2)
                for orbital2 in range(orbital1, self.Job.NOrb[atom1]) for spin2 in range(spin1, 2)
                )/(self.Job.Electron.NElectrons**2)

    def idempotency_error(self, rho):
        """
        Determine how far from idempotency the density matrix is. If the
        density matrix is idempotent then

        rho*rho - rho = 0.

        We normalise by the number of electrons.
        """
        rho_err = np.linalg.norm((np.dot(rho, rho) - rho))/self.NElectrons
        return rho_err

    def McWeeny(self):
        """
        Make the density matrix idempotent using the McWeeny transformation,
        R.McWeeny, Rev. Mod. Phys. (1960):

        rho_n+1 = 3*rho_n^3 - 2*rho_n^2

        """
        if self.Job.isNoncollinearHami:
            rho_temp = self.rhotot
        else:
            rho_temp = self.rho

        # Make sure that it isn't already idempotent
        err_orig = self.idempotency_error(rho_temp)
        if err_orig < self.Job.Def['McWeeny_tol']:
            # if already idempotent then don't do anything, just exit function
            return

        flag, iterations, err, rho_temp = self.McWeeny_iterations(rho_temp)
        # if the flag is false it means that idempotency was reduced below the tolerance
        if flag is False:
            # if the iterations did not converge but the idempotency error has
            # gotten smaller then print a warning but treat as a success.
            if err < err_orig:
                print("Max iterations, {} reached. Idempotency error = {}".format(iterations, err))
                flag = True
            else:
                print("McWeeny transformation unsuccessful. Proceeding using input density matrix.")
                # Turn off using the McWeeny transformation as once it doesn't work it seems to not work again.
                self.Job.Def["McWeeny"] = 0

        # if this is going to be treated like a success then reassign rho_temp.
        if flag is True:
            if self.Job.isNonCollinearHami:
                self.rhotot = rho_temp
            else:
                self.rho = rho_temp

    def McWeeny_iterations(self, rho):
        """
        Iterations of the McWeeny scheme for the inputted rho.
        Return a True/False flag that indicates convergence, the number of
        iterations required to reach convergence, the error and the converged density
        matrix.
        """
        converge_flag = False
        for ii in range(self.Job.Def['McWeeny_max_loops']):
            # McWeeny transformation
            rho = 3*np.dot(rho, np.dot(rho, rho)) - 2*np.dot(rho, rho)
            err = self.idempotency_error(rho)
            verboseprint(self.Job.Def['extraverbose'], "McWeeny iteration: ", ii, "; Idempotency error = ", err)
            if err < self.Job.Def['McWeeny_tol']:
                converge_flag = True
                return converge_flag, ii, err, rho
            # Check to make sure that the error hasn't become a nan.
            elif np.isnan(err):
                return converge_flag, ii, err, rho

        # if it gets to this statement then it probably hasn't converged.
        return converge_flag, ii, err, rho

    def linear_mixing(self):
        """
        Mix the new and the old density matrix by linear mixing.
        The form of this mixing is 
            rho_out = (1-A)*rho_old + A*rho_new
        for which, using our notation, rho_new is self.rho, rho_old is
        self.rhotot and we overwrite self.rhotot to make rho_out.
        """
        A = self.Job.Def['A']
        self.rhotot = (1-A)*self.rhotot + A*self.rho

    def GR_Pulay(self, scf_iteration):
        """
        This is the guaranteed reduction Pulay mixing scheme proposed by
        Bowler and Gillan in 2008. If the number of density matrices to be
        used, num_rho, is 1, it reduces to just linear mixing. 

        The scf_iteration is a required input because when scf_iteration is
        less than num_rho then scf_iteration is the number of density matrices
        that should be used.

        The output is an updated self.rhotot to be used in the construction of
        the Fock matrix. Also, self.inputrho, self.outputrho and self.residue
        are updated for the next iteration.

        """
        num_rho = self.Job.Def['num_rho']
        # If the number of scf iterations is less than num_rho replace it by
        # the number of scf iterations (as there will only be that number of
        # density matrices).
        if scf_iteration < num_rho:
            num_rho = scf_iteration

        # Shift along the density and residue matrices
        for ii in range(num_rho-1):
            self.inputrho[num_rho - 1 - ii] = np.copy(self.inputrho[num_rho - 2 - ii])
            self.outputrho[num_rho - 1 - ii] = np.copy(self.outputrho[num_rho - 2 - ii])
            self.residue[num_rho - 1 - ii] = np.copy(self.residue[num_rho - 2 - ii])

        # Add in the new density and residue matrices 
        self.inputrho[0] = self.rhotot
        self.outputrho[0] = self.rho
        self.residue[0] = self.rho - self.rhotot

        # Calculate the values of alpha to minimise the residue
        alpha, igo = self.optimisation_routine(num_rho)
        if igo == 1:
            print("WARNING: Unable to optimise alpha for combining density matrices. Proceeding using guess.")
            # Guess for alpha is just 1.0 divided by the number of density matrices
            alpha = np.zeros((num_rho), dtype='double')
            alpha.fill(1.0/num_rho)
        verboseprint(self.Job.Def['extraverbose'], "alpha: ", alpha)
        # Create an optimised rhotot and an optimised rho and do linear mixing to make next input matrix
        self.rhotot = sum(alpha[i]*self.inputrho[i] for i in range(num_rho))
        self.rho = sum(alpha[i]*self.outputrho[i] for i in range(num_rho))
        self.linear_mixing()


    def chargepersite(self):
        """Compute the net charge on each site."""
        norb = np.diag(self.rho)
        qsite = np.zeros(self.Job.NAtom, dtype='double')
        jH = self.Job.Hamilton

        for a in range(0, self.Job.NAtom):
            qsite[a] = (self.zcore[a] -
                        (np.sum(norb[jH.Hindex[a]:jH.Hindex[a+1]].real) +
                         np.sum(norb[jH.H0size+jH.Hindex[a]:jH.H0size+jH.Hindex[a+1]].real)))
        return qsite

    def electrons_site_orbital_spin(self,site,orbital,spin):
        """Compute the number of electrons with specified spin, orbital and site. """
        index = map_atomic_to_index(site, orbital, spin, self.Job.NAtom, self.Job.NOrb)
        return self.rho[index,index].real

    def electrons_orbital_occupation_vec(self):
        """ Return a vector of the occupation of each spin orbital. """
        occupation = []
        # Just collect the real part of the diagonal of the density matrix.
        for ii in range(self.Job.Hamilton.HSOsize):
            occupation.append(self.rho[ii,ii].real)

        return occupation


    def electrons_site_orbital(self,site,orbital):
        """Compute the number of electrons in a particular orbital on the specified site. """
        return self.electrons_site_orbital_spin(site,orbital,0)+self.electrons_site_orbital_spin(site,orbital,1).real


    def electrons_site(self,site):
        """Compute the number of electrons on a specified site. """
        return sum(self.electrons_site_orbital(site,ii) for ii in range(self.Job.Model.atomic[self.Job.AtomType[site]]['NOrbitals'])).real


    def electronspersite(self):
        """ Return a vector of the number of electrons on each site. """
        esite = np.zeros(self.Job.NAtom, dtype='double')
        for a in range(self.Job.NAtom):
            esite[a] = self.electrons_site(a).real

        return esite


    def spinpersite(self):
        """Compute the net spin on each site."""
        ssite = np.zeros((3, self.Job.NAtom), dtype='double')
        jH = self.Job.Hamilton

        for a in range(0, self.Job.NAtom):
            srho = np.zeros((2, 2), dtype='complex')
            for j in range(jH.Hindex[a], jH.Hindex[a+1]):
                #
                # Sum over orbitals on one site to produce a 2x2 spin density matrix for the site
                srho[0, 0] += self.rho[          j,           j]
                srho[0, 1] += self.rho[          j, jH.H0size+j]
                srho[1, 0] += self.rho[jH.H0size+j,           j]
                srho[1, 1] += self.rho[jH.H0size+j, jH.H0size+j]
            #
            # Now compute the net spin vector for the site
            ssite[0, a] = (srho[0, 1] + srho[1, 0]).real
            ssite[1, a] = (srho[0, 1] - srho[1, 0]).imag
            ssite[2, a] = (srho[0, 0] - srho[1, 1]).real
        #
        return ssite


    def magnetic_correlation(self, site1, site2):
        """
        Compute the direction averaged magnetic correlation between sites 1
        and site 2. This requires the two particle density matrix. As we are
        using the mean field approximation the two particle density matrix is
        expressible in terms of the one particle density matrix. The equation
        below is the equation for the magnetic correlation using the single
        particle density matrix.

        C_avg = 1/3 sum_{absz}( 2(rho_{aa}^{zs} rho_{bb}^{sz} - rho_{ab}^{zz}rho_{ba}^{ss})
                - rho_{aa}^{ss}rho_{bb}^{zz}+rho_{ab}^{sz}rho_{ba}^{zs})

        where a are the spatial orbitals on site 1, b are the spatial orbitals
        on site 2, s and z are spin indices.
        """

        C_avg = np.float64(0.0)
        norb_1 = self.Job.Model.atomic[self.Job.AtomType[site1]]['NOrbitals']
        norb_2 = self.Job.Model.atomic[self.Job.AtomType[site2]]['NOrbitals']

        for s in range(2):
            for z in range(2):
                for a in range(norb_1):
                    for b in range(norb_2):
                        index_az = map_atomic_to_index(site1,a,z,self.Job.NAtom, self.Job.NOrb)
                        index_bz = map_atomic_to_index(site2,b,z,self.Job.NAtom, self.Job.NOrb)
                        index_bs = map_atomic_to_index(site2,b,s,self.Job.NAtom, self.Job.NOrb)
                        index_as = map_atomic_to_index(site1,a,s,self.Job.NAtom, self.Job.NOrb)
                        # term 1: 2.0*rho_{1a1a}^{zs} rho_{2b2b}^{sz}
                        C_avg += 2.0*self.rho[index_az,index_as]*self.rho[index_bs,index_bz]
                        # term 2: -2.0*rho_{1a2b}^{zz}rho_{2b1a}^{ss})
                        C_avg -= 2.0*self.rho[index_az,index_bz]*self.rho[index_bs,index_as]
                        # term 3: -rho_{1a1a}^{ss}rho_{2b2b}^{zz}
                        C_avg -= self.rho[index_as,index_as]*self.rho[index_bz,index_bz]
                        # term 4: rho_{1a2b}^{sz}rho_{1b2a}^{zs}
                        C_avg += self.rho[index_as,index_bz]*self.rho[index_bz,index_as]


        # remember to divide by 3
        C_avg = C_avg/3.0

        return C_avg

    def optimisation_routine1(self, num_rho):
        """
        Optimisation routine where we try to solve for the norm squared of the
        optimal density matrix with the constraint that the sum of the
        coefficients is equal to one. To include the constraint we set up the
        problem:

        minimise: alpha_i M_ij alpha_j - lambda (sum_i alpha_i - 1)

        where M_ij = Tr(R_i^dag R_j). We then differentiate with respect to
        alpha_k and set to zero to minimise:

        2 M alpha = lambda

        We solve this equation for lambda = 1. We then can simply scale alpha,
        such that sum_i alpha_i = 1, which is equivalent to having solved for
        a different lambda.
        """
        verboseprint(self.Job.Def['extraverbose'], "optimisation_routine")
        small = 1e-14
        # If there is only one density matrix the solution is simple.
        if num_rho == 1:
            return np.array([1.0], dtype='double'), 0

        alpha = np.zeros(num_rho, dtype='double')
        Mmat = np.matrix(np.zeros((num_rho, num_rho), dtype='complex'))
        lamb = 0.5*np.ones(num_rho, dtype='double')
        for i in range(num_rho):
            Mmat[i, i] = np.trace(np.matrix(self.residue[i])*np.matrix(self.residue[i]).H)
            for j in range(i+1, num_rho):
                Mmat[i, j] = np.trace(np.matrix(self.residue[i])*np.matrix(self.residue[j]).H)
                # if np.sum(np.matrix(self.residue[j]).H*np.matrix(self.residue[i])) != Mmat[i, j].conj():
                #     print "Mmat[%i,%i] = %f. Mmat[%i,%i].conj() = %f." % (j, i, np.sum(np.matrix(self.residue[j]).H*np.matrix(self.residue[i])), i, j, Mmat[i, j].conj())
                Mmat[j, i] = Mmat[i, j].conj()
        # if np.linalg.det(Mmat) < small:
        #     return alpha, 1

        alpha = np.linalg.solve(Mmat, lamb)
        myscale = np.sum(alpha)
        if myscale == 0:
            print("ERROR: alpha summed to 0 in optimisation_routine. Cannot be scaled to 1.")
            print(alpha)
            return alpha, 1
        else:
            alpha = alpha/myscale
        return alpha, 0

    def optimisation_routine2(self, num_rho):
        """
        Optimisation routine where we try to solve for the norm squared of the
        optimal density matrix with the constraint that the sum of the
        coefficients is equal to one. To include the constraint we set up the
        problem:

        minimise: alpha_i M_ij alpha_j - lambda (sum_i alpha_i - 1)

        where M_ij = Tr(R_i^dag R_j). We then differentiate with respect to
        alpha_k and set to zero to minimise:

        2 M alpha - lambda = 0

        We solve this equation. We have to add a buffer row and column to
        include lambda as well as the constraint that the sum of alpha is
        equal to one. We absorb the 2 into lambda:

        {M_11   M_12    ...    -1   {alpha_1     {0
         M_21   M_22    ...    -1    alpha_2      0
         .              .            .            .
         .                .          .        =   .
         .                  .        .            .
         -1     -1      ...     0}   lambda}      -1}

        """
        small = 1e-10
        verboseprint(self.Job.Def['extraverbose'], "optimisation_routine2")
        # If there is only one density matrix the solution is simple.
        if num_rho == 1:
            return np.array([1.0], dtype='double'), 0

        alpha = np.zeros(num_rho+1, dtype='double')
        Mmat = np.matrix(np.zeros((num_rho+1, num_rho+1), dtype='complex'))
        # make all the elements -1
        Mmat.fill(-1.0)
        # replace the bottom right hand corner by 0
        Mmat[-1,-1] = 0.0
        # calculate the rest of the Mmat.
        for i in range(num_rho):
            Mmat[i, i] = np.trace(np.matrix(self.residue[i])*np.matrix(self.residue[i]).H)
            for j in range(i+1, num_rho):
                Mmat[i, j] = np.trace(np.matrix(self.residue[i])*np.matrix(self.residue[j]).H)
                # if np.sum(np.matrix(self.residue[j]).H*np.matrix(self.residue[i])) != Mmat[i, j].conj():
                #     print "Mmat[%i,%i] = %f. Mmat[%i,%i].conj() = %f." % (j, i, np.sum(np.matrix(self.residue[j]).H*np.matrix(self.residue[i])), i, j, Mmat[i, j].conj())
                Mmat[j, i] = Mmat[i, j].conj()
        # if abs(np.linalg.det(Mmat)) < small:
        #     return alpha, 1

        RHS = np.zeros(num_rho+1, dtype = 'double')
        RHS[-1] = -1.0
        alpha = np.linalg.solve(Mmat, RHS)
        myscale = abs(np.sum(alpha)-alpha[-1])
        if abs(myscale-1.0) > small:
            print("ERROR: optimisation_routine2 -- sum alpha = %f. alpha must sum to 1.0." % myscale)
            print(alpha)
            return alpha, 1
        # if successful then return result and no error code.
        return alpha, 0
