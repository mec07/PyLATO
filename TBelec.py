"""
Created on Thursday 16 April 2015

@author: Andrew Horsfield

This module contains functions that are needed once the molecular orbitals are
populated by electrons.
"""
#
# Import the modules that will be needed
import numpy as np
import math
import TBH


class Electronic:
    """Initialise and build the density matrix."""
    def __init__(self, JobClass):
        """Compute the total number of electrons in the system, allocate space for occupancies"""

        # Save job reference as an attribute for internal use.
        self.Job = JobClass

        # Set up the core charges, and count the number of electrons
        self.zcore = np.zeros(self.Job.NAtom, dtype='double')

        for a in range(0, self.Job.NAtom):
            self.zcore[a] = TBH.AtomData[self.Job.AtomType[a]]['NElectrons']
        self.NElectrons = np.sum(self.zcore)

        #
        # Allocate memory for the level occupancies and density matrix
        self.occ = np.zeros( self.Job.Hamilton.HSOsize, dtype='double')
        self.rho = np.zeros((self.Job.Hamilton.HSOsize, self.Job.Hamilton.HSOsize), dtype='complex')


    def fermi(self, e, mu, kT):
        """Evaluate the Fermi function."""
        x = (e-mu)/kT
        f = np.zeros(x.size, dtype='double')
        for i in range(0, x.size):
            if x[i] > 35.0:
                f[i] = 0.0
            elif x[i] < -35.0:
                f[i] = 1.0
            else:
                f[i] = 1.0/(1.0 + np.exp(x[i]))
        return f


    def occupy(self, s, kT, n_tol):
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
        while math.fabs(self.NElectrons-n) > n_tol*self.NElectrons:
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
