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
import TBH
import sys
import time

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
        self.rho = np.zeros((self.Job.Hamilton.HSOsize, self.Job.Hamilton.HSOsize), dtype='complex')
        # for the pcase, dcase and vector Stoner Hamiltonians we need two density matrices
        if self.Job.Def['Hamiltonian'] in ('pcase','dcase','vectorS'):
            self.rhotot = np.zeros((self.Job.Hamilton.HSOsize, self.Job.Hamilton.HSOsize), dtype='complex')


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

    def SCFerror(self):
        """
        Calculate the self-consistent field error. We do this by comparing the
        on-site elements of the new density matrix (self.rho) with the old 
        density matrix, self.rhotot. It is normalised by dividing by the total
        number of electrons.

        """
        return sum(abs(
                self.rho[TBH.map_atomic_to_index(atom1, orbital1, spin1, self.Job.NAtom, self.Job.NOrb),TBH.map_atomic_to_index(atom1, orbital2, spin2, self.Job.NAtom, self.Job.NOrb)]
           - self.rhotot[TBH.map_atomic_to_index(atom1, orbital1, spin1, self.Job.NAtom, self.Job.NOrb),TBH.map_atomic_to_index(atom1, orbital2, spin2, self.Job.NAtom, self.Job.NOrb)])
                for atom1 in range(self.Job.NAtom) for orbital1 in range(self.Job.NOrb[atom1]) for spin1 in range(2)
                for orbital2 in range(orbital1,self.Job.NOrb[atom1]) for spin2 in range(spin1,2)
                )/(self.Job.Electron.NElectrons**2)

    def linear_mixing(self):
        """
        Mix the new and the old density matrix by linear mixing.
        The form of this mixing is 
            rho_out = (1-alpha) rho_old + alpha rho_new
        for which, using our notation, rho_new is self.rho, rho_old is
        self.rhotot and we overwrite self.rhotot to make rho_out.
        """
        alpha = self.Job.Def['alpha']
        self.rhotot = (1-alpha)*self.rhotot + alpha*self.rho


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
        index = TBH.map_atomic_to_index(site, orbital, spin, self.Job.NAtom, self.Job.NOrb)
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
                        index_az = TBH.map_atomic_to_index(site1,a,z,self.Job.NAtom, self.Job.NOrb)
                        index_bz = TBH.map_atomic_to_index(site1,b,z,self.Job.NAtom, self.Job.NOrb)
                        index_bs = TBH.map_atomic_to_index(site1,b,s,self.Job.NAtom, self.Job.NOrb)
                        index_as = TBH.map_atomic_to_index(site1,a,s,self.Job.NAtom, self.Job.NOrb)
                        # term 1: 2.0*rho_{aa}^{zs} rho_{bb}^{sz}
                        C_avg += 2.0*self.rho[index_az,index_as]*self.rho[index_bs,index_bz]
                        # term 2: -2.0*rho_{ab}^{zz}rho_{ba}^{ss})
                        C_avg -= 2.0*self.rho[index_az,index_bz]*self.rho[index_as,index_bs]
                        # term 3: -rho_{aa}^{ss}rho_{bb}^{zz}
                        C_avg -= self.rho[index_as,index_as]*self.rho[index_bz,index_bz]
                        # term 4: rho_{ab}^{sz}rho_{ba}^{zs}
                        C_avg += self.rho[index_as,index_bz]*self.rho[index_bz,index_as]


        # remember to divide by 3
        C_avg = C_avg/3.0

        return C_avg



