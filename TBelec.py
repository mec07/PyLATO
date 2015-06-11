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
import TBgeom


def init(JobDef):
    """Initialise the global variables. Compute the total number of
    electrons in the system, and allocates space for the occupancies"""
    global NElectrons
    global occ, rho, zcore
    #
    # Set up the core charges, and count the number of electrons
    zcore = np.zeros(TBgeom.NAtom, dtype='double')
    for a in range(0, TBgeom.NAtom):
        zcore[a] = TBH.AtomData[TBgeom.AtomType[a]]['NElectrons']
    NElectrons = np.sum(zcore)
    #
    # Allocate memory for the level occupancies and density matrix
    occ = np.zeros(TBH.HSOsize, dtype='double')
    rho = np.zeros((TBH.HSOsize, TBH.HSOsize), dtype='complex')


def Fermi(e, mu, kT):
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


def Occupy(e, kT, n_tol):
    """Populate the eigenstates using the Fermi function.
    This function uses binary section."""
    global NElectrons, occ, mu
    #
    # Find the lower bound to the chemical potential
    mu_l = e[0]
    while np.sum(Fermi(e, mu_l, kT)) > NElectrons:
        mu_l -= 10.0*kT
    #
    # Find the upper bound to the chemical potential
    mu_u = e[-1]
    while np.sum(Fermi(e, mu_u, kT)) < NElectrons:
        mu_u += 10.0*kT
    #
    # Find the correct chemical potential using binary section
    mu = 0.5*(mu_l + mu_u)
    n = np.sum(Fermi(e, mu, kT))
    while math.fabs(NElectrons-n) > n_tol*NElectrons:
        if n > NElectrons:
            mu_u = mu
        elif n < NElectrons:
            mu_l = mu
        mu = 0.5*(mu_l + mu_u)
        n = np.sum(Fermi(e, mu, kT))
    occ = Fermi(e, mu, kT)


def DensityMatrix(psi):
    """Buld the density matrix."""
    global occ, rho
    rho = np.matrix(psi)*np.diag(occ)*np.matrix(psi).H


def ChargePerSite():
    """Compute the net charge on each site."""
    global zcore, rho
    norb = np.diag(rho)
    qsite = np.zeros(TBgeom.NAtom, dtype='double')
    for a in range(0,TBgeom.NAtom):
        qsite[a] = zcore[a] - (np.sum(norb[TBH.Hindex[a]:TBH.Hindex[a+1]].real) + np.sum(norb[TBH.H0size+TBH.Hindex[a]:TBH.H0size+TBH.Hindex[a+1]].real))
    return qsite


def SpinPerSite():
    """Compute the net spin on each site."""
    global rho
    ssite = np.zeros((3,TBgeom.NAtom), dtype='double')
    for a in range(0,TBgeom.NAtom):
        srho = np.zeros((2,2), dtype='complex')
        for j in range(TBH.Hindex[a], TBH.Hindex[a+1]):
            #
            # Sum over orbitals on one site to produce a 2x2 spin density matrix for the site
            srho[0, 0] += rho[           j,            j]
            srho[0, 1] += rho[           j, TBH.H0size+j]
            srho[1, 0] += rho[TBH.H0size+j,            j]
            srho[1, 1] += rho[TBH.H0size+j, TBH.H0size+j]
        #
        # Now compute the net spin vector for the site
        ssite[0, a] = (srho[0, 1] + srho[1, 0]).real
        ssite[1, a] = (srho[0, 1] - srho[1, 0]).imag
        ssite[2, a] = (srho[0, 0] - srho[1, 1]).real
    #
    return ssite
