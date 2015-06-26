# -*- coding: utf-8 -*- 
"""
Created on Thursday, April 16, 2015

@author: Andrew Horsfield, Marc Coury and Max Boleininger

This module builds the tight binding Hamiltonians.
It merges the old TBH0 and TBHSO modules
"""
#
# Import the modules that will be needed
import numpy as np
from scipy.special import erf
import math
from Verbosity import *

# Spin orbit data
SOmatrix = {0: np.array([[complex( 0.0, 0.0), complex( 0.0, 0.0)],
                         [complex( 0.0, 0.0), complex( 0.0, 0.0)]]),
            1: np.array([[complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0), 
                          complex( 0.0, 0.0), complex(-1.0, 0.0), complex( 0.0, 1.0)],
                         [complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0,-1.0), 
                          complex( 1.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0)],
                         [complex( 0.0, 0.0), complex( 0.0, 1.0), complex( 0.0, 0.0), 
                          complex( 0.0,-1.0), complex( 0.0, 0.0), complex( 0.0, 0.0)],
                         [complex( 0.0, 0.0), complex( 1.0, 0.0), complex( 0.0, 1.0), 
                          complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0)],
                         [complex(-1.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0), 
                          complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 1.0)],
                         [complex( 0.0,-1.0), complex( 0.0, 0.0), complex( 0.0, 0.0), 
                          complex( 0.0, 0.0), complex( 0.0,-1.0), complex( 0.0, 0.0)]])}

class Hamiltonian:
    """Initialise and build the hamiltonian."""
    def __init__(self, JobClass):
        """Initialise the hamiltonian."""

        # Save job reference as an attribute for internal use.
        self.Job = JobClass

        # Build index of starting positions in Hamiltonian for each atom
        self.Hindex = np.zeros(self.Job.NAtom + 1, dtype='int')
        self.Hindex[0] = 0

        for a in range(self.Job.NAtom):
            self.Hindex[a+1] = self.Hindex[a] + self.Job.Model.atomic[self.Job.AtomType[a]]['NOrbitals']

        # Compute the sizes of the Hamiltonian matrices
        self.H0size = self.Hindex[self.Job.NAtom]
        self.HSOsize = 2*self.H0size
        
        # Allocate memory for the Hamiltonian matrices
        self.H0   = np.zeros((self.H0size,  self.H0size),  dtype='double')
        self.HSO  = np.zeros((self.HSOsize, self.HSOsize), dtype='complex')
        self.Fock = np.zeros((self.HSOsize, self.HSOsize), dtype='complex')
        
        # Allocate memeory for the charge and spin
        self.q = np.zeros(self.Job.NAtom, dtype='double')
        self.s = np.zeros((3, self.Job.NAtom), dtype='double')

    def electrostatics(self):
        def SCFGamma(self, atom1, atom2):

            atype1 = self.Job.AtomType[atom1]
            atype2 = self.Job.AtomType[atom2]

            if atom1 == atom2:
                gij = complex(self.Job.Model.atomic[atype1]['U'], 0.0)
            else:
                if self.Job.Def["InterSiteElectrostatics"] == 1:
                    
                    dr = self.Job.Pos[atom2] - self.Job.Pos[atom1]
                    rij = np.sqrt(dr.dot(dr))

                    vacuum = 0.00552639
                    ai = vacuum*vacuum*np.pi*np.pi*np.pi*8.0 * self.Job.Model.atomic[atype1]['U'] * self.Job.Model.atomic[atype1]['U'] 
                    aj = vacuum*vacuum*np.pi*np.pi*np.pi*8.0 * self.Job.Model.atomic[atype2]['U'] * self.Job.Model.atomic[atype2]['U']
                    b =  np.sqrt(ai * aj / (ai + aj))
                    
                    gij = 1.0/vacuum/(4.0*np.pi) * complex(erf(b * rij) / rij, 0.0)
                else:
                    gij = complex(0.0, 0.0)                  

            return gij

        SCF1g = np.zeros((self.Job.NAtom, self.Job.NAtom), dtype="complex")
        Wi = np.zeros(self.Job.NAtom, dtype="complex")

        # Compute SCF1g matrix
        for i in range(self.Job.NAtom):
            for j in range(i, self.Job.NAtom):
                gij = SCFGamma(self, i, j)
                SCF1g[i, j] = gij
                SCF1g[j, i] = gij

        # Compute electrostatic term
        for i in range(self.Job.NAtom):
            for j in range(self.Job.NAtom):
                Wi[i] += -self.q[j] * SCF1g[i, j]

        self.Wi = Wi

    def buildfock(self):
        """Build the Fock matrix by adding charge and spin terms to the Hamiltonian."""
        
        # Copy the Hamiltonian, complete with spin-orbit terms, to the Fock matrix
        self.fock = np.copy(self.HSO)
        h0s = self.H0size

        if self.Job.Def["Hamiltonian"] == "standard":

            # Add in diagonal corrections for charge and spin
            des = np.zeros((2, 2), dtype="complex")
            
            # Compute on-site and (if enabled) intersite electrostatic terms.
            self.electrostatics()

            for a in range(self.Job.NAtom):
                #
                # Get the atom type
                atype = self.Job.AtomType[a]
                #
                # Stoner onsite energy shifts are present for all four spins combinations
                des[0, 0] = -0.5 * self.Job.Model.atomic[atype]['I'] * complex( self.s[2, a],           0.0)
                des[0, 1] = -0.5 * self.Job.Model.atomic[atype]['I'] * complex( self.s[0, a], -self.s[1, a])
                des[1, 0] = -0.5 * self.Job.Model.atomic[atype]['I'] * complex( self.s[0, a],  self.s[1, a])
                des[1, 1] = -0.5 * self.Job.Model.atomic[atype]['I'] * complex(-self.s[2, a],           0.0)
                #
                # Step through each orbital on the atom
                for j in range(self.Hindex[a], self.Hindex[a+1]):
                    self.fock[    j,     j] += self.Wi[a] + des[0, 0]  # up/up block
                    self.fock[    j, h0s+j] +=              des[0, 1]  # up/down block
                    self.fock[h0s+j,     j] +=              des[1, 0]  # down/up block
                    self.fock[h0s+j, h0s+j] += self.Wi[a] + des[1, 1]  # down/down block

        elif self.Job.Def['Hamiltonian'] == "vectorS":
            norb = self.Job.NOrb
            natom = self.Job.NAtom
            rho = self.Job.Electron.rhotot
            J_ph = 0.0

            for a in range(natom):
                # Get the atom type
                atype = self.Job.AtomType[a]
                for j in range(self.Hindex[a], self.Hindex[a+1]):
                    J_S = self.Job.Model.atomic[atype]['I']
                    U = self.Job.Model.atomic[atype]['U']

                    # up/up block
                    self.fock[    j,     j] += self.add_H_pcase(    j,     j, U, J_S, J_ph, natom, norb, rho)
                    # up/down block
                    self.fock[    j, h0s+j] += self.add_H_pcase(    j, h0s+j, U, J_S, J_ph, natom, norb, rho)
                    # down/up block
                    self.fock[h0s+j,     j] += self.add_H_pcase(h0s+j,     j, U, J_S, J_ph, natom, norb, rho)
                    # down/down block
                    self.fock[h0s+j, h0s+j] += self.add_H_pcase(h0s+j, h0s+j, U, J_S, J_ph, natom, norb, rho)

        elif self.Job.Def['Hamiltonian'] == "pcase":
            norb = self.Job.NOrb
            natom = self.Job.NAtom
            rho = self.Job.Electron.rhotot

            for a in range(natom):
                # Get the atom type
                atype = self.Job.AtomType[a]
                for j in range(self.Hindex[a], self.Hindex[a+1]):
                    J = self.Job.Model.atomic[atype]['I']
                    U = self.Job.Model.atomic[atype]['U']

                    # up/up block
                    self.fock[    j,     j] += self.add_H_pcase(    j,     j, U, J, J, natom, norb, rho)
                    # up/down block
                    self.fock[    j, h0s+j] += self.add_H_pcase(    j, h0s+j, U, J, J, natom, norb, rho)
                    # down/up block
                    self.fock[h0s+j,     j] += self.add_H_pcase(h0s+j,     j, U, J, J, natom, norb, rho)
                    # down/down block
                    self.fock[h0s+j, h0s+j] += self.add_H_pcase(h0s+j, h0s+j, U, J, J, natom, norb, rho)

        # for i in range(h0s):
        #     for j in range(i, h0s):
        #         if abs(self.fock[i,j] - self.fock[i,j]) > 0.000001:
        #             verboseprint(self.Job.Def['extraverbose'], i, j,
        #                          round(self.fock[i,j].real, 4), round(self.fock2[i,j].real, 4), "|",
        #                          round(self.fock[i,j].imag, 4), round(self.fock2[i,j].imag, 4))
        #print "fock:", self.fock

    def slaterkoster(self, l1, l2, dr, v):
        """
        Evaluate the Slater-Koster table.

        The input is a pair of angular momentum quantum
        numbers (l1 and l2), the displacement between atoms 1 and 2 (dr, equal to r1 - r2),
        and a tuple that provides the (sigma, pi, delta , ...) hopping matrix elements.
        The function returns the full block of the Hamiltonian.

        The sign convention used here is as follows. The tables are computed with the atom
        corresponding to the second index being at the origin, and the atom with the first
        index placed vertically above it along the positive z axis.

        The cubic harmonic (K) orbital convention used is as follows. Let the magnetic quantum number
        be m. Then:
        K(l,2m-1) = sqrt(4 pi/(2l+1)) r^l ((-1)^m Y(l,m) + Y(l,-m))/sqrt(2)
        K(l,2m  ) = sqrt(4 pi/(2l+1)) r^l ((-1)^m Y(l,m) - Y(l,-m))/(i sqrt(2))
        Thus we have
        K(0,0) = 1

        K(1,0) = z
        K(1,1) = x
        K(1,2) = y

        K(2,0) = (3z^2 - r^2)/2
        K(2,1) = sqrt(3) zx
        K(2,2) = sqrt(3) zy
        K(2,3) = sqrt(3) (x^2-y^2)/2
        K(2,4) = sqrt(3) xy

        etc.
        """

        # Allocate space for the final block of the Hamiltonian
        block = np.zeros((2*l1+1, 2*l2+1), dtype='double')
        
        # Compute the direction cosines
        d = math.sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2])
        if d > 1.0e-15:
            l = dr[0]/d
            m = dr[1]/d
            n = dr[2]/d
        else:
            l = 0.0
            m = 0.0
            n = 1.0
        
        # Evaluate the matrix elements
        if (l1,l2) == (0,0):
            
            # s-s
            block[0,0] = v[0]
        elif (l1,l2) == (1,0):
            
            # p-s
            block[0,0] = n*v[0]
            block[1,0] = l*v[0]
            block[2,0] = m*v[0]
        elif (l1,l2) == (0,1):
            
            # s-p
            block[0,0] = n*v[0]
            block[0,1] = l*v[0]
            block[0,2] = m*v[0]
        elif (l1,l2) == (1,1):
            
            # p-p
            block[0,0] = n*n*(v[0]-v[1]) + v[1]
            block[0,1] = n*l*(v[0]-v[1])
            block[0,2] = n*m*(v[0]-v[1])
            
            block[1,0] = l*n*(v[0]-v[1])
            block[1,1] = l*l*(v[0]-v[1]) + v[1]
            block[1,2] = l*m*(v[0]-v[1])
            
            block[2,0] = m*n*(v[0]-v[1])
            block[2,1] = m*l*(v[0]-v[1])
            block[2,2] = m*m*(v[0]-v[1]) + v[1]
        
        # Return the Hamiltonian block
        return block


    def buildH0(self):
        """
        Build the hamiltonian one block at a time, with a block corresponding to
        a pair of atoms.

        This function invokes a function to compute the hopping
        integrals for the reference geometry for a tight binding model using the
        coefficients associated with the model. The block of the Hamiltonian
        matrix is built using the integrals from the model and Slater-Koster tables.
        """
        
        # Clear the memory for the Hamiltonian matrix
        self.H0.fill(0.0)
        
        # Step through all pairs of atoms
        for a1 in range(self.Job.NAtom):
            type1 = self.Job.AtomType[a1]  # The type of atom 1
            for a2 in range(self.Job.NAtom):
                type2 = self.Job.AtomType[a2]  # The type of atom 2
                
                # If the atoms are the same, compute an onsite block, otherwise compute a hopping block
                if a1 == a2:
                    self.H0[self.Hindex[a1]:self.Hindex[a1+1],
                            self.Hindex[a2]:self.Hindex[a2+1]] = self.Job.Model.atomic[type1]['e']
                else:
                    # Compute the atomic displacement
                    dr = self.Job.Pos[a1] - self.Job.Pos[a2]
                    distance = np.sqrt(dr.dot(dr))
                    
                    # Build the set of hopping integrals
                    v, v_bgn, v_end = self.Job.Model.helements(distance, type1, type2)
                	
                    # Build the block one pair of shells at a time
                    k1 = self.Hindex[a1] # Counter for orbital

                    # Step through shells of atom 1
                    for i1, l1 in enumerate(self.Job.Model.atomic[type1]['l']):  
                        n1 = 2*l1+1  # Compute number of orbitals in the shell                       
                        k2 = self.Hindex[a2]  # Counter for orbital	

						# Step through shells of atom 2
                        for i2, l2 in enumerate(self.Job.Model.atomic[type2]['l']):  
                            n2 = 2*l2+1  # Compute number of orbitals in the shell
                            self.H0[k1:k1+n1, k2:k2+n2] = self.slaterkoster(l1, l2, dr, v[v_bgn[i1,i2]:v_end[i1,i2]])
                            k2 += n2  # Advance to the start of the next set of orbitals

                        k1 += n1  # Advance to the start of the next set of orbitals

    def buildHSO(self):
        """Build the Hamiltonian with spin orbit coupling."""
        global SOmatrix

        h0s = self.H0size

        # Clear memory for the Hamiltonian spinorbit marix
        self.HSO.fill(0.0)

        # Build the basic Hamiltonian. This is independent of spin and appears in the
        # up-up and down-down blocks of the full spin dependent Hamiltonian
        self.buildH0()

        # Copy H0 into the two diagonal blocks
        self.HSO[0           :h0s,            0:h0s] = np.copy(self.H0)
        self.HSO[h0s:self.HSOsize, h0s:self.HSOsize] = np.copy(self.H0)
       
        # Add in magnetic field contribution
        eB = self.Job.Def['so_eB']
        for i in range(h0s):
            self.HSO[      i,       i] += complex( eB[2],    0.0)
            self.HSO[      i, h0s + i] += complex( eB[0], -eB[1])
            self.HSO[h0s + i,       i] += complex( eB[0],  eB[1])
            self.HSO[h0s + i, h0s + i] += complex(-eB[2],    0.0)
        
        # Add in the spin-orbit corrections
        for atom in range(self.Job.NAtom):
            atype = self.Job.AtomType[atom]

            k = self.Hindex[atom]  # Counter for orbital
            for i, l in enumerate(self.Job.Model.atomic[atype]['l']):  # Step through each shell
                n = 2*l+1  # Compute number of orbitals in the shell
                self.HSO[    k:    k+n,     k:    k+n] += self.Job.Model.atomic[atype]['so'][i]*SOmatrix[l][  0:  n,   0:  n]
                self.HSO[    k:    k+n, h0s+k:h0s+k+n] += self.Job.Model.atomic[atype]['so'][i]*SOmatrix[l][  0:  n, n+0:n+n]
                self.HSO[h0s+k:h0s+k+n,     k:    k+n] += self.Job.Model.atomic[atype]['so'][i]*SOmatrix[l][n+0:n+n,   0:  n]
                self.HSO[h0s+k:h0s+k+n, h0s+k:h0s+k+n] += self.Job.Model.atomic[atype]['so'][i]*SOmatrix[l][n+0:n+n, n+0:n+n]
                k += n  # Advance to the start of the next set of orbitals

    def add_H_pcase(self, ii, jj, U, J_S, J_ph, num_atoms, num_orbitals, rho):
        """
        Add the noncollinear Hamiltonian to the on-site contributions for
        the p-case Hubbard-like Hamiltonian.

        The tensorial form for this is:

        F^{s,s'}_{i a j b} = Kd(i,j) (\sum_{s''}Kd(s,s')(\sum_{a'}U Kd(a,b)
            rho^{s'',s''}_{i a i a} + J_S rho^{s'',s''}_{i a i b}
            + J_{ph} rho^{s'',s''}_{i b i a} )
            - (U rho^{s,s'}_{i a i b} + \sum_{a'} J_S Kd(a,b)rho^{s,s'}_{i a i a}
            + J_{ph} rho^{s,s'}_{i b i a}) ),

        where Kd is the Kronecker delta symbol; s, s' and s'' are spin indices; i
        and j are atom indices; a, a' and b are orbital indices; rho is the
        density matrix; U is the Hartree Coulomb integral and J_S and J_{ph} are
        the exchange Coulomb integrals. Physically, J_S is equivalent to J_{ph};
        however for the Stoner Hamiltonian J_S is the same as the Stoner I and
        J_{ph} is equal to zero.

        INPUT                 DATA TYPE       DESCRIPTION

        ii                    int             Fock matrix index 1.

        jj                    int             Fock matrix index 2.

        U                     float           The Hartree integral.

        J_S                   float           The exchange integral that goes in
                                              front of magnetism.

        J_ph                  float           The pair hopping integral that goes
                                              in front of the pair hopping term.

        num_atoms             int             The number of atoms.

        num_orbitals          list of int     A list of the number of orbitals for
                                              each atom.

        rho                   numpy matrix    The density matrix.


        OUTPUT                DATA TYPE       DESCRIPTION

        F                     float           The Fock matrix element for p-case
                                              symmetry on-site contributions for
                                              provided density matrix.

        """
        # atom, spatial orbital and spin for index 1
        i, a, s  = map_index_to_atomic(ii, num_atoms, num_orbitals)
        # atom, spatial orbital and spin for index 2
        j, b, sp = map_index_to_atomic(jj, num_atoms, num_orbitals)
        F = 0.0
        if i == j:
            # The negative terms
            F -= U*rho[ii, jj]
            F -= J_ph*rho[map_atomic_to_index(i, b, s, num_atoms, num_orbitals), map_atomic_to_index(i, a, sp, num_atoms, num_orbitals)]
            if a == b:
                F -= J_S*sum(rho[map_atomic_to_index(i, orb, s, num_atoms, num_orbitals), map_atomic_to_index(i, orb, sp, num_atoms, num_orbitals)] for orb in range(num_orbitals[i]))
            # The positive terms
            if s == sp:
                F += J_S*sum(rho[map_atomic_to_index(i, a, sig, num_atoms, num_orbitals), map_atomic_to_index(i, b, sig, num_atoms, num_orbitals)] for sig in range(2))
                F += J_ph*sum(rho[map_atomic_to_index(i, b, sig, num_atoms, num_orbitals), map_atomic_to_index(i, a, sig, num_atoms, num_orbitals)] for sig in range(2))
                if a == b:
                    F += U*sum(rho[map_atomic_to_index(i, orb, sig, num_atoms, num_orbitals), map_atomic_to_index(i, orb, sig, num_atoms, num_orbitals)] for sig in range(2) for orb in range(num_orbitals[i]))
        # return the matrix element
        return F


    def add_H_dcase(self):
        """
        Add the noncollinear Hamiltonian to the on-site contributions for the
        d-case Hubbard-like Hamiltonian.

        The tensorial form for this is:

        F^{s,s'}_{i a j b} = Kd(i,j) (\sum_{s''}Kd(s,s')(\sum_{a'}U Kd(a,b) rho^{s'',s''}_{i a i a}
            + J'_S rho^{s'',s''}_{i a i b} + J'_{ph} rho^{s'',s''}_{i b i a}
            - 48 dJ \sum_{cd stuv} xi_{c st}xi_{a tu}xi_{b uv}xi_{d vs} rho^{})
            - (U rho^{s,s'}_{i a i b} + \sum_{a'} J_S Kd(a,b)rho^{s,s'}_{i a i a}
            + J_{ph} rho^{s,s'}_{i b i a}) ),

        where Kd is the Kronecker delta symbol; s, s' and s'' are spin indices; i
        and j are atom indices; a, a' and b are orbital indices; rho is the
        density matrix; U is the Hartree Coulomb integral and J_S and J_{ph} are
        the exchange Coulomb integrals. Physically, J_S is equivalent to J_{ph};
        however for the Stoner Hamiltonian J_S is the same as the Stoner I and
        J_{ph} is equal to zero.

        INPUT                 DATA TYPE       DESCRIPTION

        ii                    int             Fock matrix index 1.

        jj                    int             Fock matrix index 2.

        U                     float           The Hartree integral.

        J_S                   float           The average of the t_2g and e_g
                                              exchange integrals that goes in
                                              front of magnetism.

        J_ph                  float           Same as J_S except that it goes in
                                              front of the pair hopping term.

        dJ                    float           The difference in the t_2g and e_g
                                              d-orbital exchange integrals.

        num_atoms             int             The number of atoms.

        num_orbitals          list of int     A list of the number of orbitals for
                                              each atom.

        rho                   numpy matrix    The density matrix.


        OUTPUT                DATA TYPE       DESCRIPTION

        F                     float           The Fock matrix element for p-case
                                              symmetry on-site contributions for
                                              provided density matrix.

        """


def Kd(a, b):
    """
    Function Kd is the Kronecker delta symbol. If a == b then return 1 otherwise
    return zero.

    INPUT                 DATA TYPE       DESCRIPTION

    a                     int             Value 1

    b                     int             Value 2


    OUTPUT                DATA TYPE       DESCRIPTION

    Kd                    int             delta_{a,b}

    """
    if a == b:
        return 1
    else:
        return 0


def map_index_to_atomic(index, num_atoms, num_orbitals):
    """
    Function map_index_to_atomic converts the index of the Fock matrix into atom
    number, orbital number and spin. Index numbering starts at 0 and goes up to
    the length of the Fock matrix. Atomic numbering starts at 0 and goes up to
    num_atoms-1, orbital numbering goes starts at 0 and goes up to num_orbitals-1
    for that particular atom.

    ASSUMPTIONS
    The function treats orbital number and spin separately. It will work for atoms
    with different numbers of spatial orbitals -- if only atoms of the same type
    are going to be used the code could be made to be more efficient.

    This could probably also be improved by providing the length of the side of the
    Fock matrix.

    INPUT                 DATA TYPE       DESCRIPTION

    index                 int             The index to be converted to atom,
                                          orbital and spin.

    num_atoms             int             The number of atoms.

    num_orbitals          list of int     The number of orbitals for each atom.


    OUTPUT                DATA TYPE       DESCRIPTION

    atom                  int             The atom number.

    orbital               int             The spatial orbital number.

    spin                  int             The spin, either 0 (up) or 1 (down).

    """
    upper_bound = 0
    # loop over spin
    for ss in range(2):
        # loop over the atoms
        for ii in range(num_atoms):
            upper_bound += num_orbitals[ii]
            if index < upper_bound:
                atom = ii
                spin = ss
                # orbital is the index minus the previous upper bound
                orbital = index-(upper_bound-num_orbitals[ii])
                return atom, orbital, spin


def map_atomic_to_index(atom, orbital, spin, num_atoms, num_orbitals):
    """
    Function map_atomic_to_index converts the atom, orbital number and spin into
    the corresponding index of the Fock matrix. Index numbering starts at 0 and
    goes up to the length of the Fock matrix. Atomic numbering starts at 0 and
    goes up to num_atoms-1, orbital numbering goes starts at 0 and goes up to
    num_orbitals-1 for that particular atom.

    ASSUMPTIONS
    The function treats orbital number and spin separately. It will work for atoms
    with different numbers of spatial orbitals -- if only atoms of the same type
    are going to be used the code could be made to be more efficient.

    This could probably also be improved by providing the length of the side of the
    Fock matrix.

    INPUT                 DATA TYPE       DESCRIPTION

    atom                  int             The atom number.

    orbital               int             The spatial orbital number.

    spin                  int             The spin, either 0 (up) or 1 (down).

    num_atoms             int             The number of atoms.

    num_orbitals          list of int     The number of orbitals for each atom.


    OUTPUT                DATA TYPE       DESCRIPTION

    index                 int             The index to be converted to atom,
                                          orbital and spin.

    """
    index = 0
    # add atom contribution
    for ii in range(atom):
        index += num_orbitals[ii]
    # add spin contribution
    if spin == 0:
        pass
    elif spin == 1:
        index = 2*index
        for ii in range(atom, num_atoms):
            index += num_orbitals[ii]

    # add orbital contribution
    index += orbital

    return index