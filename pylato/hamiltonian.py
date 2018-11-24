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
import math
from scipy.special import erf

from pylato.exceptions import UnimplementedModelError


class Hamiltonian:
    """Initialise and build the hamiltonian."""
    def __init__(self, Job):
        """Initialise the hamiltonian."""

        # Build index of starting positions in Hamiltonian for each atom
        self.Hindex = np.zeros(Job.NAtom + 1, dtype='int')
        self.Hindex[0] = 0

        for a in range(Job.NAtom):
            self.Hindex[a+1] = self.Hindex[a] + Job.Model.atomic[Job.AtomType[a]]['NOrbitals']

        # Compute the sizes of the Hamiltonian matrices
        self.H0size = self.Hindex[Job.NAtom]
        self.HSOsize = 2*self.H0size

        # Allocate memory for the Hamiltonian matrices
        self.H0   = np.zeros((self.H0size,  self.H0size),  dtype='double')
        self.HSO  = np.zeros((self.HSOsize, self.HSOsize), dtype='complex')
        self.fock = np.zeros((self.HSOsize, self.HSOsize), dtype='complex')

        # Allocate memeory for the charge and spin
        self.q = np.zeros(Job.NAtom, dtype='double')
        self.s = np.zeros((3, Job.NAtom), dtype='double')

        if Job.Def["spin_orbit"] == 1:
            # Spin orbit data
            self.SOmatrix = {0: np.array([[complex( 0.0, 0.0), complex( 0.0, 0.0)],
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

        if Job.Def["Hamiltonian"] == "dcase":
            self.quad_xi_mat = np.zeros((5, 5, 5, 5), dtype='double')
            for alpha in range(5):
                for beta in range(5):
                    for gamma in range(5):
                        for chi in range(5):
                            self.quad_xi_mat[alpha, beta, gamma, chi] = xicontfour(alpha, beta, gamma, chi)

    def electrostatics(self, Job):
        def SCFGamma(self, Job, atom1, atom2):

            atype1 = Job.AtomType[atom1]
            atype2 = Job.AtomType[atom2]

            if atom1 == atom2:
                gij = complex(Job.Model.atomic[atype1]['U'], 0.0)
            else:
                if Job.Def["InterSiteElectrostatics"] == 1:
                    dr = Job.Pos[atom2] - Job.Pos[atom1]
                    rij = np.sqrt(dr.dot(dr))

                    vacuum = 0.00552639
                    ai = vacuum*vacuum*np.pi*np.pi*np.pi*8.0 * Job.Model.atomic[atype1]['U'] * Job.Model.atomic[atype1]['U']
                    aj = vacuum*vacuum*np.pi*np.pi*np.pi*8.0 * Job.Model.atomic[atype2]['U'] * Job.Model.atomic[atype2]['U']
                    b =  np.sqrt(ai * aj / (ai + aj))

                    gij = 1.0/vacuum/(4.0*np.pi) * complex(erf(b * rij) / rij, 0.0)
                else:
                    gij = complex(0.0, 0.0)

            return gij

        SCF1g = np.zeros((Job.NAtom, Job.NAtom), dtype="complex")
        Wi = np.zeros(Job.NAtom, dtype="complex")

        # Compute SCF1g matrix
        for i in range(Job.NAtom):
            for j in range(i, Job.NAtom):
                gij = SCFGamma(self, Job, i, j)
                SCF1g[i, j] = gij
                SCF1g[j, i] = gij

        # Compute electrostatic term
        for i in range(Job.NAtom):
            for j in range(Job.NAtom):
                Wi[i] += -self.q[j] * SCF1g[i, j]

        self.Wi = Wi

    def buildFock(self, Job):
        """Build the Fock matrix by adding charge and spin terms to the Hamiltonian."""
        # Copy the Hamiltonian, complete with spin-orbit terms, to the Fock matrix
        self.fock = np.copy(self.HSO)
        h0s = self.H0size
        norb = Job.NOrb
        natom = Job.NAtom
        rho = Job.Electron.rhotot

        if Job.Def["Hamiltonian"] == "collinear":

            # Add in diagonal corrections for charge and spin
            des = np.zeros((2, 2), dtype="complex")

            # Compute on-site and (if enabled) intersite electrostatic terms.
            self.electrostatics(Job)

            for a in range(Job.NAtom):
                #
                # Get the atom type
                atype = Job.AtomType[a]
                #
                # Stoner onsite energy shifts are present for all four spins combinations
                des[0, 0] = -0.5 * Job.Model.atomic[atype]['I'] * complex( self.s[2, a],           0.0)
                des[0, 1] = -0.5 * Job.Model.atomic[atype]['I'] * complex( self.s[0, a], -self.s[1, a])
                des[1, 0] = -0.5 * Job.Model.atomic[atype]['I'] * complex( self.s[0, a],  self.s[1, a])
                des[1, 1] = -0.5 * Job.Model.atomic[atype]['I'] * complex(-self.s[2, a],           0.0)
                #
                # Step through each orbital on the atom
                for j in range(self.Hindex[a], self.Hindex[a+1]):
                    self.fock[    j,     j] += self.Wi[a] + des[0, 0]  # up/up block
                    self.fock[    j, h0s+j] +=              des[0, 1]  # up/down block
                    self.fock[h0s+j,     j] +=              des[1, 0]  # down/up block
                    self.fock[h0s+j, h0s+j] += self.Wi[a] + des[1, 1]  # down/down block

        else:
            for a in range(natom):
                # Get the atom type
                atype = Job.AtomType[a]
                J  = Job.Model.atomic[atype].get('I')
                U  = Job.Model.atomic[atype].get('U')
                dJ = Job.Model.atomic[atype].get('dJ')
                for jj in range(self.Hindex[a], self.Hindex[a+1]):
                    for ii in range(self.Hindex[a], self.Hindex[a+1]):
                        # up/up block
                        self.fock[    jj,     ii] += self.add_Coulomb_term(    jj,     ii, U, J, J, dJ, natom, norb, rho, Job.Def['Hamiltonian'])
                        # up/down block
                        self.fock[    jj, h0s+ii] += self.add_Coulomb_term(    jj, h0s+ii, U, J, J, dJ, natom, norb, rho, Job.Def['Hamiltonian'])
                        # down/up block
                        self.fock[h0s+jj,     ii] += self.add_Coulomb_term(h0s+jj,     ii, U, J, J, dJ, natom, norb, rho, Job.Def['Hamiltonian'])
                        # down/down block
                        self.fock[h0s+jj, h0s+ii] += self.add_Coulomb_term(h0s+jj, h0s+ii, U, J, J, dJ, natom, norb, rho, Job.Def['Hamiltonian'])

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
        K(2,3) = sqrt(3) xy
        K(2,4) = sqrt(3) (x^2-y^2)/2

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

            block[1,0] = block[0,1]
            block[1,1] = l*l*(v[0]-v[1]) + v[1]
            block[1,2] = l*m*(v[0]-v[1])

            block[2,0] = block[0,2]
            block[2,1] = block[1,2]
            block[2,2] = m*m*(v[0]-v[1]) + v[1]
        elif (l1,l2) == (2,2):
            # d-d
            block[0,0] = (n*n-0.5*(l*l+m*m))*(n*n-0.5*(l*l+m*m))*v[0] + 3*n*n*(l*l+m*m)*v[1] + 0.75*(l*l+m*m)*(l*l+m*m)*v[2]
            block[0,1] = math.sqrt(3.0)*n*l*(n*n*(v[0]-v[1]) - 0.5*(l*l+m*m)*(v[0] - 2.0*v[1] + v[2]))
            block[0,2] = math.sqrt(3.0)*n*m*(n*n*(v[0]-v[1]) - 0.5*(l*l+m*m)*(v[0] - 2.0*v[1] + v[2]))
            block[0,3] = math.sqrt(3.0)*l*m*          ((n*n-0.5*(l*l+m*m))*v[0] - 2.0*n*n*v[1] + 0.5*(1.0+n*n)*v[2])
            block[0,4] = 0.5*math.sqrt(3.0)*(l*l-m*m)*((n*n-0.5*(l*l+m*m))*v[0] - 2.0*n*n*v[1] + 0.5*(1.0+n*n)*v[2])

            block[1,0] = block[0,1]
            block[1,1] = 3.0*n*n*l*l*v[0] + (n*n+l*l-4.0*n*n*l*l)*v[1] + (m*m+n*n*l*l)*v[2]
            block[1,2] = l*m*(3.0*n*n*v[0] + (1.0-4.0*n*n)*v[1] + (n*n-1.0)*v[2])
            block[1,3] = n*m*(3*l*l*v[0] + (1.0-4.0*l*l)*v[1] + (l*l-1.0)*v[2])
            block[1,4] = n*l*(1.5*(l*l-m*m)*v[0] + (1.0-2.0*(l*l-m*m))*v[1] - (1.0-0.5*(l*l-m*m))*v[2])

            block[2,0] = block[0,2]
            block[2,1] = block[1,2]
            block[2,2] = 3.0*n*n*m*m*v[0] + (n*n+m*m-4.0*n*n*m*m)*v[1] + (l*l+n*n*m*m)*v[2]
            block[2,3] = n*l*(3.0*m*m*v[0] + (1.0-4.0*m*m)*v[1] + (m*m-1.0)*v[2])
            block[2,4] = n*m*(1.5*(l*l-m*m)*v[0] - (1.0+2.0*(l*l-m*m))*v[1] + (1.0+0.5*(l*l-m*m))*v[2])

            block[3,0] = block[0,3]
            block[3,1] = block[1,3]
            block[3,2] = block[2,3]
            block[3,3] = 3.0*l*l*m*m*v[0] + (l*l+m*m-4.0*l*l*m*m)*v[1] + (n*n+l*l*m*m)*v[2]
            block[3,4] = m*l*(1.5*(l*l-m*m)*v[0] - 2.0*(l*l-m*m)*v[1] + 0.5*(l*l-m*m)*v[2])

            block[4,0] = block[0,4]
            block[4,1] = block[1,4]
            block[4,2] = block[2,4]
            block[4,3] = block[3,4]
            block[4,4] = 0.75*(l*l-m*m)*(l*l-m*m)*v[0] + (l*l+m*m-(l*l-m*m)*(l*l-m*m))*v[1] + (n*n+0.25*(l*l-m*m)*(l*l-m*m))*v[2]

        # Return the Hamiltonian block
        return block

    def buildH0(self, Job):
        """
        Build the hamiltonian one block at a time, with a block corresponding to
        a pair of atoms.

        This function invokes a function to compute the hopping
        integrals for the reference geometry for a tight binding model using the
        coefficients associated with the model. The block of the Hamiltonian
        matrix is built using the integrals from the model and Slater-Koster tables.

        We are currently assuming in the periodic boundary conditions that the
        interactions do not extend any further than the adjacent 26 cells. If
        they do extend further, the number of cells to loop over would have to
        be calculated e.g. extend the jx, jy, and jz loops to go from -2 to 2
        or -3 to 3...
        """

        # Clear the memory for the Hamiltonian matrix
        self.H0.fill(0.0)

        # Step through all pairs of atoms
        for a1 in range(Job.NAtom):
            for a2 in range(a1, Job.NAtom):
                self.H0[self.Hindex[a1]:self.Hindex[a1+1],
                    self.Hindex[a2]:self.Hindex[a2+1]] = self.hopping_block(Job, a1, a2)
                # Now add the corresponding section with a1 and a2 swapped.
                self.H0[self.Hindex[a2]:self.Hindex[a2+1],
                    self.Hindex[a1]:self.Hindex[a1+1]] = \
                        np.transpose(self.H0[self.Hindex[a1]:self.Hindex[a1+1],
                            self.Hindex[a2]:self.Hindex[a2+1]])


        # periodic boundary conditions
        if Job.Def["PBC"] == 1:
            for a1 in range(Job.NAtom):
                for a2 in range(a1, Job.NAtom):
                    # loop over the 26 adjacent cubes.
                    for jx in range(-1,2):
                        for jy in range(-1,2):
                            for jz in range(-1,2):
                                # Do not consider the centre cube again
                                if (jx, jy, jz) == (0, 0, 0):
                                    pass
                                else:
                                    self.H0[self.Hindex[a1]:self.Hindex[a1+1],
                                        self.Hindex[a2]:self.Hindex[a2+1]] += \
                                            self.hopping_block(Job, a1, a2, jx, jy, jz)
                                    self.H0[self.Hindex[a2]:self.Hindex[a2+1],
                                        self.Hindex[a1]:self.Hindex[a1+1]] = \
                                            np.transpose(self.H0[self.Hindex[a1]:self.Hindex[a1+1],
                                                self.Hindex[a2]:self.Hindex[a2+1]])

    def hopping_block(self, Job, atom1, atom2, jx=0, jy=0, jz=0):
        """
        Calculate and return the matrix block of the hopping integrals between
        the provided two atoms, atom1 and atom2. This also works for periodic
        boundary conditions by supplying a displacement vector, jx, jy and jz,
        which by default is turned off. For the periodic boundary conditions
        a file containing the three unit cell vectors must be included in the
        simulation directory.
        """
        # If the atoms are the same, compute an onsite block, otherwise compute a hopping block
        type1 = Job.AtomType[atom1]
        if atom1 == atom2 and (jx, jy, jz) == (0, 0, 0):
            return Job.Model.atomic[type1]['e']
        else:
            type2 = Job.AtomType[atom2]
            norb1 = Job.Model.atomic[type1]['NOrbitals']
            norb2 = Job.Model.atomic[type2]['NOrbitals']
            temp_mat = np.zeros((norb1, norb2), dtype='double')
            # Compute the atomic displacement
            dr = Job.Pos[atom1] - (Job.Pos[atom2]
                                   + jx*Job.UnitCell[0]
                                   + jy*Job.UnitCell[1]
                                   + jz*Job.UnitCell[2])
            distance = np.sqrt(dr.dot(dr))

            # Build the set of hopping integrals
            v, v_bgn, v_end = Job.Model.helements(distance, type1, type2)

            # Build the block one pair of shells at a time
            k1 = 0  # Counter for orbital

            # Step through shells of atom 1
            for i1, l1 in enumerate(Job.Model.atomic[type1]['l']):
                n1 = 2*l1+1  # Compute number of orbitals in the shell
                k2 = 0  # Counter for orbital

                # Step through shells of atom 2
                for i2, l2 in enumerate(Job.Model.atomic[type2]['l']):
                    n2 = 2*l2+1  # Compute number of orbitals in the shell
                    temp_mat[k1:k1+n1, k2:k2+n2] = self.slaterkoster(l1, l2, dr, v[v_bgn[i1,i2]:v_end[i1,i2]])
                    k2 += n2  # Advance to the start of the next set of orbitals

                k1 += n1  # Advance to the start of the next set of orbitals

            return temp_mat

    def buildHSO(self, Job):
        """Build the Hamiltonian with spin orbit coupling."""

        h0s = self.H0size

        # Clear memory for the Hamiltonian spinorbit marix
        self.HSO.fill(0.0)

        # Build the basic Hamiltonian. This is independent of spin and appears in the
        # up-up and down-down blocks of the full spin dependent Hamiltonian
        self.buildH0(Job)

        # Copy H0 into the two diagonal blocks
        self.HSO[0           :h0s,            0:h0s] = np.copy(self.H0)
        self.HSO[h0s:self.HSOsize, h0s:self.HSOsize] = np.copy(self.H0)

        # Add in magnetic field contribution
        eB = Job.Def['so_eB']
        for i in range(h0s):
            self.HSO[      i,       i] += complex( eB[2],    0.0)
            self.HSO[      i, h0s + i] += complex( eB[0], -eB[1])
            self.HSO[h0s + i,       i] += complex( eB[0],  eB[1])
            self.HSO[h0s + i, h0s + i] += complex(-eB[2],    0.0)

        # Add in the spin-orbit corrections
        if Job.Def["spin_orbit"] == 1:
            for atom in range(Job.NAtom):
                atype = Job.AtomType[atom]

                k = self.Hindex[atom]  # Counter for orbital
                for i, l in enumerate(Job.Model.atomic[atype]['l']):  # Step through each shell
                    n = 2*l+1  # Compute number of orbitals in the shell
                    self.HSO[    k:    k+n,     k:    k+n] += Job.Model.atomic[atype]['so'][i]*self.SOmatrix[l][  0:  n,   0:  n]
                    self.HSO[    k:    k+n, h0s+k:h0s+k+n] += Job.Model.atomic[atype]['so'][i]*self.SOmatrix[l][  0:  n, n+0:n+n]
                    self.HSO[h0s+k:h0s+k+n,     k:    k+n] += Job.Model.atomic[atype]['so'][i]*self.SOmatrix[l][n+0:n+n,   0:  n]
                    self.HSO[h0s+k:h0s+k+n, h0s+k:h0s+k+n] += Job.Model.atomic[atype]['so'][i]*self.SOmatrix[l][n+0:n+n, n+0:n+n]
                    k += n  # Advance to the start of the next set of orbitals

    def add_Coulomb_term(self, ii, jj, U, J_S, J_ph, dJ, num_atoms,
                         num_orbitals, rho, hamiltonian):
        if hamiltonian == "scase":
            return self.add_H_scase(ii, jj, U, num_atoms, num_orbitals, rho)
        if hamiltonian == "pcase":
            return self.add_H_pcase(ii, jj, U, J_S, J_ph, num_atoms,
                                    num_orbitals, rho)
        if hamiltonian == "dcase":
            return self.add_H_dcase(ii, jj, U, J_S, J_ph, dJ, num_atoms,
                                    num_orbitals, rho)
        else:
            raise UnimplementedModelError(
                "Hamiltonian: '{}' is unrecognised".format(hamiltonian)
            )

    def add_H_scase(self, ii, jj, U, num_atoms, num_orbitals, rho):
        """
        Add the noncollinear Hamiltonian to the on-site contributions for
        the s-case Hubbard-like Hamiltonian.

        The tensorial form for this is:

        F^{s,s'}_{i j} = Kd(i,j) U (\sum_{s''}Kd(s,s') rho^{s'',s''}_{i i} - rho^{s,s'}_{i i} ),

        where Kd is the Kronecker delta symbol; s, s' and s'' are spin indices; i
        and j are atom indices; rho is the density matrix; U is the Coulomb
        integral.

        INPUT                 DATA TYPE       DESCRIPTION

        ii                    int             Fock matrix index 1.

        jj                    int             Fock matrix index 2.

        U                     float           The Coulomb integral.

        num_atoms             int             The number of atoms.

        num_orbitals          list of int     A list of the number of orbitals for
                                              each atom.

        rho                   numpy matrix    The density matrix.


        OUTPUT                DATA TYPE       DESCRIPTION

        F                     float           The Fock matrix element for s-case
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
            # The positive terms
            if s == sp:
                # the 0 is the orbital number (there is only 1 spatial orbital for s-orbitals).
                F += U*sum(rho[map_atomic_to_index(i, 0, sig, num_atoms, num_orbitals), map_atomic_to_index(i, 0, sig, num_atoms, num_orbitals)] for sig in range(2))
        # return the matrix element
        return F

    def add_H_pcase(self, ii, jj, U, J_S, J_ph, num_atoms, num_orbitals, rho):
        """
        Add the noncollinear Hamiltonian to the on-site contributions for
        the p-case Hubbard-like Hamiltonian.

        The tensorial form for this is:

        F^{s,s'}_{i a j b} = Kd(i,j) (\sum_{s''}Kd(s,s')(\sum_{a'}U Kd(a,b)
            rho^{s'',s''}_{i a' i a'} + J_S rho^{s'',s''}_{i a i b}
            + J_{ph} rho^{s'',s''}_{i b i a} )
            - (U rho^{s,s'}_{i a i b} + \sum_{a'} J_S Kd(a,b)rho^{s,s'}_{i a' i a'}
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

    def add_H_dcase(self, ii, jj, U, J_S, J_ph, dJ, num_atoms, num_orbitals, rho):
        """
        Add the noncollinear Hamiltonian to the on-site contributions for the
        d-case Hubbard-like Hamiltonian.

        The tensorial form for this is:

        F^{s,s'}_{i a j b} = Kd(i,j) (\sum_{s''}Kd(s,s')(\sum_{a'}U Kd(a,b) rho^{s'',s''}_{i a' i a'}
            + Jp_S rho^{s'',s''}_{i a i b} + Jp_{ph} rho^{s'',s''}_{i b i a}
            - 48 dJ \sum_{cd stuv} xi_{c st}xi_{a tu}xi_{d uv}xi_{b vs} rho^{i d i c})
            - (U rho^{s,s'}_{i a i b} + \sum_{a'} Jp_S Kd(a,b)rho^{s,s'}_{i a' i a'}
            + Jp_{ph} rho^{s,s'}_{i b i a}
            - 48 dJ \sum_{cd stuv} xi_{c st}xi_{a tu}xi_{b uv}xi_{d vs} rho^{i d i c})),

        where Kd is the Kronecker delta symbol; s, s' and s'' are spin indices; i
        and j are atom indices; a, a' and b are orbital indices; rho is the
        density matrix; U is the Hartree Coulomb integral and J_S and J_{ph} are
        the exchange Coulomb integrals, Jp_S = J_S + 5/2*dJ and the same for Jp_{ph}.
        Physically, J_S is equivalent to J_{ph}; however for the Stoner Hamiltonian J_S
        is the same as the Stoner I and J_{ph} is equal to zero.

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
        # atom, spatial orbital and spin for index 1
        i, a, s  = map_index_to_atomic(ii, num_atoms, num_orbitals)
        # atom, spatial orbital and spin for index 2
        j, b, sp = map_index_to_atomic(jj, num_atoms, num_orbitals)
        F = 0.0
        Jp_ph = J_ph+2.5*dJ
        Jp_S = J_S+2.5*dJ
        if i == j:
            # The negative terms
            F -= U*rho[ii, jj]
            F -= Jp_ph*rho[map_atomic_to_index(i, b, s, num_atoms, num_orbitals), map_atomic_to_index(i, a, sp, num_atoms, num_orbitals)]
            F += 48*dJ*sum(self.quad_xi_mat[orb1, a, b, orb2]*rho[map_atomic_to_index(i, orb2, s, num_atoms, num_orbitals), map_atomic_to_index(i, orb1, sp, num_atoms, num_orbitals)] for orb1 in range(num_orbitals[i]) for orb2 in range(num_orbitals[i]))
            if a == b:
                F -= Jp_S*sum(rho[map_atomic_to_index(i, orb, s, num_atoms, num_orbitals), map_atomic_to_index(i, orb, sp, num_atoms, num_orbitals)] for orb in range(num_orbitals[i]))
            # The positive terms
            if s == sp:
                F += Jp_S*sum(rho[map_atomic_to_index(i, a, sig, num_atoms, num_orbitals), map_atomic_to_index(i, b, sig, num_atoms, num_orbitals)] for sig in range(2))
                F += Jp_ph*sum(rho[map_atomic_to_index(i, b, sig, num_atoms, num_orbitals), map_atomic_to_index(i, a, sig, num_atoms, num_orbitals)] for sig in range(2))
                F -= 48*dJ*sum(self.quad_xi_mat[orb1, a, orb2, b]*rho[map_atomic_to_index(i, orb2, sig, num_atoms, num_orbitals), map_atomic_to_index(i, orb1, sig, num_atoms, num_orbitals)] for sig in range(2) for orb1 in range(num_orbitals[i]) for orb2 in range(num_orbitals[i]))
                if a == b:
                    F += U*sum(rho[map_atomic_to_index(i, orb, sig, num_atoms, num_orbitals), map_atomic_to_index(i, orb, sig, num_atoms, num_orbitals)] for sig in range(2) for orb in range(num_orbitals[i]))

        return F

    def total_energy(self, Job):
        """
        The expression for the total energy is:
            E = h_0 + \sum_{ij} (F_{ij} - 0.5*\tilde{F}_{ij})\rho_{ji}
        where \tilde{F} is the Coulombic part of the Fock Matrix.

        We first need to subtract the double counting correction and then we
        can perform the trace of the dot product of the corrected Fock matrix
        and the density matrix.
        """
        norb = Job.NOrb
        natom = Job.NAtom
        rho = Job.Electron.rhotot
        h0s = self.H0size

        corrected_fock = np.copy(self.fock)
        for a in range(natom):
            # Get the atom type
            atype = Job.AtomType[a]
            J  = Job.Model.atomic[atype].get('I')
            U  = Job.Model.atomic[atype].get('U')
            dJ = Job.Model.atomic[atype].get('dJ')
            for jj in range(self.Hindex[a], self.Hindex[a+1]):
                for ii in range(self.Hindex[a], self.Hindex[a+1]):
                    # up/up block
                    corrected_fock[    jj,     ii] -= 0.5*self.add_Coulomb_term(    jj,     ii, U, J, J, dJ, natom, norb, rho, Job.Def['Hamiltonian'])
                    # up/down block
                    corrected_fock[    jj, h0s+ii] -= 0.5*self.add_Coulomb_term(    jj, h0s+ii, U, J, J, dJ, natom, norb, rho, Job.Def['Hamiltonian'])
                    # down/up block
                    corrected_fock[h0s+jj,     ii] -= 0.5*self.add_Coulomb_term(h0s+jj,     ii, U, J, J, dJ, natom, norb, rho, Job.Def['Hamiltonian'])
                    # down/down block
                    corrected_fock[h0s+jj, h0s+ii] -= 0.5*self.add_Coulomb_term(h0s+jj, h0s+ii, U, J, J, dJ, natom, norb, rho, Job.Def['Hamiltonian'])

        return np.trace(np.dot(corrected_fock, rho))



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


def xicontfour(alph,bet,gam,chi):
    """
    The fourth contraction of the xi matrix.
    """
    xi=[[[-1.0/(2.0*math.sqrt(3)),0.0,0.0],
        [0.0,-1.0/(2.0*math.sqrt(3)),0.0],
        [0.0,0.0,1/math.sqrt(3)]],
        [[0.0,0.0,0.5],
        [0.0,0.0,0.0],
        [0.5,0.0,0.0]],
        [[0.0,0.0,0.0],
        [0.0,0.0,0.5],
        [0.0,0.5,0.0]],
        [[0.0,0.5,0.0],
        [0.5,0.0,0.0],
        [0.0,0.0,0.0]],
        [[0.5,0.0,0.0],
        [0.0,-0.5,0.0],
        [0.0,0.0,0.0]]]
    return sum(sum(sum(sum(xi[alph][s][t]*xi[bet][t][u]*xi[gam][u][v]*xi[chi][v][s] for s in range(3)) for t in range(3)) for u in range(3)) for v in range(3))
