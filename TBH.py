"""
Created on Thursday, April 16, 2015

@author: Andrew Horsfield and Marc Coury

This module builds the tight binding Hamiltonians.
It merges the old TBH0 and TBHSO modules
"""
#
# Import the modules that will be needed
import numpy as np
import math as m
import TBgeom
import TBelec
from Verbosity import *


# Functions used to evauate tight binding models
def model01(r, coeffs):
    #
    # This is a simple exponential model for an sp system
    #
    # Compute the hopping integrals for the reference geometry
    v = np.zeros(5, dtype='double')
    v[0] = coeffs['vsss']*m.exp(-coeffs['kss']*(r-coeffs['r0']))
    v[1] = coeffs['vsps']*m.exp(-coeffs['ksp']*(r-coeffs['r0']))
    v[2] = coeffs['vpss']*m.exp(-coeffs['kps']*(r-coeffs['r0']))
    v[3] = coeffs['vpps']*m.exp(-coeffs['kpp']*(r-coeffs['r0']))
    v[4] = coeffs['vppp']*m.exp(-coeffs['kpp']*(r-coeffs['r0']))
    #
    # Generate pair of indices for each pair of shells, showing which values
    #     of v to use
    v_bgn = np.zeros((2, 2), dtype='double')
    v_end = np.zeros((2, 2), dtype='double')
    #
    # ss
    v_bgn[0, 0] = 0
    v_end[0, 0] = 1
    #
    # sp
    v_bgn[0, 1] = 1
    v_end[0, 1] = 2
    #
    # ps
    v_bgn[1, 0] = 2
    v_end[1, 0] = 3
    #
    # pp
    v_bgn[1, 1] = 3
    v_end[1, 1] = 5
    #
    # Return integrals and indices
    return v, v_bgn, v_end
#
# Atom data
AtomData = [
    {'Name': 'Helix', 'ChemSymb': 'C ',
     'NElectrons': 4, 'NOrbitals': 4, 'NShells': 2,
     'l': (0, 1), 'U': 10.0, 'I': 5.0,
     'e': np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]]),
     'so': (0.0, 0.0)},
    {'Name': 'Carbon', 'ChemSymb': 'C ',
     'NElectrons': 4, 'NOrbitals': 4, 'NShells': 2,
     'l': (0, 1), 'U': 10.0, 'I': 5.0,
     'e': np.array([[0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]]),
     'so': (0.0, 0.0)}
]
#
# Bond data
BondData = [
    [
        {'model': model01,
            'kss': 1.0,  'ksp': 1.0, 'kps': 1.0, 'kpp': 1.0, 'r0': 1.0,
            'vsss': 0.0, 'vsps': 0.0, 'vpss': 0.0, 'vpps': -1.0, 'vppp': 0.5},
        {'model': model01,
            'kss': 1.0,  'ksp': 1.0, 'kps': 1.0, 'kpp': 1.0, 'r0': 1.0,
            'vsss': 0.0, 'vsps': 0.0, 'vpss': 0.0, 'vpps': -1.0, 'vppp': 0.5}
    ],
    [
        {'model': model01,
            'kss': 1.0,  'ksp': 1.0, 'kps': 1.0, 'kpp': 1.0, 'r0': 1.0,
            'vsss': 0.0, 'vsps': 0.0, 'vpss': 0.0, 'vpps': -1.0, 'vppp': 0.5},
        {'model': model01,
            'kss': 1.0,  'ksp': 1.0, 'kps': 1.0, 'kpp': 1.0, 'r0': 1.0,
            'vsss': 0.0, 'vsps': 0.0, 'vpss': 0.0, 'vpps': -1.0, 'vppp': 0.5}
    ]
]
#
# Spin orbit data
SOmatrix = {0: np.array([[complex( 0.0, 0.0), complex( 0.0, 0.0)],
                        [complex( 0.0, 0.0), complex( 0.0, 0.0)]]),
            1: np.array([[complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0), complex(-1.0, 0.0), complex( 0.0, 1.0)],
                        [complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0,-1.0), complex( 1.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0)],
                        [complex( 0.0, 0.0), complex( 0.0, 1.0), complex( 0.0, 0.0), complex( 0.0,-1.0), complex( 0.0, 0.0), complex( 0.0, 0.0)],
                        [complex( 0.0, 0.0), complex( 1.0, 0.0), complex( 0.0, 1.0), complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0)],
                        [complex(-1.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 1.0)],
                        [complex( 0.0,-1.0), complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0, 0.0), complex( 0.0,-1.0), complex( 0.0, 0.0)]])}


# Initialise the module
def init(JobDef):
    global H0, H0size, Hindex
    global HSO, HSOsize
    global Fock, q, s
    #
    # Build index of starting position in Hamiltonian for each atom
    Hindex = np.zeros(TBgeom.NAtom+1, dtype='int')
    Hindex[0] = 0
    for a in range(0, TBgeom.NAtom):
        Hindex[a+1] = Hindex[a] + AtomData[TBgeom.AtomType[a]]['NOrbitals']
    #
    # Compute the sizes of the Hamiltonian matrices
    H0size = Hindex[TBgeom.NAtom]
    HSOsize = 2*H0size
    #
    # Allocate memory for the Hamiltonian matrices
    H0 = np.zeros((H0size, H0size), dtype='double')
    HSO = np.zeros((HSOsize, HSOsize), dtype='complex')
    Fock = np.zeros((HSOsize, HSOsize), dtype='complex')
    #
    # Allocate memeory for the charge and spin
    q = np.zeros(TBgeom.NAtom, dtype='double')
    s = np.zeros((3, TBgeom.NAtom), dtype='double')


# Build the Fock matrix by adding charge and spin terms to the Hamiltonian
def BuildFock(JobDef):
    global Fock, HSO, H0size, Hindex
    global AtomData
    global q, s
    #
    # Copy the Hamiltonian, complete with spin-orbit terms, to the Fock matrix 
    Fock = np.copy(HSO)
    #
    # Now add in diagonal corrections for charge and spin
    des = np.zeros((2,2), dtype='complex')
    for a in range(0,TBgeom.NAtom):
        #
        # Get the atom type
        ta = TBgeom.AtomType[a]
        #
        # Onsite energy shift is U*q
        deq = complex(-q[a]*AtomData[ta]['U'], 0.0)
        #
        # Stoner onsite energy shifts are present for all four spins combinations
        des[0,0] = -0.5*AtomData[ta]['I']*complex( s[2,a],     0.0)
        des[0,1] = -0.5*AtomData[ta]['I']*complex( s[0,a], -s[1,a])
        des[1,0] = -0.5*AtomData[ta]['I']*complex( s[0,a],  s[1,a])
        des[1,1] = -0.5*AtomData[ta]['I']*complex(-s[2,a],     0.0)
        #
        # Step through each orbital on the atom
        for j in range(Hindex[a], Hindex[a+1]):
            Fock[       j,       j] += deq + des[0,0]  # up/up block
            Fock[       j,H0size+j] +=       des[0,1]  # up/down block
            Fock[H0size+j,       j] +=       des[1,0]  # down/up block
            Fock[H0size+j,H0size+j] += deq + des[1,1]  # down/down block

    #
    # !!!!!DEBUGGING PURPOSES!!!!!!
    #
    Fock2 = np.copy(HSO)
    norb = [AtomData[TBgeom.AtomType[a]]['NOrbitals'] for a in range(TBgeom.NAtom)]
    natom = TBgeom.NAtom
    rho = TBelec.rho
    for a in range(0,natom):
        # Get the atom type
        ta = TBgeom.AtomType[a]
        for j in range(Hindex[a],Hindex[a+1]):
            J_ph = 0.0
            J_S = AtomData[ta]['I']
            U = AtomData[ta]['U']
            
            # up/up block
            Fock2[       j,        j] += H_pcase(       j,        j, U, J_S, J_ph, natom, norb, rho)
            # up/down block
            Fock2[       j, H0size+j] += H_pcase(       j, H0size+j, U, J_S, J_ph, natom, norb, rho)
            # down/up block
            Fock2[H0size+j,        j] += H_pcase(H0size+j,        j, U, J_S, J_ph, natom, norb, rho)
            # down/down block
            Fock2[H0size+j, H0size+j] += H_pcase(H0size+j, H0size+j, U, J_S, J_ph, natom, norb, rho)

    for i in range(HSOsize):
        for j in range(i,HSOsize):
            if abs(Fock[i,j]-Fock2[i,j]) > 0.000001:
                verboseprint(JobDef['extraverbose'], i, j, round(Fock[i,j].real,4), round(Fock2[i,j].real,4), "|", round(Fock[i,j].imag,4), round(Fock2[i,j].imag,4))

#
# Evaluate the Slater-Koster table. The input is a pair of angular momentum quantum
# numbers (l1 and l2), the displacement between atoms 1 and 2 (dr, equal to r1 - r2),
# and a tuple that provides the (sigma, pi, delta , ...) hopping matrix elements.
# The function returns the full block of the Hamiltonian.
#
# The sign convention used here is as follows. The tables are computed with the atom
# corresponding to the second index being at the origin, and the atom with the first
# index placed vertically above it along the positive z axis.
#
# The cubic harmonic (K) orbital convention used is as follows. Let the magnetic quantum number
# be m. Then:
#   K(l,2m-1) = sqrt(4 pi/(2l+1)) r^l ((-1)^m Y(l,m) + Y(l,-m))/sqrt(2)
#   K(l,2m  ) = sqrt(4 pi/(2l+1)) r^l ((-1)^m Y(l,m) - Y(l,-m))/(i sqrt(2))
# Thus we have
#   K(0,0) = 1
#
#   K(1,0) = z
#   K(1,1) = x
#   K(1,2) = y
#
#   K(2,0) = (3z^2 - r^2)/2
#   K(2,1) = sqrt(3) zx
#   K(2,2) = sqrt(3) zy
#   K(2,3) = sqrt(3) (x^2-y^2)/2
#   K(2,4) = sqrt(3) xy
#
# etc
def SlaterKoster(l1, l2, dr, v):
    import math
    #
    # Allocate space for the final block of the Hamiltonian
    block = np.zeros((2*l1+1, 2*l2+1), dtype='double')
    #
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
    #
    # Evaluate the matrix elements
    if (l1,l2) == (0,0):
        #
        # s-s
        block[0,0] = v[0]
    elif (l1,l2) == (1,0):
        #
        # p-s
        block[0,0] = n*v[0]
        block[1,0] = l*v[0]
        block[2,0] = m*v[0]
    elif (l1,l2) == (0,1):
        #
        # s-p
        block[0,0] = n*v[0]
        block[0,1] = l*v[0]
        block[0,2] = m*v[0]
    elif (l1,l2) == (1,1):
        #
        # p-p
        block[0,0] = n*n*(v[0]-v[1]) + v[1]
        block[0,1] = n*l*(v[0]-v[1])
        block[0,2] = n*m*(v[0]-v[1])
        #
        block[1,0] = l*n*(v[0]-v[1])
        block[1,1] = l*l*(v[0]-v[1]) + v[1]
        block[1,2] = l*m*(v[0]-v[1])
        #
        block[2,0] = m*n*(v[0]-v[1])
        block[2,1] = m*l*(v[0]-v[1])
        block[2,2] = m*m*(v[0]-v[1]) + v[1]
    #
    # Return the Hamiltonian block
    return block
#
# Function to buld the hamiltonian one block at a time, with a block corresponding to
# a pair of atoms. This function invokes a function to compute the hopping
# integrals for the reference geometry for a tight binding model using the
# coefficients associated with the model. The block of the Hamiltonian
# matrix is built using the integrals from the model and Slater-Koster table
def BuildH0():
    global H0, H0size, Hindex
    global AtomData, BondData
    #
    # Clear the memory for the Hamiltonian matrix
    H0 = np.zeros([H0size, H0size], dtype='double')
    #
    # Step through all pairs of atoms
    for a1 in range(0, TBgeom.NAtom):
        t1 = TBgeom.AtomType[a1] # The type of atom 1
        for a2 in range(0, TBgeom.NAtom):
            t2 = TBgeom.AtomType[a2] # The type of atom 2
            #
            # If the atoms are the same, compute an onsite block, otherwise compute a hopping block
            if a1 == a2:
                H0[Hindex[a1]:Hindex[a1+1], Hindex[a2]:Hindex[a2+1]] = AtomData[t1]['e']
            else:
                #
                # Compute the atomic displacements
                dr = (TBgeom.Pos[0,a1]-TBgeom.Pos[0,a2], TBgeom.Pos[1,a1]-TBgeom.Pos[1,a2], TBgeom.Pos[2,a1]-TBgeom.Pos[2,a2])
                d = m.sqrt(dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2])
                #
                # Build the set of hopping integrals
                v, v_bgn, v_end = BondData[t1][t2]['model'](d, BondData[t1][t2])
                #
                # Build the block one pair of shells at a time
                i1 = 0 # Counter for shell
                k1 = Hindex[a1] # Counter for orbital
                for l1 in AtomData[t1]['l']: # Step through each shell
                    n1 = 2*l1+1 # Compute number of orbitals in the shell
                    #
                    i2 = 0 # Counter for shell
                    k2 = Hindex[a2] # Counter for orbital
                    for l2 in AtomData[t2]['l']: # Step through each shell
                        n2 = 2*l2+1 # Compute number of orbitals in the shell
                        H0[k1:k1+n1, k2:k2+n2] = SlaterKoster(l1, l2, dr, v[v_bgn[i1,i2]:v_end[i1,i2]])
                        i2 += 1 # Advance to the next shell
                        k2 += n2 # Advance to the start of the next set of orbitals
                    #
                    i1 += 1 # Advance to the next shell
                    k1 += n1 # Advance to the start of the next set of orbitals
#
# Function to add spin orbit coupling to the Hamiltonian
def BuildHSO(JobDef):
    global H0, H0size, Hindex
    global HSO, HSOsize
    global AtomData, SOmatrix
    #
    # Clear memory for the Hamiltonian marix
    HSO = np.zeros((HSOsize, HSOsize), dtype='complex')
    #
    # Build the basic Hamiltonian. This is independent of spin and appears in the
    # up-up and down-down blocks of the full spin dependent Hamiltonian
    BuildH0()
    #
    # Copy H0 into the two diagonal blocks
    HSO[0     :H0size ,0     :H0size ] = np.copy(H0)
    HSO[H0size:HSOsize,H0size:HSOsize] = np.copy(H0)
    #
    # Add in magnetic field contribution
    eB = JobDef['so_eB']
    for i in range(0,H0size):
        HSO[         i,          i] += complex( eB[2],    0.0)
        HSO[         i, H0size + i] += complex( eB[0], -eB[1])
        HSO[H0size + i,          i] += complex( eB[0],  eB[1])
        HSO[H0size + i, H0size + i] += complex(-eB[2],    0.0)
    #
    # Add in the spin-orbit corrections
    for a in range(0, TBgeom.NAtom):
        ta = TBgeom.AtomType[a]
        i = 0 # Counter for shell
        k = Hindex[a] # Counter for orbital
        for l in AtomData[ta]['l']: # Step through each shell
            n = 2*l+1 # Compute number of orbitals in the shell
            HSO[       k:       k+n,        k:       k+n] += AtomData[ta]['so'][i]*SOmatrix[l][  0:  n,  0:  n]
            HSO[       k:       k+n, H0size+k:H0size+k+n] += AtomData[ta]['so'][i]*SOmatrix[l][  0:  n, n+0:n+n]
            HSO[H0size+k:H0size+k+n,        k:       k+n] += AtomData[ta]['so'][i]*SOmatrix[l][n+0:n+n,  0:  n]
            HSO[H0size+k:H0size+k+n, H0size+k:H0size+k+n] += AtomData[ta]['so'][i]*SOmatrix[l][n+0:n+n, n+0:n+n]
            i += 1  # Advance to the next shell
            k += n  # Advance to the start of the next set of orbitals


def Kd(a, b):
    '''
    Function Kd is the Kronecker delta symbol. If a == b then return 1 otherwise
    return zero.

    INPUT                 DATA TYPE       DESCRIPTION

    a                     int             Value 1

    b                     int             Value 2


    OUTPUT                DATA TYPE       DESCRIPTION

    Kd                    int             delta_{a,b}

    '''
    if a == b:
        return 1
    else:
        return 0


def map_index_to_atomic(index, num_atoms, num_orbitals):
    '''
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

    '''
    upper_bound = 0
    # loop over spin
    for ss in range(2):
        # loop over the atoms
        for ii in range(num_atoms):
            upper_bound +=num_orbitals[ii]
            if index < upper_bound:
                atom = ii
                spin = ss
                # orbital is the index minus the previous upper bound
                orbital = index-(upper_bound-num_orbitals[ii])
                return atom, orbital, spin


def map_atomic_to_index(atom, orbital, spin, num_atoms, num_orbitals):
    '''
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

    '''
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


def H_pcase(ii, jj, U, J_S, J_ph, num_atoms, num_orbitals, rho):
    '''
    The function H_pcase takes the noncollinear Hamiltonian and adds to it the
    on-site contributions for the p-case Hubbard-like Hamiltonian.

    The tensorial form for this is:

    F^{s,s'}_{i a j b} = Kd(i,j) (\sum_{s''}Kd(s,s')(\sum_{a'}U Kd(a,b) rho^{s'',s''}_{i a i a} + J_S rho^{s'',s''}_{i a i b} + J_{ph} rho^{s'',s''}_{i b i a} )
        - (U rho^{s,s'}_{i a i b} + \sum_{a'} J_S Kd(a,b)rho^{s,s'}_{i a i a} + J_{ph} rho^{s,s'}_{i b i a}) ),

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

    '''
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


def H_dcase():
    '''
    The function H_dcase takes the noncollinear Hamiltonian and adds to it the
    on-site contributions for the d-case Hubbard-like Hamiltonian.

    The tensorial form for this is:

    F^{s,s'}_{i a j b} = Kd(i,j) (\sum_{s''}Kd(s,s')(\sum_{a'}U Kd(a,b) rho^{s'',s''}_{i a i a} 
        + J'_S rho^{s'',s''}_{i a i b} + J'_{ph} rho^{s'',s''}_{i b i a} 
        - 48 dJ \sum_{cd stuv} xi_{c st}xi_{a tu}xi_{b uv}xi_{d vs} rho^{})
        - (U rho^{s,s'}_{i a i b} + \sum_{a'} J_S Kd(a,b)rho^{s,s'}_{i a i a} + J_{ph} rho^{s,s'}_{i b i a}) ),

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

    '''
