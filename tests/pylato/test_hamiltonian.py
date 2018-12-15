import numpy as np
import pytest

from pylato.exceptions import UnimplementedModelError
from pylato.hamiltonian import Hamiltonian
from pylato.init_job import InitJob


def fake_helements(*args):
    v = np.zeros(2, dtype="double")
    v[0] = -1.0
    v_bgn = np.array([[0]])
    v_end = np.array([[1]])

    return v, v_bgn, v_end


def fake_electrostatics(self, Job):
    self.Wi = [1 for i in range(Job.NAtom)]


def fake_spin_onsite_energy_shift(*args):
    return 1


class TestHamiltonian:
    def test_build_fock_collinear(self, monkeypatch):
        # Fake
        # Mock out the electrostatics method to just set Wi to be an array of
        # ones.
        monkeypatch.setattr('pylato.hamiltonian.Hamiltonian.electrostatics',
                            fake_electrostatics)

        # Setup
        Job = InitJob("test_data/JobDef_collinear.json")

        Job.Hamilton.buildHSO(Job)
        Job.e, Job.psi = np.linalg.eigh(Job.Hamilton.HSO)
        Job.Electron.occupy(
            Job.e, Job.Def['el_kT'], Job.Def['mu_tol'], Job.Def['mu_max_loops']
        )
        Job.Electron.densitymatrix()

        # we need rhotot to build the fock
        Job.Electron.rhotot = Job.Electron.rho

        # Fake the spins and stoner I:
        for a in range(Job.NAtom):
            atype = Job.AtomType[a]
            Job.Model.atomic[atype]['I'] = 1
            for b in range(3):
                Job.Hamilton.s[b, a] = 1
        # Fake HSO
        Job.Hamilton.HSO = np.eye(4)

        expected_fock = np.array([
            [1.5, 0, -0.5, 0],
            [0, 1.5, 0, -0.5],
            [-0.5, 0, 2.5, 0],
            [0, -0.5, 0, 2.5]
        ])

        # Action
        Job.Hamilton.buildFock(Job)

        # Result
        assert np.array_equal(Job.Hamilton.fock, expected_fock)

    def test_total_energy_happy_path(self):
        """
        The expression for the total energy is:
            E = h_0 + \sum_{ij} (F_{ij} - 0.5*\tilde{F}_{ij})\rho_{ji}
        where \tilde{F} is the Coulombic part of the Fock Matrix.
        """
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")

        # To obtain a simple Fock matrix, we can just mock the helements method
        # from the model (i.e. remove any distance dependence).
        Job.Model.helements = fake_helements

        #####################################
        # create fock matrix & density matrix
        #####################################
        Job.Hamilton.buildHSO(Job)
        Job.e, Job.psi = np.linalg.eigh(Job.Hamilton.HSO)
        Job.Electron.occupy(
            Job.e, Job.Def['el_kT'], Job.Def['mu_tol'], Job.Def['mu_max_loops']
        )
        Job.Electron.densitymatrix()

        # we need rhotot to build the fock
        Job.Electron.rhotot = Job.Electron.rho
        Job.Hamilton.buildFock(Job)

        #####################################
        # Calculate the expected total energy
        #####################################
        # For the simple scase dimer, the Coulomb part of the Fock matrix are
        # just the diagonal terms. Therefore, to calculate the Fock matrix with
        # the double counting correction, we can just subtract 0.5*diagonal of
        # the Fock matrix from the Fock matrix.
        corrected_fock = (
            Job.Hamilton.fock - 0.5*np.diag(np.diag(Job.Hamilton.fock))
        )

        # Just calculate the expected energy as a straight summation (no numpy
        # trickery -- that will be reserved for the total_energy method)
        expected_total_energy = 0
        for ii in range(4):
            for jj in range(4):
                expected_total_energy += (
                    corrected_fock[ii, jj]*Job.Electron.rho[jj, ii]
                )

        # Action
        energy = Job.Hamilton.total_energy(Job)

        # Result
        assert energy == expected_total_energy

    @pytest.mark.parametrize(
        ("hamiltonian", "U", "J", "dJ", "norb", "expected_value"),
        [
            ("collinear", 0, 1, 0, [1], 2),
            ("scase", 1, 0, 0, [1], 1),
            ("pcase", 1, 1, 0, [3], 5),
            ("dcase", 1, 1, 1, [5], 18),
            ("vector_stoner", 1, 1, 0, [3], 4),
            ("vector_stoner", 1, 1, 1, [5], 6),
        ]

    )
    def test_add_Coulomb_term_happy_path(self, monkeypatch, hamiltonian, U, J,
                                         dJ, norb, expected_value):
        # fake spin_onsite_energy_shift (for collinear case)
        monkeypatch.setattr(
            'pylato.hamiltonian.Hamiltonian.spin_onsite_energy_shift',
            fake_spin_onsite_energy_shift
        )

        # Setup
        Job = InitJob("test_data/JobDef_dcase.json")
        rho = np.eye(10)
        atype = Job.AtomType[0]
        Job.NAtom = 1
        Job.NOrb = norb
        Job.Model.atomic[atype]['I'] = J
        Job.Def["Hamiltonian"] = hamiltonian

        hami = Hamiltonian(Job)

        # fake electrostatics (for collinear case)
        hami.Wi = [1 for i in range(Job.NAtom)]

        # Action
        result = hami.add_Coulomb_term(Job, 0, 0, U, J, J, dJ, rho)

        # Result
        assert result == expected_value

    def test_add_Coulomb_term_unrecognised_model_error(self):
        Job = InitJob("test_data/JobDef_dcase.json")
        rho = np.eye(10)

        hami = Hamiltonian(Job)
        Job.Def["Hamiltonian"] = "invalid_hami_type"

        with pytest.raises(UnimplementedModelError):
            hami.add_Coulomb_term(Job, 0, 0, 1, 1, 1, 1, rho)

    @pytest.mark.parametrize(
        ("spin1", "spin2", "expected_value"),
        [
            (0, 0, -0.5),
            (0, 1, complex(-0.5, 0.5)),
            (1, 0, complex(-0.5, -0.5)),
            (1, 1, 0.5),
        ]
    )
    def test_spin_onsite_energy_shift(self, spin1, spin2, expected_value):
        # Setup
        Job = InitJob("test_data/JobDef_collinear.json")

        # Fake the Stoner I and the spin values for all atoms
        for atom in range(len(Job.Model.atomic)):
            atype = Job.AtomType[atom]
            Job.Model.atomic[atype]['I'] = 1
            Job.Hamilton.s[0, atom] = 1
            Job.Hamilton.s[1, atom] = 1
            Job.Hamilton.s[2, atom] = 1

        # Action
        result = Job.Hamilton.spin_onsite_energy_shift(Job, spin1, spin2, 0)

        # Result
        assert result == expected_value

    @pytest.mark.parametrize(
        ("name", "ii", "jj", "expected_value"),
        [
            ("same atom (0) and same spin (0)", 0, 0, 2),
            ("same atom (1) and same spin (0)", 1, 1, 2),
            ("same atom (0) and same spin (1)", 2, 2, 2),
            ("same atom (1) and same spin (1)", 3, 3, 2),
            ("same atom (0) opposite spin", 0, 2, 1),
            ("same atom (1) opposite spin", 1, 3, 1),
            ("different atom and same spin (0)", 0, 1, 0),
            ("different atom and same spin (1)", 3, 2, 0),
            ("different atom and opposite spin", 0, 3, 0),
            ("different atom and opposite spin again", 2, 1, 0),
        ]
    )
    def test_add_H_collinear(self, monkeypatch, name, ii, jj, expected_value):
        # fake spin_onsite_energy_shift
        monkeypatch.setattr(
            'pylato.hamiltonian.Hamiltonian.spin_onsite_energy_shift',
            fake_spin_onsite_energy_shift
        )

        # Setup
        Job = InitJob("test_data/JobDef_collinear.json")
        # fake electrostatics (Hamilton.Wi)
        Job.Hamilton.Wi = [1 for i in range(Job.NAtom)]

        # Action
        result = Job.Hamilton.add_H_collinear(Job, ii, jj)

        # Result
        assert result == expected_value

    def test_buildH0_one_atom(self):
        """
        A single scase atom without PBCs should have an H0 of: [[e]],
        where e is the onsite energy.
        """
        # Setup
        Job = InitJob("test_data/JobDef_single_atom.json")
        expected_result = np.array(Job.Model.atomic[0]['e'])

        # Action
        Job.Hamilton.buildH0(Job)

        # Result
        assert np.array_equal(Job.Hamilton.H0, expected_result)

    def test_buildH0_two_atoms(self):
        """
        Two scase atoms without PBCs should have an H0 of: [
            [e, t],
            [t, e]
        ],
        where e is the onsite energy and t is the value of the interatomic
        hopping integral.
        """
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")
        e = Job.Model.atomic[0]['e'][0][0]
        expected_result = np.array([
            [e, -1],
            [-1, e]
        ])

        # Action
        Job.Hamilton.buildH0(Job)

        # Result
        assert np.array_equal(Job.Hamilton.H0, expected_result)
