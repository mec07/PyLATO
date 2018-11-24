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


class TestHamiltonian:
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
        ("hamiltonian", "U", "J", "dJ", "norb", "result"),
        [
            ("scase", 1, 0, 0, [1], 1),
            ("pcase", 1, 1, 0, [3], 5),
            ("dcase", 1, 1, 1, [5], 18),
        ]

    )
    def test_add_Coulomb_term_happy_path(self, hamiltonian, U, J, dJ, norb,
                                         result):
        Job = InitJob("test_data/JobDef_dcase.json")
        rho = np.eye(10)
        natoms = 1

        hami = Hamiltonian(Job)

        assert result == hami.add_Coulomb_term(0, 0, U, J, J, dJ, natoms,
                                               norb, rho, hamiltonian)

    def test_add_Coulomb_term_unrecognised_model_error(self):
        Job = InitJob("test_data/JobDef_dcase.json")
        rho = np.eye(10)

        hami = Hamiltonian(Job)

        with pytest.raises(UnimplementedModelError):
            hami.add_Coulomb_term(0, 0, 1, 1, 1, 1, 1, 1, rho,
                                  "invalid_hami_type")
