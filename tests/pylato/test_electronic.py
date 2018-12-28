import json
import numpy as np
import pytest

from pylato.electronic import Electronic
from pylato.exceptions import UnimplementedMethodError
from pylato.init_job import InitJob

from tests.conftest import load_json_file


class TestElectronic:
    def test_init_input_density_matrix(self):
        # Setup
        Job = InitJob("test_data/JobDef_input_density_matrix.json")
        input_rho_file = Job.Def['input_rho']
        with open(input_rho_file, 'r') as file_handle:
            input_rho = np.matrix(json.load(file_handle))

        # Action
        electronic = Electronic(Job)

        # Result
        assert np.array_equal(electronic.rho, input_rho)
        assert np.array_equal(electronic.rhotot, input_rho)

    def test_init_incorrect_input_density_matrix_dimensions(self):
        # Setup
        Job = InitJob("test_data/JobDef_input_density_matrix.json")
        bad_rho_file = "test_data/bad_rho.json"
        Job.Def['input_rho'] = bad_rho_file

        expected_rho = np.matrix(np.zeros(
            (Job.Hamilton.HSOsize, Job.Hamilton.HSOsize), dtype='complex'))

        # Action
        electronic = Electronic(Job)

        # Result
        assert np.array_equal(electronic.rho, expected_rho)
        assert np.array_equal(electronic.rhotot, expected_rho)

    @pytest.mark.parametrize(
        ("name", "rho", "expected_S"),
        [
            ("singlet", [
                [0.5, 0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ], 0),
            ("triplet up", [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ], 1),
            ("triplet down", [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], 1),
        ]
    )
    def test_quantum_number_S(self, name, rho, expected_S):
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")
        electronic = Electronic(Job)
        # Spin 0 density matrix
        electronic.rho = np.matrix(rho)

        # Action
        S = electronic.quantum_number_S(Job)

        # Result
        assert S == expected_S

    @pytest.mark.parametrize(
        ("name", "rho", "expected_L_z"),
        [
            ("s atom", [
                [1.0, 0.0],
                [0.0, 1.0]
            ], 0),
            ("p atom 2 electrons", [
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ], 1),
            ("p atom 3 electrons", [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ], 0),
            ("d atom 2 electrons", [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ], 1),

        ]
    )
    def test_quantum_number_L_z(self, name, rho, expected_L_z):
        # Setup
        Job = InitJob("test_data/JobDef_pcase.json")

        # Fake
        Job.NAtom = 1
        Job.NOrb = [len(rho)/2]
        electronic = Electronic(Job)
        electronic.NElectrons = sum(rho[ii][ii] for ii in range(len(rho)))
        electronic.rho = np.matrix(rho, dtype='complex')

        # Action
        L_z = electronic.quantum_number_L_z(Job)

        # Result
        assert L_z == expected_L_z

    @pytest.mark.parametrize(
        ("job_file", "rho_file", "expected_L_z"),
        [
            ("test_data/JobDef_scase.json", "test_data/rho_scase.json", 0),
            ("test_data/JobDef_pcase.json", "test_data/rho_pcase_2.json", 1),
            ("test_data/JobDef_dcase.json", "test_data/rho_dcase.json", 0),
        ]
    )
    def test_quantum_number_L_z_dimers(self, job_file, rho_file, expected_L_z):
        # Setup
        Job = InitJob(job_file)
        electronic = Electronic(Job)
        electronic = Electronic(Job)

        # Fake
        rho = load_json_file(rho_file)
        electronic.rho = np.matrix(rho, dtype='complex')

        # Action
        L_z = electronic.quantum_number_L_z(Job)

        # Result
        assert L_z == expected_L_z

    def test_quantum_number_L_z_not_implemented_error(self):
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")
        Job.NOrb = [1, 3]
        electronic = Electronic(Job)
        expected_message = (
            "Quantum Number L_z methods have only been implemented for "
            "simulations consisting of solely s, p or d orbital atoms"
        )

        # Action
        with pytest.raises(UnimplementedMethodError, message=expected_message):
            print(electronic.quantum_number_L_z(Job))

    def test_quantum_number_L_z_1_electron(self):
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")
        electronic = Electronic(Job)
        electronic.NElectrons = 1

        # Action
        L_z = electronic.quantum_number_L_z(Job)

        # Result
        assert L_z == 0


