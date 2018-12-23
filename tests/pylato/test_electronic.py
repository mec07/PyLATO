import json
import numpy as np
import pytest

from pylato.electronic import Electronic
from pylato.init_job import InitJob


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
    def test_quantum_number_S(self, name, rho, expected_S, capsys):
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")
        electronic = Electronic(Job)
        # Spin 0 density matrix
        electronic.rho = np.matrix(rho)

        # Action
        with capsys.disabled():
            S = electronic.quantum_number_S(Job)

        # Result
        assert S == expected_S
