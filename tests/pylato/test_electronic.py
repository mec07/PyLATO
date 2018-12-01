import json
import numpy as np

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
        electron = Electronic(Job)

        # Result
        assert np.array_equal(electron.rho, input_rho)
        assert np.array_equal(electron.rhotot, input_rho)

    def test_init_incorrect_input_density_matrix_dimensions(self):
        # Setup
        Job = InitJob("test_data/JobDef_input_density_matrix.json")
        bad_rho_file = "test_data/bad_rho.json"
        Job.Def['input_rho'] = bad_rho_file

        expected_rho = np.matrix(np.zeros(
            (Job.Hamilton.HSOsize, Job.Hamilton.HSOsize), dtype='complex'))

        # Action
        electron = Electronic(Job)

        # Result
        assert np.array_equal(electron.rho, expected_rho)
        assert np.array_equal(electron.rhotot, expected_rho)
