import json
import numpy as np
import pytest

from pylato.electronic import Electronic, num_swaps_to_sort
from pylato.exceptions import UnimplementedMethodError
from pylato.init_job import InitJob
from pylato.main import execute_job

from tests.conftest import load_json_file


@pytest.mark.parametrize(
    ("array", "expected_num_swaps"),
    [
        ([1, 2, 3, 4, 5], 0),
        ([2, 1, 3, 4, 5], 1),
        ([2, 3, 1, 4, 5], 2),
        ([2, 3, 4, 1, 5], 3),
        ([2, 4, 3, 1, 5], 4),
        ([4, 2, 3, 1, 5], 5),
        ([4, 2, 3, 5, 1], 6),
        ([4, 2, 5, 3, 1], 7),
        ([4, 5, 2, 3, 1], 8),
        ([5, 4, 2, 3, 1], 9),
    ]
)
def test_num_swaps_to_sort(array, expected_num_swaps):
    assert num_swaps_to_sort(array) == expected_num_swaps


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
        # Spin 0 density matrix
        Job.Electron.rho = np.matrix(rho)

        # Action
        S = Job.Electron.quantum_number_S(Job)

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
        Job.Electron.NElectrons = sum(rho[ii][ii] for ii in range(len(rho)))
        Job.Electron.rho = np.matrix(rho, dtype='complex')

        # Action
        L_z = Job.Electron.quantum_number_L_z(Job)

        # Result
        assert L_z == expected_L_z

    @pytest.mark.parametrize(
        ("job_file", "rho_file", "expected_L_z"),
        [
            ("test_data/JobDef_scase.json", "test_data/rho_scase.json", 0),
            ("test_data/JobDef_pcase.json", "test_data/rho_pcase_2.json", 1),
            ("test_data/JobDef_dcase.json", "test_data/rho_dcase.json", 0),
            # Not too sure about this last test case, rho_dcase_2 was
            # arbitrarily constructed to get this result...
            ("test_data/JobDef_dcase.json", "test_data/rho_dcase_2.json", 1),
        ]
    )
    def test_quantum_number_L_z_dimers(self, job_file, rho_file, expected_L_z):
        # Setup
        Job = InitJob(job_file)

        # Fake
        rho = load_json_file(rho_file)
        Job.Electron.rho = np.matrix(rho, dtype='complex')

        # Action
        L_z = Job.Electron.quantum_number_L_z(Job)

        # Result
        assert L_z == expected_L_z

    def test_quantum_number_L_z_not_implemented_error(self):
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")
        Job.NOrb = [1, 3]
        expected_message = (
            "Quantum Number L_z methods have only been implemented for "
            "simulations consisting of solely s, p or d orbital atoms"
        )

        # Action
        with pytest.raises(UnimplementedMethodError, message=expected_message):
            print(Job.Electron.quantum_number_L_z(Job))

    def test_quantum_number_L_z_1_electron(self):
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")
        Job.Electron.NElectrons = 1

        # Action
        L_z = Job.Electron.quantum_number_L_z(Job)

        # Result
        assert L_z == 0

    @pytest.mark.parametrize(
        ("norb", "expected_result"),
        [
            ([1, 1, 1, 1], True),
            ([2, 2, 2, 2, 2, 2, 2], True),
            ([1, 2, 1], False),
            ([4, 3, 3, 3, 3, 3, 3, 3, 3, 3], False),
        ]
    )
    def test_all_atoms_same_num_orbitals(self, norb, expected_result):
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")

        # Fake
        Job.NOrb = norb

        # Action
        result = Job.Electron.all_atoms_same_num_orbitals(Job)

        # Result
        assert result is expected_result

    @pytest.mark.parametrize(
        ("job_file", "old_eigenvectors", "expected_eigenvectors"),
        [
            (
                "test_data/JobDef_scase.json", [
                    np.array([0.5, 0.7, 0.3, 0.2]),
                    np.array([0.1, 0.6, 0.8, 0.9]),
                ], [
                    np.array([0.7, 0.5, 0.2, 0.3]),
                    np.array([0.6, 0.1, 0.9, 0.8]),
                ]
            ),
            (
                "test_data/JobDef_pcase.json", [
                    np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                    np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
                ], [
                    np.array([-0.4, -0.5, -0.6, -0.1, -0.2, -0.3, -1.0, -1.1, -1.2, -0.7, -0.8, -0.9]),
                    np.array([-0.9, -0.8, -0.7, -1.2, -1.1, -1.0, -0.3, -0.2, -0.1, -0.6, -0.5, -0.4]),
                ]
            ),
        ]
    )
    def test_perform_inversion(self, job_file, old_eigenvectors,
                               expected_eigenvectors):
        # Setup
        Job = InitJob(job_file)

        # Action
        new_eigenvectors = Job.Electron.perform_inversion(Job, old_eigenvectors)

        # Result
        assert all(np.array_equal(new_eigenvectors[i], expected_eigenvectors[i])
                   for i in range(len(expected_eigenvectors)))

    @pytest.mark.parametrize(
        ("name", "new_eigenvectors", "old_eigenvectors", "expected_result"),
        [
            (
                "swap",
                [
                    np.array([0.1, 0.2, 0.3, 0.4]),
                    np.array([0.2, 0.1, 0.4, 0.3]),
                ],
                [
                    np.array([0.2, 0.1, 0.4, 0.3]),
                    np.array([0.1, 0.2, 0.3, 0.4]),
                ],
                -1
            ),
            (
                "double swap",
                [
                    np.array([0.1, 0.2, 0.3, 0.4]),
                    np.array([0.2, 0.1, 0.4, 0.3]),
                    np.array([0.5, 0.6, 0.7, 0.8]),
                    np.array([0.6, 0.5, 0.8, 0.7]),
                ],
                [
                    np.array([0.2, 0.1, 0.4, 0.3]),
                    np.array([0.1, 0.2, 0.3, 0.4]),
                    np.array([0.6, 0.5, 0.8, 0.7]),
                    np.array([0.5, 0.6, 0.7, 0.8]),
                ],
                1
            ),
            (
                "separated double swap",
                [
                    np.array([0.1, 0.2, 0.3, 0.4]),
                    np.array([0.5, 0.6, 0.7, 0.8]),
                    np.array([0.2, 0.1, 0.4, 0.3]),
                    np.array([0.6, 0.5, 0.8, 0.7]),
                ],
                [
                    np.array([0.2, 0.1, 0.4, 0.3]),
                    np.array([0.6, 0.5, 0.8, 0.7]),
                    np.array([0.1, 0.2, 0.3, 0.4]),
                    np.array([0.5, 0.6, 0.7, 0.8]),
                ],
                1
            ),
            (
                "minus signs",
                [
                    np.array([-0.1, -0.2, -0.3, -0.4]),
                    np.array([0.2, 0.1, 0.4, 0.3]),
                ],
                [
                    np.array([0.1, 0.2, 0.3, 0.4]),
                    np.array([0.2, 0.1, 0.4, 0.3]),
                ],
                -1
            ),
            (
                "swap with minus signs",
                [
                    np.array([0.1, -0.2, 0.3, -0.4]),
                    np.array([0.2, -0.1, 0.4, -0.3]),
                ],
                [
                    np.array([-0.2, 0.1, -0.4, 0.3]),
                    np.array([-0.1, 0.2, -0.3, 0.4]),
                ],
                -1
            ),
            (
                "separated double swap with minus signs",
                [
                    np.array([0.1, -0.2, 0.3, -0.4]),
                    np.array([0.5, 0.6, 0.7, 0.8]),
                    np.array([0.2, -0.1, 0.4, -0.3]),
                    np.array([0.6, 0.5, 0.8, 0.7]),
                ],
                [
                    np.array([-0.2, 0.1, -0.4, 0.3]),
                    np.array([0.6, 0.5, 0.8, 0.7]),
                    np.array([-0.1, 0.2, -0.3, 0.4]),
                    np.array([0.5, 0.6, 0.7, 0.8]),
                ],
                1
            ),
        ]
    )
    def test_symmetry_operation_result(self, name, new_eigenvectors,
                                       old_eigenvectors, expected_result):
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")

        # Action
        result = Job.Electron.symmetry_operation_result(Job, new_eigenvectors,
                                                        old_eigenvectors)

        # Result
        assert result == expected_result

    def test_gerade_not_dimer_error(self):
        # Setup
        Job = InitJob("test_data/JobDef_scase.json")

        # Fake
        Job.NAtom = 1

        # Action
        with pytest.raises(UnimplementedMethodError):
            Job.Electron.gerade(Job)

    @pytest.mark.parametrize(
        ("job_file", "expected_gerade"),
        [
            ("test_data/JobDef_scase.json", 'g'),
            ("test_data/JobDef_pcase.json", 'g'),
            ("test_data/JobDef_dcase.json", 'g'),
        ]
    )
    def test_gerade(self, job_file, expected_gerade):
        # Setup
        Job = InitJob(job_file)
        Job = execute_job(Job)

        # Action
        gerade = Job.Electron.gerade(Job)

        # Result
        assert gerade == expected_gerade

    @pytest.mark.parametrize(
        ("job_file", "orbital", "initial_value", "expected_reflected_value"),
        [
            ("test_data/JobDef_scase.json", 0, 0.1, 0.1),
            ("test_data/JobDef_pcase.json", 0, 0.2, 0.2),
            ("test_data/JobDef_pcase.json", 1, 0.3, -0.3),
            ("test_data/JobDef_pcase.json", 2, 0.4, 0.4),
            ("test_data/JobDef_dcase.json", 0, 0.5, 0.5),
            ("test_data/JobDef_dcase.json", 1, 0.6, 0.6),
            ("test_data/JobDef_dcase.json", 2, 0.7, -0.7),
            ("test_data/JobDef_dcase.json", 3, 0.8, -0.8),
            ("test_data/JobDef_dcase.json", 4, 0.9, 0.9),
        ]
    )
    def test_get_reflected_value(self, job_file, orbital, initial_value,
                                 expected_reflected_value):
        # Setup
        Job = InitJob(job_file)
        atom = 0

        # Action
        reflected_value = Job.Electron.get_reflected_value(Job, initial_value,
                                                           atom, orbital)

        # Result
        assert reflected_value == expected_reflected_value

    @pytest.mark.parametrize(
        ("job_file", "old_eigenvectors", "expected_eigenvectors"),
        [
            (
                "test_data/JobDef_scase.json", [
                    np.array([0.5, 0.7, 0.3, 0.2]),
                    np.array([0.1, 0.6, 0.8, 0.9]),
                ], [
                    np.array([0.5, 0.7, 0.3, 0.2]),
                    np.array([0.1, 0.6, 0.8, 0.9]),
                ]
            ),
            (
                "test_data/JobDef_pcase.json", [
                    np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]),
                    np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
                ], [
                    np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2]),
                    np.array([1.2, -1.1, 1.0, 0.9, -0.8, 0.7, 0.6, -0.5, 0.4, 0.3, -0.2, 0.1]),
                ]
            ),
        ]
    )
    def test_perform_reflection(self, job_file, old_eigenvectors,
                                expected_eigenvectors):
        # Setup
        Job = InitJob(job_file)

        # Action
        new_eigenvectors = Job.Electron.perform_reflection(Job, old_eigenvectors)

        # Result
        assert all(np.array_equal(new_eigenvectors[i], expected_eigenvectors[i])
                   for i in range(len(expected_eigenvectors)))

    @pytest.mark.parametrize(
        ("job_file", "eigenvectors_file", "expected_plus_minus"),
        [
            ("test_data/JobDef_scase.json", "test_data/eigenvectors_s.json", '+'),
            ("test_data/JobDef_pcase.json", "test_data/eigenvectors_p.json", '+'),
            ("test_data/JobDef_pcase.json", "test_data/eigenvectors_p_2.json", '-'),
            ("test_data/JobDef_dcase.json", "test_data/eigenvectors_d.json", '+'),
        ]
    )
    def test_plus_minus(self, job_file, eigenvectors_file, expected_plus_minus, capsys):
        # Setup
        Job = InitJob(job_file)
        with open(eigenvectors_file, 'r') as fh:
            Job.psi = np.array(json.load(fh))

        # Action
        plus_minus = Job.Electron.plus_minus(Job)

        # Result
        assert plus_minus == expected_plus_minus
