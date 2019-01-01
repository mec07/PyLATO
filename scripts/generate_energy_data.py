import commentjson
import numpy as np
import os

from unittest.mock import patch

from pylato.main import main
from pylato.exceptions import ChemicalPotentialError, SelfConsistencyError
from scripts.utils import (
    BackupFiles, InputDensity, JobDef, Model, save_1D_raw_data, save_2D_raw_data
)

"""
Run this script from the top level of the repo. If you run this from the
`scripts` folder it will just error.
"""


def generate_energy_data_local_minimum():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_s.json"
    model = Model(modelfile)

    with BackupFiles(jobdef_file, modelfile):
        jobdef.write_total_energy()
        jobdef.update_hamiltonian("scase")
        jobdef.update_model("TBcanonical_s")

        results_dir = jobdef['results_dir']
        energy_file = os.path.join(results_dir, "energy.txt")
        energy_array_filename = os.path.join(results_dir, "total_energy_array_local_minimum.csv")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=100)
        energy_array = []
        for U in U_array:
            energy_array.append(calculate_energy_result(
                U, 0, 0, model, energy_file, execution_args))

        x_label = "U/|t|"
        y_label = "energy/eV"
        save_1D_raw_data(U_array, energy_array, x_label, y_label, energy_array_filename)


def generate_energy_data_global_minimum():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_s.json"
    model = Model(modelfile)
    input_density_file = "scripts/rho.json"
    input_density = InputDensity(input_density_file)

    with BackupFiles(input_density_file, jobdef_file, modelfile):
        jobdef.write_total_energy()
        jobdef.update_hamiltonian("scase")
        jobdef.update_model("TBcanonical_s")
        jobdef.update_input_rho(input_density_file)

        results_dir = jobdef['results_dir']
        energy_file = os.path.join(jobdef['results_dir'], "energy.txt")
        energy_array_filename = os.path.join(results_dir, "total_energy_array_global_minimum.csv")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=100)
        energy_array = []
        for U in U_array:
            input_density.update_U(U)
            energy_array.append(calculate_energy_result(
                U, 0, 0, model, energy_file, execution_args))

        x_label = "U/|t|"
        y_label = "energy/eV"
        save_1D_raw_data(U_array, energy_array, x_label, y_label, energy_array_filename)


def generate_energy_data_pcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_p.json"
    model = Model(modelfile)

    with BackupFiles(jobdef_file, modelfile):
        jobdef.write_total_energy()
        for num_electrons in range(1, 6):
            jobdef.update_hamiltonian("pcase")
            jobdef.update_model("TBcanonical_p")
            model.update_num_electrons(num_electrons)

            results_dir = jobdef['results_dir']
            energy_file = os.path.join(results_dir, "energy.txt")
            energy_array_filename = os.path.join(
                results_dir,
                "total_energy_array_pcase_{}_electrons_per_atom.csv".format(
                    num_electrons)
            )
            execution_args = ['pylato/main.py', jobdef_file]

            U_array = np.linspace(0.005, 10, num=20)
            J_array = np.linspace(0.005, 2.5, num=20)
            energy_result = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    energy_result[(U_index, J_index)] = calculate_energy_result(
                        U, J, 0, model, energy_file, execution_args)

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "energy/eV"
            save_2D_raw_data(U_array, J_array, energy_result, x_label, y_label, values_label, energy_array_filename)


def generate_energy_data_dcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_d.json"
    model = Model(modelfile)

    electrons_of_interest = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    with BackupFiles(jobdef_file, modelfile):
        jobdef.write_total_energy()
        for num_electrons in electrons_of_interest:
            jobdef.update_hamiltonian("dcase")
            jobdef.update_model("TBcanonical_d")
            model.update_num_electrons(num_electrons)

            results_dir = jobdef['results_dir']
            energy_file = os.path.join(results_dir, "energy.txt")
            dJ_val1 = 0.0
            dJ_val2 = 0.1
            filename = "total_energy_array_dcase_{}_electrons_per_atom_dJ_{}.csv"
            energy_array_filename_1 = os.path.join(
                results_dir,
                filename.format(num_electrons, dJ_val1)
            )
            energy_array_filename_2 = os.path.join(
                results_dir,
                filename.format(num_electrons, dJ_val2)
            )
            execution_args = ['pylato/main.py', jobdef_file]

            U_array = np.linspace(0.005, 10, num=5)
            J_array = np.linspace(0.005, 2.5, num=5)
            energy_result_1 = {}
            energy_result_2 = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    if U >= J:
                        energy_result_1[(U_index, J_index)] = calculate_energy_result(
                            U, J, dJ_val1, model, energy_file, execution_args)
                        energy_result_2[(U_index, J_index)] = calculate_energy_result(
                            U, J, dJ_val2, model, energy_file, execution_args)
                    else:
                        energy_result_1[(U_index, J_index)] = None
                        energy_result_2[(U_index, J_index)] = None

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "energy/eV"
            save_2D_raw_data(U_array, J_array, energy_result_1, x_label, y_label, values_label, energy_array_filename_1)
            save_2D_raw_data(U_array, J_array, energy_result_2, x_label, y_label, values_label, energy_array_filename_2)


def calculate_energy_result(U, J, dJ, model, energy_file, execution_args):
    print("U = ", U)
    print("J = ", J)
    print("dJ = ", dJ)
    model.update_U(U)
    model.update_J(J)
    model.update_dJ(dJ)

    with patch('sys.argv', execution_args):
        try:
            main()
            return get_energy(energy_file)
        except np.linalg.linalg.LinAlgError:
            return None
        except ChemicalPotentialError:
            return None
        except SelfConsistencyError:
            return None


def get_energy(energy_file):
    with open(energy_file, 'r') as file_handle:
        energy = file_handle.read()

    return float(energy)


if __name__ == "__main__":
    generate_energy_data_local_minimum()
    #generate_energy_data_global_minimum()
    #generate_energy_data_pcase()
    #generate_energy_data_dcase()
