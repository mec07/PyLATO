import commentjson
import numpy as np
import os

from unittest.mock import patch

from pylato.main import main
from scripts.utils import (
    BackupFiles, InputDensity, JobDef, Model, save_1D_raw_data
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
        jobdef.update_hamiltonian("scase")

        results_dir = jobdef['results_dir']
        energy_file = os.path.join(results_dir, "energy.txt")
        energy_array_filename = os.path.join(results_dir, "total_energy_array_local_minimum.csv")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=100)
        energy_array = []
        for U in U_array:
            energy_array.append(calculate_energy_result(
                U, model, energy_file, execution_args))

    x_label = "U/|t|"
    y_label = "energy/eV"
    save_1D_raw_data(U_array, energy_array, x_label, y_label, energy_array_filename)
    print("Saved {} by {} to {}".format(y_label, x_label, energy_array_filename))


def generate_energy_data_global_minimum():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_s.json"
    model = Model(modelfile)
    input_density_file = "scripts/rho.json"
    input_density = InputDensity(input_density_file)

    with BackupFiles(input_density_file, jobdef_file, modelfile):
        jobdef.update_hamiltonian("scase")
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
                U, model, energy_file, execution_args))

    x_label = "U/|t|"
    y_label = "energy/eV"
    save_1D_raw_data(U_array, energy_array, x_label, y_label, energy_array_filename)
    print("Saved {} by {} to {}".format(y_label, x_label, energy_array_filename))


def calculate_energy_result(U, model, energy_file, execution_args):
    print("U = ", U)
    model.update_U(U)

    with patch('sys.argv', execution_args):
        try:
            assert main()
            return get_energy(energy_file)
        except np.linalg.linalg.LinAlgError:
            return None

def get_energy(energy_file):
    with open(energy_file, 'r') as file_handle:
        energy = file_handle.read()

    return float(energy)


if __name__ == "__main__":
    generate_energy_data_local_minimum()
    generate_energy_data_global_minimum()
