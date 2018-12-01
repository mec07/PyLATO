import commentjson
import numpy as np
import os

from unittest.mock import patch

from pylato.main import main
from scripts.utils import (
    BackupFiles, InputDensity, JobDef, Model, save_1D_raw_data, save_2D_raw_data
)

"""
Run this script from the top level of the repo. If you run this from the
`scripts` folder it will just error.
"""


def generate_mag_mom_corr_scase_local_minimum():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_s.json"
    model = Model(modelfile)

    with BackupFiles(jobdef_file, modelfile):
        jobdef.update_hamiltonian("scase")
        results_dir = jobdef['results_dir']
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        mag_corr_array_filename = os.path.join(results_dir, "mag_mom_corr_scase_local_minimum.csv")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=100)
        mag_corr_array = []
        for U in U_array:
            mag_corr_array.append(calculate_mag_corr_result(
                U, 0, 0, model, mag_corr_file, execution_args))

    x_label = "U/|t|"
    y_label = "C_avg"
    save_1D_raw_data(U_array, mag_corr_array, x_label, y_label, mag_corr_array_filename)


def generate_mag_mom_corr_scase_global_minimum():
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
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        mag_corr_array_filename = os.path.join(results_dir, "mag_mom_corr_scase_global_minimum.csv")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=100)
        mag_corr_array = []
        for U in U_array:
            input_density.update_U(U)
            mag_corr_array.append(calculate_mag_corr_result(
                U, 0, 0, model, mag_corr_file, execution_args))

    x_label = "U/|t|"
    y_label = "C_avg"
    save_1D_raw_data(U_array, mag_corr_array, x_label, y_label, mag_corr_array_filename)


def generate_mag_mom_corr_pcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_p.json"
    model = Model(modelfile)

    with BackupFiles(jobdef_file, modelfile):
        jobdef.update_hamiltonian("pcase")

        results_dir = jobdef['results_dir']
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        mag_corr_result_filename = os.path.join(results_dir, "mag_mom_corr_pcase.csv")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=10)
        J_array = np.linspace(0.005, 5, num=10)
        mag_corr_result = {}
        for U_index, U in enumerate(U_array):
            for J_index, J in enumerate(J_array):
                mag_corr_result[(U_index, J_index)] = calculate_mag_corr_result(
                    U, J, 0, model, mag_corr_file, execution_args)

    x_label = "U/|t|"
    y_label = "J/|t|"
    values_label = "C_avg"
    save_2D_raw_data(U_array, J_array, mag_corr_result, x_label, y_label, values_label, mag_corr_result_filename)


def generate_mag_mom_corr_dcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_d.json"
    model = Model(modelfile)

    with BackupFiles(jobdef_file, modelfile):
        jobdef.update_hamiltonian("dcase")

        results_dir = jobdef['results_dir']
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        dJ_val1 = 0.0
        dJ_val2 = 0.1
        mag_corr_result_filename_1 = os.path.join(
            results_dir, "mag_mom_corr_dcase_dJ_{}.csv".format(dJ_val1))
        mag_corr_result_filename_2 = os.path.join(
            results_dir, "mag_mom_corr_dcase_dJ_{}.csv".format(dJ_val2))
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=5)
        J_array = np.linspace(0.005, 5, num=5)
        mag_corr_result_1 = {}
        mag_corr_result_2 = {}
        for U_index, U in enumerate(U_array):
            for J_index, J in enumerate(J_array):
                mag_corr_result_1[(U_index, J_index)] = calculate_mag_corr_result(
                    U, J, dJ_val1, model, mag_corr_file, execution_args)
                mag_corr_result_2[(U_index, J_index)] = calculate_mag_corr_result(
                    U, J, dJ_val2, model, mag_corr_file, execution_args)

    x_label = "U/|t|"
    y_label = "J/|t|"
    values_label = "C_avg"
    save_2D_raw_data(U_array, J_array, mag_corr_result_1, x_label, y_label, values_label, mag_corr_result_filename_1)
    save_2D_raw_data(U_array, J_array, mag_corr_result_2, x_label, y_label, values_label, mag_corr_result_filename_2)


def calculate_mag_corr_result(U, J, dJ, model, mag_corr_file, execution_args):
    print("U = ", U)
    print("J = ", J)
    print("dJ = ", dJ)
    model.update_U(U)
    model.update_J(J)
    model.update_dJ(dJ)

    with patch('sys.argv', execution_args):
        try:
            assert main()
            return get_mag_corr(mag_corr_file)
        except np.linalg.linalg.LinAlgError:
            return None


def get_mag_corr(filename):
    with open(filename, 'r') as file_handle:
        mag_corr = file_handle.read()

    return float(mag_corr)


if __name__ == "__main__":
    generate_mag_mom_corr_scase_local_minimum()
    generate_mag_mom_corr_scase_global_minimum()
    generate_mag_mom_corr_pcase()
    generate_mag_mom_corr_dcase()
