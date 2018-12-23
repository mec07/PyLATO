import numpy as np
import os
import shutil

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


def generate_mag_mom_corr_scase_local_minimum():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_s.json"
    model = Model(modelfile)

    with BackupFiles(jobdef_file, modelfile):
        jobdef.update_hamiltonian("scase")
        jobdef.update_model("TBcanonical_s")
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
        jobdef.update_model("TBcanonical_s")
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
        for num_electrons in range(1, 6):
            jobdef.update_hamiltonian("pcase")
            jobdef.update_model("TBcanonical_p")
            model.update_num_electrons(num_electrons)

            results_dir = jobdef['results_dir']
            mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
            mag_corr_result_filename = os.path.join(
                results_dir,
                "mag_mom_corr_pcase_{}_electrons_per_atom.csv".format(
                    num_electrons))
            execution_args = ['pylato/main.py', jobdef_file]

            U_array = np.linspace(0.005, 10, num=20)
            J_array = np.linspace(0.005, 2.5, num=20)
            mag_corr_result = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    if U >= J:
                        mag_corr_result[(U_index, J_index)] = calculate_mag_corr_result(
                            U, J, 0, model, mag_corr_file, execution_args)
                    else:
                        mag_corr_result[(U_index, J_index)] = None

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "C_avg"
            save_2D_raw_data(U_array, J_array, mag_corr_result, x_label, y_label, values_label, mag_corr_result_filename)


def generate_mag_mom_corr_dcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_d.json"
    model = Model(modelfile)

    electrons_of_interest = [6]
    with BackupFiles(jobdef_file, modelfile):
        for num_electrons in electrons_of_interest:
            jobdef.update_hamiltonian("dcase")
            jobdef.update_model("TBcanonical_d")
            model.update_num_electrons(num_electrons)

            results_dir = jobdef['results_dir']
            mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
            dJ_val1 = 0.0
            dJ_val2 = 0.1
            filename = "mag_mom_corr_dcase_{}_electrons_per_atom_dJ_{}.csv"
            mag_corr_result_filename_1 = os.path.join(
                results_dir, filename.format(num_electrons, dJ_val1))
            mag_corr_result_filename_2 = os.path.join(
                results_dir, filename.format(num_electrons, dJ_val2))
            orig_rho_mat = os.path.join(results_dir, "rhoMatrix.txt")
            rho_mat_dir = os.path.join(results_dir, "rho")
            os.makedirs(rho_mat_dir, exist_ok=True)
            rho_mat_basename = "rho_mat_dcase_U_{}_J_{}_dJ_{}.txt"

            execution_args = ['pylato/main.py', jobdef_file]

            U_array = np.linspace(0.005, 10, num=20)
            J_array = np.linspace(0.005, 2.5, num=20)
            mag_corr_result_1 = {}
            mag_corr_result_2 = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    if U >= J:
                        mag_corr_result_1[(U_index, J_index)] = calculate_mag_corr_result(
                            U, J, dJ_val1, model, mag_corr_file, execution_args)
                        mag_corr_result_2[(U_index, J_index)] = calculate_mag_corr_result(
                            U, J, dJ_val2, model, mag_corr_file, execution_args)
                        if mag_corr_result_1[(U_index, J_index)] is not None:
                            save_rho_mat(orig_rho_mat, os.path.join(
                                rho_mat_dir,
                                rho_mat_basename.format(U, J, dJ_val1)
                            ))
                        if mag_corr_result_2[(U_index, J_index)] is not None:
                            save_rho_mat(orig_rho_mat, os.path.join(
                                rho_mat_dir,
                                rho_mat_basename.format(U, J, dJ_val2)
                            ))

                    else:
                        mag_corr_result_1[(U_index, J_index)] = None
                        mag_corr_result_2[(U_index, J_index)] = None

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "C_avg"
            save_2D_raw_data(U_array, J_array, mag_corr_result_1, x_label, y_label, values_label, mag_corr_result_filename_1)
            save_2D_raw_data(U_array, J_array, mag_corr_result_2, x_label, y_label, values_label, mag_corr_result_filename_2)


def generate_mag_mom_corr_vector_stoner_pcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_p.json"
    model = Model(modelfile)
    results_file = "mag_mom_corr_vector_stoner_pcase_{}_electrons_per_atom.csv"

    with BackupFiles(jobdef_file, modelfile):
        for num_electrons in range(1, 6):
            jobdef.update_hamiltonian("vector_stoner")
            jobdef.update_model("TBcanonical_p")
            model.update_num_electrons(num_electrons)

            results_dir = jobdef['results_dir']
            mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
            mag_corr_result_filename = os.path.join(
                results_dir, results_file.format(num_electrons)
            )
            execution_args = ['pylato/main.py', jobdef_file]

            U_array = np.linspace(0.005, 10, num=20)
            J_array = np.linspace(0.005, 2.5, num=20)
            mag_corr_result = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    if U >= J:
                        mag_corr_result[(U_index, J_index)] = calculate_mag_corr_result(
                            U, J, 0, model, mag_corr_file, execution_args)
                    else:
                        mag_corr_result[(U_index, J_index)] = None

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "C_avg"
            save_2D_raw_data(U_array, J_array, mag_corr_result, x_label, y_label, values_label, mag_corr_result_filename)


def generate_mag_mom_corr_vector_stoner_dcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_d.json"
    model = Model(modelfile)
    filename = "mag_mom_corr_vector_stoner_dcase_{}_electrons_per_atom.csv"

    electrons_of_interest = [4, 6]
    with BackupFiles(jobdef_file, modelfile):
        for num_electrons in electrons_of_interest:
            jobdef.update_hamiltonian("vector_stoner")
            jobdef.update_model("TBcanonical_d")
            model.update_num_electrons(num_electrons)

            results_dir = jobdef['results_dir']
            mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
            mag_corr_result_filename = os.path.join(
                results_dir, filename.format(num_electrons))
            orig_rho_mat = os.path.join(results_dir, "rhoMatrix.txt")
            rho_mat_dir = os.path.join(results_dir, "rho")
            os.makedirs(rho_mat_dir, exist_ok=True)
            rho_mat_basename = "rho_mat_vector_stoner_dcase_U_{}_J_{}.txt"

            execution_args = ['pylato/main.py', jobdef_file]

            U_array = np.linspace(0.005, 10, num=20)
            J_array = np.linspace(0.005, 2.5, num=20)
            mag_corr_result = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    if U >= J:
                        mag_corr_result[(U_index, J_index)] = calculate_mag_corr_result(
                            U, J, 0, model, mag_corr_file, execution_args)
                        if mag_corr_result[(U_index, J_index)] is not None:
                            save_rho_mat(orig_rho_mat, os.path.join(
                                rho_mat_dir, rho_mat_basename.format(U, J)))
                    else:
                        mag_corr_result[(U_index, J_index)] = None

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "C_avg"
            save_2D_raw_data(U_array, J_array, mag_corr_result, x_label, y_label, values_label, mag_corr_result_filename)


def calculate_mag_corr_result(U, J, dJ, model, mag_corr_file, execution_args):
    print("U = ", U)
    print("J = ", J)
    print("dJ = ", dJ)
    model.update_U(U)
    model.update_J(J)
    model.update_dJ(dJ)

    with patch('sys.argv', execution_args):
        try:
            main()
            return get_mag_corr(mag_corr_file)
        except np.linalg.linalg.LinAlgError:
            return None
        except ChemicalPotentialError:
            return None
        except SelfConsistencyError:
            return None


def get_mag_corr(filename):
    with open(filename, 'r') as file_handle:
        mag_corr = file_handle.read()

    return float(mag_corr)


def save_rho_mat(original_name, new_name):
    shutil.copyfile(original_name, new_name)


if __name__ == "__main__":
    # generate_mag_mom_corr_scase_local_minimum()
    # generate_mag_mom_corr_scase_global_minimum()
    # generate_mag_mom_corr_pcase()
    generate_mag_mom_corr_dcase()
    # generate_mag_mom_corr_vector_stoner_pcase()
    generate_mag_mom_corr_vector_stoner_dcase()
