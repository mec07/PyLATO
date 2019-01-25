import numpy as np
import os
import shutil

from unittest.mock import patch

from pylato.main import main
from pylato.exceptions import ChemicalPotentialError, SelfConsistencyError
from scripts.utils import (
    BackupFiles, InputDensity, JobDef, Model, save_1D_raw_data,
    save_1D_with_extra_info_raw_data, save_2D_with_extra_info_raw_data
)
from scripts.generate_classification_data import get_classification

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
        jobdef.write_magnetic_correlation()
        jobdef.update_hamiltonian("scase")
        jobdef.write_groundstate_classification()
        jobdef.update_model("TBcanonical_s")
        results_dir = jobdef['results_dir']
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        classification_file = os.path.join(results_dir, "classification.txt")
        mag_corr_array_filename = os.path.join(results_dir, "mag_mom_corr_with_class_scase_local_minimum.csv")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=100)
        mag_corr_array = []
        classification_array = []
        for U in U_array:
            mag_corr, classification = calculate_mag_corr_result(
                U, 0, 0, model, mag_corr_file, execution_args,
                classification_file=classification_file)
            mag_corr_array.append(mag_corr)
            classification_array.append(classification)

        x_label = "U/|t|"
        y_label = "C_avg"
        extra_info_label = "classification"
        save_1D_with_extra_info_raw_data(
            x_vals=U_array, results=mag_corr_array,
            extra_info=classification_array,
            labels=[x_label, y_label, extra_info_label],
            filename=mag_corr_array_filename)


def generate_mag_mom_corr_scase_global_minimum():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_s.json"
    model = Model(modelfile)
    input_density_file = "scripts/rho.json"
    input_density = InputDensity(input_density_file)

    with BackupFiles(input_density_file, jobdef_file, modelfile):
        jobdef.write_magnetic_correlation()
        jobdef.write_groundstate_classification()
        jobdef.update_hamiltonian("scase")
        jobdef.update_model("TBcanonical_s")
        jobdef.update_input_rho(input_density_file)

        results_dir = jobdef['results_dir']
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        classification_file = os.path.join(results_dir, "classification.txt")
        mag_corr_array_filename = os.path.join(results_dir, "mag_mom_corr_with_class_scase_global_minimum.csv")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=100)
        mag_corr_array = []
        classification_array = []
        for U in U_array:
            input_density.update_U(U)

            mag_corr, classification = calculate_mag_corr_result(
                U, 0, 0, model, mag_corr_file, execution_args,
                classification_file=classification_file)
            mag_corr_array.append(mag_corr)
            classification_array.append(classification)

        x_label = "U/|t|"
        y_label = "C_avg"
        extra_info_label = "classification"
        save_1D_with_extra_info_raw_data(
            x_vals=U_array, results=mag_corr_array,
            extra_info=classification_array,
            labels=[x_label, y_label, extra_info_label],
            filename=mag_corr_array_filename)


def generate_mag_mom_corr_pcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_p.json"
    model = Model(modelfile)

    electrons_of_interest = [2]
    with BackupFiles(jobdef_file, modelfile):
        jobdef.write_magnetic_correlation()
        jobdef.write_groundstate_classification()
        jobdef.update_hamiltonian("pcase")
        jobdef.update_model("TBcanonical_p")

        results_dir = jobdef['results_dir']
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        classification_file = os.path.join(results_dir, "classification.txt")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=50)
        J_array = np.linspace(0.005, 2.5, num=50)
        for num_electrons in electrons_of_interest:
            model.update_num_electrons(num_electrons)

            mag_corr_result_filename = os.path.join(
                results_dir,
                "mag_mom_corr_and_class_pcase_{}_electrons_per_atom.csv".format(
                    num_electrons))
            mag_corr_result = {}
            classification_result = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    key = (U_index, J_index)
                    if U >= J:
                        mag_corr_result[key], classification_result[key] = (
                            calculate_mag_corr_result(
                                U, J, 0, model, mag_corr_file, execution_args,
                                classification_file=classification_file)
                        )
                    else:
                        mag_corr_result[key] = None
                        classification_result[key] = None

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "C_avg"
            extra_info_label = "classification"
            save_2D_with_extra_info_raw_data(
                x_vals=U_array, y_vals=J_array, results=mag_corr_result,
                extra_info=classification_result,
                labels=[x_label, y_label, values_label, extra_info_label],
                filename=mag_corr_result_filename
            )


def generate_mag_mom_corr_dcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_d.json"
    model = Model(modelfile)

    electrons_of_interest = [6]
    with BackupFiles(jobdef_file, modelfile):
        jobdef.write_magnetic_correlation()
        jobdef.write_groundstate_classification()
        jobdef.update_hamiltonian("dcase")
        jobdef.update_model("TBcanonical_d")

        results_dir = jobdef['results_dir']
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        filename = "mag_mom_corr_and_class_dcase_{}_electrons_per_atom_dJ_{}.csv"
        execution_args = ['pylato/main.py', jobdef_file]

        dJ_val1 = 0.0
        dJ_val2 = 0.1
        U_array = np.linspace(0.005, 10, num=20)
        J_array = np.linspace(0.005, 2.5, num=20)
        for num_electrons in electrons_of_interest:
            model.update_num_electrons(num_electrons)

            mag_corr_result_filename_1 = os.path.join(
                results_dir, filename.format(num_electrons, dJ_val1))
            mag_corr_result_filename_2 = os.path.join(
                results_dir, filename.format(num_electrons, dJ_val2))
            classification_file = os.path.join(results_dir, "classification.txt")

            mag_corr_result_1 = {}
            mag_corr_result_2 = {}
            classification_result_1 = {}
            classification_result_2 = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    key = (U_index, J_index)
                    if U >= J:
                        mag_corr_result_1[key], classification_result_1[key] = (
                            calculate_mag_corr_result(
                                U, J, dJ_val1, model, mag_corr_file,
                                execution_args,
                                classification_file=classification_file)
                        )
                        mag_corr_result_2[key], classification_result_2[key] = (
                            calculate_mag_corr_result(
                                U, J, dJ_val2, model, mag_corr_file,
                                execution_args,
                                classification_file=classification_file)
                        )
                    else:
                        mag_corr_result_1[key] = None
                        classification_result_1[key] = None
                        mag_corr_result_2[key] = None
                        classification_result_2[key] = None

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "C_avg"
            extra_info_label = "classification"
            save_2D_with_extra_info_raw_data(
                x_vals=U_array, y_vals=J_array, results=mag_corr_result_1,
                extra_info=classification_result_1,
                labels=[x_label, y_label, values_label, extra_info_label],
                filename=mag_corr_result_filename_1)
            save_2D_with_extra_info_raw_data(
                x_vals=U_array, y_vals=J_array, results=mag_corr_result_2,
                extra_info=classification_result_2,
                labels=[x_label, y_label, values_label, extra_info_label],
                filename=mag_corr_result_filename_2)


def generate_mag_mom_corr_vector_stoner_pcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_p.json"
    model = Model(modelfile)
    results_file = "mag_mom_corr_and_class_vector_stoner_pcase_{}_electrons_per_atom.csv"

    electrons_of_interest = [2]
    with BackupFiles(jobdef_file, modelfile):
        jobdef.write_magnetic_correlation()
        jobdef.write_groundstate_classification()
        jobdef.update_hamiltonian("vector_stoner")
        jobdef.update_model("TBcanonical_p")

        results_dir = jobdef['results_dir']
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        classification_file = os.path.join(results_dir, "classification.txt")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=50)
        J_array = np.linspace(0.005, 2.5, num=50)
        for num_electrons in electrons_of_interest:
            model.update_num_electrons(num_electrons)

            mag_corr_result_filename = os.path.join(
                results_dir, results_file.format(num_electrons)
            )
            mag_corr_result = {}
            classification_result = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    key = (U_index, J_index)
                    if U >= J:
                        mag_corr_result[key], classification_result[key] = (
                            calculate_mag_corr_result(
                                U, J, 0, model, mag_corr_file, execution_args,
                                classification_file=classification_file
                            )
                        )
                    else:
                        mag_corr_result[key] = None
                        classification_result[key] = None

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "C_avg"
            extra_info_label = "classification"
            save_2D_with_extra_info_raw_data(
                x_vals=U_array, y_vals=J_array, results=mag_corr_result,
                extra_info=classification_result,
                labels=[x_label, y_label, values_label, extra_info_label],
                filename=mag_corr_result_filename
            )


def generate_mag_mom_corr_vector_stoner_dcase():
    jobdef_file = "scripts/JobDef.json"
    jobdef = JobDef(jobdef_file)
    modelfile = "models/TBcanonical_d.json"
    model = Model(modelfile)
    filename = "mag_mom_corr_and_class_vector_stoner_dcase_{}_electrons_per_atom.csv"

    electrons_of_interest = [6]
    with BackupFiles(jobdef_file, modelfile):
        jobdef.write_magnetic_correlation()
        jobdef.write_groundstate_classification()
        jobdef.update_hamiltonian("vector_stoner")
        jobdef.update_model("TBcanonical_d")

        results_dir = jobdef['results_dir']
        mag_corr_file = os.path.join(results_dir, "mag_corr.txt")
        classification_file = os.path.join(results_dir, "classification.txt")
        execution_args = ['pylato/main.py', jobdef_file]

        U_array = np.linspace(0.005, 10, num=20)
        J_array = np.linspace(0.005, 2.5, num=20)
        for num_electrons in electrons_of_interest:
            model.update_num_electrons(num_electrons)

            mag_corr_result_filename = os.path.join(
                results_dir, filename.format(num_electrons))
            mag_corr_result = {}
            classification_result = {}
            for U_index, U in enumerate(U_array):
                for J_index, J in enumerate(J_array):
                    key = (U_index, J_index)
                    if U >= J:
                        mag_corr_result[key], classification_result[key] = (
                            calculate_mag_corr_result(
                                U, J, 0, model, mag_corr_file,
                                execution_args,
                                classification_file=classification_file)
                        )
                    else:
                        mag_corr_result[key] = None
                        classification_result[key] = None

            x_label = "U/|t|"
            y_label = "J/|t|"
            values_label = "C_avg"
            extra_info_label = "classification"
            save_2D_with_extra_info_raw_data(
                x_vals=U_array, y_vals=J_array, results=mag_corr_result,
                extra_info=classification_result,
                labels=[x_label, y_label, values_label, extra_info_label],
                filename=mag_corr_result_filename)


def calculate_mag_corr_result(U, J, dJ, model, mag_corr_file, execution_args,
                              classification_file=None):
    print("U = ", U)
    print("J = ", J)
    print("dJ = ", dJ)
    model.update_U(U)
    model.update_J(J)
    model.update_dJ(dJ)

    with patch('sys.argv', execution_args):
        try:
            main()
            if classification_file:
                return get_mag_corr(mag_corr_file), get_classification(classification_file)
            else:
                return get_mag_corr(mag_corr_file)
        except np.linalg.linalg.LinAlgError as err:
            print(err)
        except ChemicalPotentialError as err:
            print(err)
        except SelfConsistencyError as err:
            print(err)

        if classification_file:
            return None, None
        return None


def get_mag_corr(filename):
    with open(filename, 'r') as file_handle:
        mag_corr = file_handle.read()

    return float(mag_corr)


def save_rho_mat(original_name, new_name):
    shutil.copyfile(original_name, new_name)


if __name__ == "__main__":
    generate_mag_mom_corr_scase_local_minimum()
    generate_mag_mom_corr_scase_global_minimum()
    # generate_mag_mom_corr_pcase()
    # generate_mag_mom_corr_dcase()
    # generate_mag_mom_corr_vector_stoner_pcase()
    # generate_mag_mom_corr_vector_stoner_dcase()
