import commentjson
import csv
import math
import os
import shutil


class BackupFiles(object):
    """
    Backup the files that are going to be changed and then restore the backup
    at the end. To do this we use the context manager pattern, i.e.
        with BackupFiles(file1, file2,...):
            ...code that manipulates files...
    Then you do what you want with the files and you know that they will get
    restored to how they originally were.
    """
    def __init__(self, *filenames):
        self.originals = list(filenames)
        self.backups = []

    def __enter__(self):
        # Setup -- create the backup files:
        for filename in self.originals:
            backup_name = ".{}.temp".format(os.path.basename(filename))
            backup_file = os.path.join(os.path.dirname(filename), backup_name)
            self.backups.append(backup_file)
            shutil.copyfile(filename, backup_file)

    def __exit__(self, type, value, traceback):
        # Cleanup -- restore the backups:
        for index, filename in enumerate(self.originals):
            shutil.copyfile(self.backups[index], filename)
            os.remove(self.backups[index])


class CommentJsonInteractor(object):
    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'r') as file_handle:
            data = commentjson.loads(file_handle.read())
        self.data = data

    def update_file(self):
        with open(self.filename, 'w') as file_handle:
            commentjson.dump(self.data, file_handle)

    def __getitem__(self, key):
        return self.data[key]


class InputDensity(CommentJsonInteractor):
    def update_U(self, U):
        for ii in range(len(self.data)):
            for jj in range(len(self.data)):
                self.data[ii][jj] = self.calculate_element(ii, jj, U)
        self.update_file()

    def calculate_element(self, ii, jj, U):
        """
        Note that this has only been implemented for the specific case of the
        scase dimer!
        """
        is_simple_rho = False
        inside_sqrt = 1.0 - 4.0/(U*U)
        if (inside_sqrt) < 0:
            is_simple_rho = True
        # diagonals:
        if ii == jj:
            if is_simple_rho:
                return 0.5
            # first type of diagonal
            if ii == 0 or ii == 3:
                return (1.0 + math.sqrt(inside_sqrt))*0.5
            # other type of diagonal
            else:
                return (1.0 - math.sqrt(inside_sqrt))*0.5

        # off-diagonals:
        off_diagonals = [(0, 1), (1, 0), (2, 3), (3, 2)]
        if (ii, jj) in off_diagonals:
            if is_simple_rho:
                return 0.5
            return 1.0/U

        # any other element
        return 0


class JobDef(CommentJsonInteractor):
    def update_hamiltonian(self, hami):
        self.data['Hamiltonian'] = hami
        self.update_file()

    def update_input_rho(self, input_rho_file):
        self.data['input_rho'] = input_rho_file
        self.update_file()


class Model(CommentJsonInteractor):
    def update_U(self, U):
        self.data["species"][0]["U"] = U
        self.update_file()

    def update_J(self, J):
        self.data["species"][0]["I"] = J
        self.update_file()

    def update_dJ(self, dJ):
        self.data["species"][0]["dJ"] = dJ
        self.update_file()


def save_1D_raw_data(x_vals, y_vals, x_col_name, y_col_name, filename):
    assert len(x_vals) == len(y_vals)

    with open(filename, 'w') as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=[x_col_name, y_col_name])
        writer.writeheader()
        print("{},{}".format(x_col_name, y_col_name))

        for index, val in enumerate(x_vals):
            writer.writerow({x_col_name: val, y_col_name: y_vals[index]})
            print("{},{}".format(val, y_vals[index]))


def save_2D_raw_data(x_vals, y_vals, results, x_col_name, y_col_name, values_col_name, filename):
    assert len(x_vals) == len(y_vals)
    assert len(x_vals)*len(y_vals) == len(results)

    fieldnames = [x_col_name, y_col_name, values_col_name]
    with open(filename, 'w') as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        print(fieldnames)

        for x_index, x_val in enumerate(x_vals):
            for y_index, y_val in enumerate(x_vals):
                writer.writerow({
                    x_col_name: x_val,
                    y_col_name: y_val,
                    values_col_name: results[(x_index, y_index)],
                })
                print("{},{},{}".format(
                    x_val, y_val, results[(x_index, y_index)]))