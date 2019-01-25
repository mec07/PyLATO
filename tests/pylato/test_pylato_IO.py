import os
import pytest

from functools import partial

from pylato.exceptions import UnimplementedMethodError
from pylato.pylato_IO import (
    WriteTotalEnergy, WriteQuantumNumberS, WriteQuantumNumberLz, classify_groundstate,
    get_spin_part_of_symbol, get_angular_part_of_symbol, get_gerade_part_of_symbol
)


class FakeHamilton(object):
    def total_energy(*args):
        return 0


class FakeElectron(object):
    def __init__(self, S=1, L_z=1, gerade='g', plus_minus='+'):
        self.S = S
        self.L_z = L_z
        self.g_or_u = gerade
        self.pm = plus_minus

    def quantum_number_S(self, *args, **kwargs):
        return self.S

    def quantum_number_L_z(self, *args, **kwargs):
        return self.L_z

    def gerade(self, *args, **kwargs):
        return self.g_or_u

    def plus_minus(self, *args, **kwargs):
        return self.pm


class FakeJob(object):
    def __init__(self, Def=None, results_dir="/tmp", not_int=False, **kwargs):
        if Def is None:
            self.Def = {}
        else:
            self.Def = Def
        self.results_dir = results_dir
        self.Hamilton = FakeHamilton()
        self.Electron = FakeElectron(**kwargs)


class TemporaryFile(object):
    """
    Keep the setup and cleanup of the file in one context manager to keep
    things neat and simple.

    Here we use the context manager pattern, so to use this in your test:
        with TemporaryFile(filename, directory) as filepath:
            ...code using filepath...

    Then you do what you want with the the file and you know that it will get
    cleaned up when you're done with it.
    """
    def __init__(self, filename, directory):
        self.filepath = os.path.join(directory, filename)

    def __enter__(self):
        # Setup
        # if the file already exists, remove it
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)

        return self.filepath

    def __exit__(self, type, value, traceback):
        # Cleanup
        # if the file exists, remove it
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)


def test_WriteTotalEnergy():
    # Fake
    Job = FakeJob({'write_total_energy': 1})
    filename = "energy.txt"
    expected_energy = Job.Hamilton.total_energy(Job)
    with TemporaryFile(filename, Job.results_dir) as filepath:
        # Action
        WriteTotalEnergy(Job, filename=filename)

        # Result
        # check that file exists and that its contents match the expectation
        with open(filepath, 'r') as fh:
            total_energy = fh.read()

        assert float(total_energy) == expected_energy


def test_classify_groundstate_not_a_dimer():
    Job = FakeJob({'write_groundstate_classification': 1})
    Job.NAtom = 5

    assert classify_groundstate(Job) == ""


@pytest.mark.parametrize(
    ("S", "expected_symbol"),
    [
        (2, "{}^5"),
        (0.1, ""),
        (0.5, "{}^2"),
    ]
)
def test_get_spin_part_of_symbol(S, expected_symbol):
    Job = FakeJob(S=S)

    assert get_spin_part_of_symbol(Job) == expected_symbol


@pytest.mark.parametrize(
    ("L_z", "plus_minus", "expected_symbol"),
    [
        (2, "+", "\Delta"),
        (0.1, "+", ""),
        (0, "+", "\Sigma^{+}"),
        (0, "", ""),
    ]
)
def test_get_angular_part_of_symbol(L_z, plus_minus, expected_symbol):
    Job = FakeJob(L_z=L_z, plus_minus=plus_minus)

    assert get_angular_part_of_symbol(Job) == expected_symbol


@pytest.mark.parametrize(
    ("gerade", "expected_symbol"),
    [
        ('g', "_{g}"),
        ('u', "_{u}"),
        ('', ""),
    ]
)
def test_get_gerade_part_of_symbol(gerade, expected_symbol):
    Job = FakeJob()
    Job.Electron = FakeElectron(gerade=gerade)

    assert get_gerade_part_of_symbol(Job) == expected_symbol


def test_write_quantum_number_S_None():
    # Setup
    results_dir = "/tmp"
    filename = "test_quantum_number_S.txt"
    expected_body = ""
    Job = FakeJob({'write_quantum_number_S': 1}, results_dir=results_dir)
    Job.Electron = FakeElectron(S=None)

    # Action & Result
    helper_test_file_gets_created(filename, results_dir, expected_body,
                                  partial(WriteQuantumNumberS, Job, filename))


def test_write_quantum_number_L_z_None():
    # Setup
    results_dir = "/tmp"
    filename = "test_quantum_number_L_z.txt"
    expected_body = ""
    Job = FakeJob({'write_quantum_number_L_z': 1}, results_dir=results_dir)
    Job.Electron = FakeElectron(L_z=None)

    # Action & Result
    helper_test_file_gets_created(filename, results_dir, expected_body,
                                  partial(WriteQuantumNumberLz, Job, filename))


def helper_test_file_gets_created(filename, directory, expected_body,
                                  function_to_test):
    with TemporaryFile(filename, directory) as filepath:
        # Action
        function_to_test()

        # Result
        assert os.path.isfile(filepath)
        with open(filepath, 'r') as fh:
            assert fh.read() == expected_body
