import os
import pytest

from pylato.pylato_IO import WriteTotalEnergy


class FakeHamilton(object):
    def total_energy(*args):
        return 0


class FakeJob(object):
    def __init__(self, Def=None, results_dir="/tmp"):
        if Def is None:
            self.Def = {}
        else:
            self.Def = Def
        self.results_dir = results_dir
        self.Hamilton = FakeHamilton()


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
