import os
import shutil

from behave import fixture


class BackupFile(object):
    """
    Backup the file that is being used and then restore the backup at the end.
    To do this we use the context manager pattern, i.e.
        with BackupFile(filename):
            ...code that manipulates filename...
    Then you do what you want with the the file and you know that it will get
    restored to what it was originally.
    """
    def __init__(self, filename):
        self.original = filename

    def __enter__(self):
        # Setup -- create the backup file:
        backup_name = ".{}.temp".format(os.path.basename(self.original))
        self.backup = os.path.join(os.path.dirname(self.original), backup_name)
        shutil.copyfile(self.original, self.backup)

    def __exit__(self, type, value, traceback):
        # Cleanup -- restore the backup:
        shutil.copyfile(self.backup, self.original)
        os.remove(self.backup)


@fixture
def backup_file(context, filename="filename", **kwargs):
    with BackupFile(filename):
        yield
