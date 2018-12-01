import os

from unittest.mock import patch
from pylato.main import main


@when(u'PyLATO is run using the job definition file: "{file_name}"')
def step_when_pylato_is_run_using_the_job_definition_file(context, file_name):
    context.job_def_file = "features/support/{}".format(file_name)
    runtime_arguments = ["pylato/main.py", context.job_def_file]
    with patch("sys.argv", runtime_arguments):
        assert main()
