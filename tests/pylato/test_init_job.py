import pytest

from pylato.exceptions import FileNotFoundError
from pylato.init_job import InitJob


class TestInitJob:
    def test_file_not_found_error(self):
        job_file = "/kjdalfkjlkhhhhhh/does/not/exist"
        expected_message = "Unable to find job file: {}".format(job_file)
        with pytest.raises(FileNotFoundError, message=expected_message):
            InitJob(job_file)
