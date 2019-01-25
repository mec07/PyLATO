import pytest

from pylato.exceptions import SelfConsistencyError
from pylato.init_job import InitJob
from pylato.self_consistency import PerformSelfConsistency


@pytest.mark.parametrize(
    ("name", "scf_max_loops", "error_type"),
    [
        ("Zero value", 0, AssertionError),
        ("Negative value", -1, AssertionError),
        ("String value", "blah", ValueError),
        ("Self consistency error", 1, SelfConsistencyError),
    ]
)
def test_PerformSelfConsistency_errors(name, scf_max_loops, error_type):
    # Setup
    Job = InitJob("test_data/JobDef_scase.json")
    Job.Def['scf_max_loops'] = scf_max_loops

    # Action & Result
    with pytest.raises(error_type):
        PerformSelfConsistency(Job)
