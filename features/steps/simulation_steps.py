import os


@when(u'PyLATO is run using the single hydrogen job definition file')
def step_impl(context):
    context.job_def_file = "features/support/single_hydrogen_job_def.json"
    exit_status = os.system(f"python pylato/main.py {context.job_def_file}")
    assert exit_status == 0
