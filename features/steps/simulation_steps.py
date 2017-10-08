import os

@when(u'PyLATO is run using the single hydrogen job definition file')
def step_impl(context):
    os.system('python TB.py features/support/single_hydrogen_job_def.json')
