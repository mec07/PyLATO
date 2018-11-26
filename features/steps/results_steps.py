import commentjson
import os


@then(u'the total energy in the output file is {value}')
def step_impl(context, value):
    with open(context.job_def_file, 'r') as fh:
        job_def = commentjson.load(fh)

    results_dir = job_def['results_dir']
    total_energy_filepath = os.path.join(results_dir, 'energy.txt')
    with open(total_energy_filepath, 'r') as fh:
        total_energy = fh.read()
    print("######################################################")
    print("total_energy = ", total_energy)
    print("######################################################")

    assert total_energy == value
