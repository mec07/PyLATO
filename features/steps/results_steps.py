import commentjson
import os


@then(u'the total energy in the output file is the onsite energy for a single hydrogen atom')
def step_impl(context):
    ################################
    # Get the outputted total energy
    ################################
    with open(context.job_def_file, 'r') as fh:
        job_def = commentjson.load(fh)

    results_dir = job_def['results_dir']
    total_energy_filepath = os.path.join(results_dir, 'energy.txt')
    with open(total_energy_filepath, 'r') as fh:
        total_energy = fh.read()

    ################################
    # Get the onsite energy
    ################################
    model = job_def['model']
    model_file = "models/{}.json".format(model)
    with open(model_file, 'r') as fh:
        model_def = commentjson.load(fh)

    onsite_energy = model_def["species"][0]["e"]

    assert float(total_energy) == onsite_energy[0][0]
