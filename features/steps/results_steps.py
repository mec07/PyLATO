import commentjson
import math
import os


@then(u'the total energy in the output file is equivalent to the onsite energy')
def step_then_the_total_energy_in_the_output_file_is_equivalent_to_the_onsite_energy(context):
    job_def = load_json_file(context.job_def_file)
    total_energy = get_outputted_total_energy(job_def)

    ################################
    # Get the onsite energy
    ################################
    model = job_def['model']
    model_def = load_json_file("models/{}.json".format(model))

    onsite_energy = model_def["species"][0]["e"]

    assert_floats_equal(float(total_energy), onsite_energy[0][0])


@then(u'the total energy in the output file is {expected_energy}')
def step_then_the_total_energy_in_the_output_file_is(context, expected_energy):
    job_def = load_json_file(context.job_def_file)
    total_energy = get_outputted_total_energy(job_def)

    assert_floats_equal(float(total_energy), float(expected_energy))


def load_json_file(json_file):
    with open(json_file, 'r') as file_handle:
        loaded_file = commentjson.load(file_handle)
    return loaded_file


def get_outputted_total_energy(job_def):
    results_dir = job_def['results_dir']
    total_energy_filepath = os.path.join(results_dir, 'energy.txt')
    with open(total_energy_filepath, 'r') as file_handle:
        total_energy = file_handle.read()

    return total_energy


def assert_floats_equal(a, b):
    assert math.isclose(a, b, abs_tol=1e-9), "{} != {}".format(a, b)
