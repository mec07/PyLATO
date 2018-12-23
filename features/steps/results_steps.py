import commentjson
import math
import os

from behave import then


@then(u'the total energy in the output file is equivalent to the onsite energy')
def then_the_total_energy_in_the_output_file_is_equivalent_to_the_onsite_energy(context):
    total_energy = get_outputted_total_energy(context)

    ################################
    # Get the onsite energy
    ################################
    job_def = load_json_file(context.job_def_file)
    model = job_def['model']
    model_def = load_json_file("models/{}.json".format(model))

    onsite_energy = model_def["species"][0]["e"]

    assert_floats_equal(float(total_energy), onsite_energy[0][0])


@then(u'the total energy in the output file is {expected_energy}')
def then_the_total_energy_in_the_output_file_is(context, expected_energy):
    total_energy = get_outputted_total_energy(context)

    assert_floats_equal(float(total_energy), float(expected_energy))


@then(u'the quantum number S is {expected_S}')
def then_the_quantum_number_S_is(context, expected_S):
    outputted_S = get_outputted_quantum_number_S(context)

    assert_floats_equal(float(expected_S), float(outputted_S))


def load_json_file(json_file):
    with open(json_file, 'r') as file_handle:
        loaded_file = commentjson.load(file_handle)
    return loaded_file


def get_outputted_total_energy(context):
    return get_value_from_file(context, 'energy.txt')


def get_outputted_quantum_number_S(context):
    return get_value_from_file(context, 'quantum_number_S.txt')


def get_value_from_file(context, filename):
    job_def = load_json_file(context.job_def_file)
    results_dir = job_def['results_dir']
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'r') as file_handle:
        value = file_handle.read()

    return value


def assert_floats_equal(a, b):
    assert math.isclose(a, b, abs_tol=1e-9), "{} != {}".format(a, b)
