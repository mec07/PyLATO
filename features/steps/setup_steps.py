import commentjson

from behave import given


@given(u'the {model_name} model is set to have U/|t| = {ratio}')
def given_the_model_is_set_to_have_U_t_ratio(context, model_name, ratio):
    model = get_model(model_name)

    model['species'][0]['U'] = float(ratio)
    model['hamiltonian'][0][0]['r0'] = 1.0
    model['hamiltonian'][0][0]['vsss'] = -1.0
    model['hamiltonian'][0][0]['vpps'] = 1.0
    model['hamiltonian'][0][0]['vdds'] = -1.0

    write_model(model, model_name)


@given(u'the {model_name} model is set to have {num_electrons} electrons')
def given_the_model_is_set_to_have_x_electrons(context, model_name,
                                               num_electrons):
    model = get_model(model_name)

    # update model
    model['species'][0]['NElectrons'] = num_electrons

    write_model(model, model_name)


def get_model(model_name):
    model_file = "models/{}.json".format(model_name)
    with open(model_file, 'r') as file_handle:
        return commentjson.load(file_handle)


def write_model(model, model_name):
    model_file = "models/{}.json".format(model_name)
    with open(model_file, 'w') as file_handle:
        commentjson.dump(model, file_handle)

