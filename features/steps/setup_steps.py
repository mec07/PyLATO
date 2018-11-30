import commentjson


@given(u'the TBcanonical_s model is set to have U/|t| = {ratio}')
def step_given_the_tbcanonical_s_model_is_set_to_have_U_t(context, ratio):
    model_file = "models/TBcanonical_s.json"

    # update model
    with open(model_file, 'r') as file_handle:
        model = commentjson.loads(file_handle.read())

    model['species'][0]['U'] = float(ratio)
    model['hamiltonian'][0][0]['r0'] = 1.0
    model['hamiltonian'][0][0]['vsss'] = -1.0

    with open(model_file, 'w') as file_handle:
        commentjson.dump(model, file_handle)
