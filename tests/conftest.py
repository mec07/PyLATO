import commentjson


def load_json_file(json_file):
    with open(json_file, 'r') as file_handle:
        loaded_file = commentjson.load(file_handle)
    return loaded_file
