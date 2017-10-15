class GenericClass(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def return_a_function(value):
    def a_function(*args, **kwargs):
        return value

    return a_function


def approx_equal(x, y, threshold=1.0e-8):
    return abs(x-y) < threshold
