class UnimplementedModelError(Exception):
    def __init__(self, message, code=None):
        error = message
        super(UnimplementedModelError, self).__init__(error)
        self.message = message
        self.code = code


class ChemicalPotentialError(Exception):
    def __init__(self, message, code=None):
        error = message
        super().__init__(error)
        self.message = message
        self.code = code


class SelfConsistencyError(Exception):
    def __init__(self, message, code=None):
        error = message
        super().__init__(error)
        self.message = message
        self.code = code


class UnimplementedMethodError(Exception):
    def __init__(self, message, code=None):
        error = message
        super().__init__(error)
        self.message = message
        self.code = code
