class UnimplementedModelError(Exception):
    def __init__(self, message, code=None):
        error = message
        super(UnimplementedModelError, self).__init__(error)
        self.message = message
        self.code = code
