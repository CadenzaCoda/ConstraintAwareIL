class OutOfBoundException(Exception):
    def __init__(self, message=None, errors=None):
        super().__init__(message)
        self.errors = errors


class SolverException(Exception):
    def __init__(self, message=None, errors=None):
        super().__init__(message)
        self.errors = errors
