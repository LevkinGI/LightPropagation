class InequalityException(Exception):
    def __init__(self):
        message = f'The inequality must be met: h <= tau^2 / (2k)'
        super().__init__(message)


class SigmaException(Exception):
    def __init__(self):
        message = f'Sigma must be 0'
        super().__init__(message)

class LengthConditionsException(Exception):
    def __init__(self):
        message = f'Length a_z_0 must be less than a_0_t'
        super().__init__(message)
