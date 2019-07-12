#
# Base class for setting/getting voltage
#


class GetVoltage(object):
    """
    The base class for setting the input voltage for a simulation. The parameters
    dictionary holds the symbols of any parameters required to evaluate the voltage.
    During processing, the evaluated parameters are stored in parameters_eval.
    """

    def __init__(self):
        self.parameters = {}
        self.parameters_eval = {}

    def __str__(self):
        return "Base voltage"

    def __call__(self, t):
        return 1
