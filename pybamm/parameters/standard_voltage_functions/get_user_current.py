#
# Allow a user-defined voltage function
#
import pybamm


class GetUserVoltage(pybamm.GetVoltage):
    """
    Sets a user-defined function as the input voltage for a simulation.

    Parameters
    ----------
    function : method
        The method which returns the voltage (in Volts) as a function of time
        (in seconds). The first argument of function must be time, followed by
        any keyword arguments, i.e. function(t, **kwargs).
    **kwargs : Any keyword arguments required by function.

    **Extends:"": :class:`pybamm.GetVoltage`
    """

    def __init__(self, function, **kwargs):
        self.parameters = kwargs
        self.parameters_eval = kwargs
        self.function = function

    def __str__(self):
        return "User defined voltage"

    def __call__(self, t):
        return self.function(t, **self.parameters_eval)
