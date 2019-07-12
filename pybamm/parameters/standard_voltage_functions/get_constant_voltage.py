#
# Constant voltage "function"
#
import pybamm


class GetConstantVoltage(pybamm.GetVoltage):
    """
    Sets a constant input voltage for a simulation.

    Parameters
    ----------
    voltage : :class:`pybamm.Symbol` or float
        The size of the voltage in Volts.

    **Extends:"": :class:`pybamm.GetVoltage`
    """

    def __init__(self, voltage=pybamm.electrical_parameters.V_typ):
        self.parameters = {"Cell voltage [V]": voltage}
        self.parameters_eval = {"Cell voltage [V]": voltage}

    def __str__(self):
        return "Constant voltage"

    def __call__(self, t):
        return self.parameters_eval["Cell voltage [V]"]
