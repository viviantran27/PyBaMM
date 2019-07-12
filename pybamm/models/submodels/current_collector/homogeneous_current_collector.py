#
# Class for uniform current collectors
#
import pybamm
from .base_current_collector import BaseModel


class Uniform(BaseModel):
    """A submodel for uniform potential in the current collectors which
    is valid in the limit of fast conductivity in the current collectors.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.current_collector.BaseModel`
    """

    def __init__(self, param, options=None):
        super().__init__(param)
        if options:
            self.options = options
        else:
            self.options = {"problem type": "potentiostatic"}

    def get_fundamental_variables(self):
        if self.options["problem type"] == "potentiostatic":
            variables = self._get_potentiostatic_fundamental_variables()
        elif self.options["problem type"] == "galvanostatic":
            variables = self._get_galvanostatic_fundamental_variables()
        else:
            raise pybamm.OptionError

        return variables

    def _get_galvanostatic_fundamental_variables(self):
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = self.param.current_with_time
        variables = self._get_standard_current_variables(i_cc, i_boundary_cc)
        return variables

    def _get_potentiostatic_fundamental_variables(self):

        phi_s_cn = pybamm.Scalar(0)
        phi_s_cp = self.param.voltage_with_time

        variables = self._get_standard_potential_variables(phi_s_cn, phi_s_cp)
        return variables

