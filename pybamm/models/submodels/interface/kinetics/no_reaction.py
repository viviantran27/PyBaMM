#
# Bulter volmer class
#

import pybamm
from .base_kinetics import BaseModel


class NoReaction(BaseModel):
    """
    Base submodel for when no reaction occurs

    Parameters
    ----------
    param :
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.


    **Extends:** :class:`pybamm.interface.kinetics.BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_kinetics(self, j0, ne, eta_r, T):
        return pybamm.Scalar(0)
