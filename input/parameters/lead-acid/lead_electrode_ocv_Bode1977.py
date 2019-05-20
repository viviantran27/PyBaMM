#
# Open-circuit voltage in the negative (lead) electrode
#
import autograd.numpy as np


def lead_electrode_ocv_Bode1977(m):
    """
    Dimensional open-circuit voltage in the negative (lead) electrode [V], from [1]_,
    as a function of the molar mass m [mol.kg-1].

    References
    ----------
    .. [1] H Bode. Lead-acid batteries. John Wiley and Sons, Inc., New York, NY, 1977.

    """
    log10m = np.log10(m)
    U = (
        -0.294
        - 0.074 * log10m
        - 0.030 * log10m ** 2
        - 0.031 * log10m ** 3
        - 0.012 * log10m ** 4
    )
    return U
