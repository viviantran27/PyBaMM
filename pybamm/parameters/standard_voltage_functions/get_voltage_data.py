#
# Load voltage profile from a csv file
#
import pybamm
import os
import pandas as pd
import numpy as np
import warnings
import scipy.interpolate as interp


class GetVoltageData(pybamm.GetVoltage):
    """
    A class which loads a voltage profile from a csv file and creates an
    interpolating function which can be called during solve.

    Parameters
    ----------
    filename : str
        The name of the file to load.
    units : str, optional
        The units of the voltage data which is to be loaded. Can be "[]" for
        dimenionless data (default), or "[V]" for voltage in Volts.
    voltage_scale : :class:`pybamm.Symbol` or float, optional
        The scale the voltage in Volts if loading non-dimensional data. Default
        is to use the typical voltage V_typ

    **Extends:"": :class:`pybamm.GetVoltage`
    """

    def __init__(
        self, filename, units="[]", voltage_scale=pybamm.electrical_parameters.V_typ
    ):
        self.parameters = {"Voltage [V]": voltage_scale}
        self.parameters_eval = {"Voltage [V]": voltage_scale}

        # Load data from csv
        if filename:
            pybamm_path = pybamm.root_dir()
            data = pd.read_csv(
                os.path.join(pybamm_path, "input", "drive_cycles", filename),
                comment="#",
                skip_blank_lines=True,
            ).to_dict("list")

            self.time = np.array(data["time [s]"])
            self.units = units
            self.voltage = np.array(data["voltage " + units])
            # If voltage data is present, load it into the class
        else:
            raise pybamm.ModelError("No input file provided for voltage")

    def __str__(self):
        return "Voltage from data"

    def interpolate(self):
        " Creates the interpolant from the loaded data "
        # If data is dimenionless, multiply by a typical voltage (e.g. data
        # could be C-rate and voltage the 1C discharge voltage). Otherwise,
        # just import the voltage data.
        if self.units == "[]":
            voltage = self.parameters_eval["Cell voltage [V]"] * self.voltage
        elif self.units == "[V]":
            voltage = self.voltage
        else:
            raise pybamm.ModelError(
                "Voltage data must have units [V] or be dimensionless"
            )
        # Interpolate using Piecewise Cubic Hermite Interpolating Polynomial
        # (does not overshoot non-smooth data)
        self.voltage_interp = interp.PchipInterpolator(self.time, voltage)

    def __call__(self, t):
        """
        Calls the interpolating function created using the data from user-supplied
        data file at time t (seconds).
        """

        if np.min(t) < self.time[0] or np.max(t) > self.time[-1]:
            warnings.warn(
                "Requested time ({}) is outside of the data range [{}, {}]".format(
                    t, self.time[0], self.time[-1]
                ),
                pybamm.ModelWarning,
            )

        return self.voltage_interp(t)
