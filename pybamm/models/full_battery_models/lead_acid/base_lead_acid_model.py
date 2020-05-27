#
# Lead acid base model class
#

import pybamm


class BaseModel(pybamm.BaseBatteryModel):
    """
    Overwrites default parameters from Base Model with default parameters for
    lead-acid models


    **Extends:** :class:`pybamm.BaseBatteryModel`

    """

    def __init__(self, options=None, name="Unnamed lead-acid model"):
        super().__init__(options, name)
        self.param = pybamm.standard_parameters_lead_acid

        # Default timescale is discharge timescale
        self.timescale = self.param.tau_discharge
        self.set_standard_output_variables()

    @property
    def default_parameter_values(self):
        return pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Sulzer2019)

    @property
    def default_geometry(self):
        if self.options["dimensionality"] == 0:
            return pybamm.Geometry("1D macro")
        elif self.options["dimensionality"] == 1:
            return pybamm.Geometry("1+1D macro")
        elif self.options["dimensionality"] == 2:
            return pybamm.Geometry("2+1D macro")

    @property
    def default_var_pts(self):
        # Choose points that give uniform grid for the standard parameter values
        var = pybamm.standard_spatial_vars
        return {var.x_n: 25, var.x_s: 41, var.x_p: 34, var.y: 10, var.z: 10}

    def set_soc_variables(self):
        "Set variables relating to the state of charge."
        # State of Charge defined as function of dimensionless electrolyte concentration
        z = pybamm.standard_spatial_vars.z
        soc = (
            pybamm.Integral(self.variables["X-averaged electrolyte concentration"], z)
            * 100
        )
        self.variables.update({"State of Charge": soc, "Depth of Discharge": 100 - soc})

        # Fractional charge input
        if "Fractional Charge Input" not in self.variables:
            fci = pybamm.Variable("Fractional Charge Input", domain="current collector")
            self.variables["Fractional Charge Input"] = fci
            self.rhs[fci] = -self.variables["Total current density"] * 100
            self.initial_conditions[fci] = self.param.q_init * 100

    def set_sei_submodel(self):

        self.submodels["negative sei"] = pybamm.sei.NoSEI(self.param, "Negative")
        self.submodels["positive sei"] = pybamm.sei.NoSEI(self.param, "Positive")
