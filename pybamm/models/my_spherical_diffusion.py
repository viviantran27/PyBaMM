#
# My spherical diffusion model
#
import pybamm


class MySphericalDiffusion(pybamm.BaseModel):
    """A model for diffusion in a sphere.

    **Extends:** :class:`pybamm.BaseModel`
    """

    def __init__(self, param, name="Spherical Diffusion"):
        # Initialise base class
        super().__init__(name)

        # Add parameters as an attribute of the model
        self.param = param

        # Make the concentration variable
        c = pybamm.Variable("Concentration [mol.m-3]", domain="negative particle")

        # Define the governing equations
        N = -self.param.D * pybamm.grad(c)
        dcdt = -pybamm.div(N)

        # Add governing equations to rhs dict
        self.rhs = {c: dcdt}

        # Set boundary conditions
        lbc = pybamm.Scalar(0)
        rbc = -self.param.j / self.param.F / self.param.D
        self.boundary_conditions = {
            c: {"left": (lbc, "Dirichlet"), "right": (rbc, "Neumann")}
        }

        # Set initial conditions
        self.initial_conditions = {c: self.param.c0}

        # Add variables of interest to dict
        self.variables = {
            "Concentration [mol.m-3]": c,
            "Surface concentration [mol.m-3]": pybamm.surf(c),
            "Flux [mol.m-2.s-1]": N,
        }
