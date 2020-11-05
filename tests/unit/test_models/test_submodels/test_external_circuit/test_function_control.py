#
# Test function control submodel
#
import pybamm
import tests
import unittest


def external_circuit_function(variables):
    I = variables["Current [A]"]
    V = variables["Terminal voltage [V]"]
    liion_param = pybamm.LithiumIonParameters()

    return (
        V
        + I
        - pybamm.FunctionParameter(
            "Current plus voltage function",
            {"Time [s]": pybamm.t * liion_param.timescale},
        )
    )


class TestFunctionControl(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()
        submodel = pybamm.external_circuit.FunctionControl(
            param, external_circuit_function
        )
        variables = {"Terminal voltage [V]": pybamm.Scalar(0)}
        std_tests = tests.StandardSubModelTests(submodel, variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
