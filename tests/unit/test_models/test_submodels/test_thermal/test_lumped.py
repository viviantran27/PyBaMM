#
# Test lumped thermal submodel
#

import pybamm
import tests
import unittest

from tests.unit.test_models.test_submodels.test_thermal.coupled_variables import (
    coupled_variables,
)


class TestLumped(unittest.TestCase):
    def test_public_functions(self):
        param = pybamm.LithiumIonParameters()

        submodel = pybamm.thermal.Lumped(param)
        std_tests = tests.StandardSubModelTests(submodel, coupled_variables)
        std_tests.test_all()

        submodel = pybamm.thermal.Lumped(param, cc_dimension=1)
        std_tests = tests.StandardSubModelTests(submodel, coupled_variables)
        std_tests.test_all()

        submodel = pybamm.thermal.Lumped(param, cc_dimension=2)
        std_tests = tests.StandardSubModelTests(submodel, coupled_variables)
        std_tests.test_all()

        submodel = pybamm.thermal.Lumped(param, cc_dimension=0, geometry="pouch")
        std_tests = tests.StandardSubModelTests(submodel, coupled_variables)
        std_tests.test_all()

        submodel = pybamm.thermal.Lumped(param, cc_dimension=1, geometry="pouch")
        std_tests = tests.StandardSubModelTests(submodel, coupled_variables)
        std_tests.test_all()

        submodel = pybamm.thermal.Lumped(param, cc_dimension=2, geometry="pouch")
        std_tests = tests.StandardSubModelTests(submodel, coupled_variables)
        std_tests.test_all()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
