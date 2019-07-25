#
# Tests for voltage input functions
#
import pybamm
import numbers
import unittest
import numpy as np


class TestVoltageFunctions(unittest.TestCase):
    def test_base_voltage(self):
        function = pybamm.GetVoltage()
        self.assertEqual(function(10), 1)

    def test_constant_voltage(self):
        function = pybamm.GetConstantVoltage(voltage=4)
        assert isinstance(function(0), numbers.Number)
        assert isinstance(function(np.zeros(3)), numbers.Number)
        assert isinstance(function(np.zeros([3, 3])), numbers.Number)

        # test simplify
        voltage = pybamm.electrical_parameters.voltage_with_time
        parameter_values = pybamm.ParameterValues(
            {
                "Typical voltage [V]": 2,
                "Typical timescale [s]": 1,
                "Voltage function": pybamm.GetConstantVoltage(),
            }
        )
        processed_voltage = parameter_values.process_symbol(voltage)
        self.assertIsInstance(processed_voltage.simplify(), pybamm.Scalar)

    def test_user_voltage(self):
        # create user-defined sin function

        def my_fun(t, A, omega):
            return A * np.sin(2 * np.pi * omega * t)

        # choose amplitude and frequency
        A = pybamm.electrical_parameters.V_typ
        omega = 3

        # pass my_fun to GetUserVoltage class, giving the additonal parameters as
        # keyword arguments
        voltage = pybamm.GetUserVoltage(my_fun, A=A, omega=omega)

        # set and process parameters
        parameter_values = pybamm.ParameterValues(
            {
                "Typical voltage [V]": 2,
                "Typical timescale [s]": 1,
                "Voltage function": voltage,
            }
        )
        dimensional_voltage = pybamm.electrical_parameters.dimensional_voltage_with_time
        dimensional_voltage_eval = parameter_values.process_symbol(dimensional_voltage)

        def user_voltage(t):
            return dimensional_voltage_eval.evaluate(t=t)

        # check output types
        standard_tests = StandardVoltageFunctionTests([user_voltage])
        standard_tests.test_all()

        # check output correct value
        time = np.linspace(0, 3600, 600)
        np.testing.assert_array_almost_equal(
            voltage(time), 2 * np.sin(2 * np.pi * 3 * time)
        )


class StandardVoltageFunctionTests(object):
    def __init__(self, function_list, always_array=False):
        self.function_list = function_list
        self.always_array = always_array

    def test_output_type(self):
        for function in self.function_list:
            if self.always_array is True:
                assert isinstance(function(0), np.ndarray)
            else:
                assert isinstance(function(0), numbers.Number)
            assert isinstance(function(np.zeros(3)), np.ndarray)
            assert isinstance(function(np.zeros([3, 3])), np.ndarray)

    def test_all(self):
        self.test_output_type()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
