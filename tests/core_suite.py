"""
core_suite
~~~~~~~~~~

Unit tests that cover the core functionality of pjinoise. They are
organized in increasing specificity, so that if basic tests fail,
the process can be aborted early to prevent having to deal with a
flood of errors.
"""
import unittest as ut

from tests import test_common as tc


core = ut.TestSuite()
core.addTest(tc.CommonTestCase('test_convert_color_to_pjinoise_grayscale'))
core.addTest(tc.CommonTestCase('test_linear_interpolation'))
core.addTest(tc.CommonTestCase('test_remove_private_attrs'))
core.addTest(tc.TerpTestCase('test_increase_size'))
core.addTest(tc.TerpTestCase('test_decrease_size'))
