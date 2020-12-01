"""
test_ease
~~~~~~~~~

Unit tests for easing functions in pjinoise.
"""
import unittest as ut

import numpy as np

from pjinoise import ease


# Utility functions.
def to_percent(value:float, scale:float = 0xff) -> float:
    return value / scale


# Test cases.
class FunctionsTestCase(ut.TestCase):
    def test_in_out_cubic(self):
        """Given a value between zero and one, ease.in_out_cubic
        should perform the in out cubic easing function on the
        value and return it.
        """
        # The expected value.
        # This is in hexadecimal integers because doing it as floats
        # is easier to read.
        exp = [
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
        ]

        # Set up test data and state.
        # Again, I'm using hexadecimal integers here to make it easier
        # to read.
        img = np.array([
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        ])
        img = to_percent(img)

        # Run the test.
        result = ease.in_out_cubic(img)

        # Extract the actual value for comparison.
        result = result * 0xff
        result = np.around(result).astype(int)
        act = result.tolist()

        # Determine whether the test passed.
        self.assertListEqual(exp, act)

    def test_in_cubic(self):
        """Given a value between zero and one, ease.in_cubic
        should perform the in quint easing function on the
        value and return it.
        """
        # The expected value.
        # This is in hexadecimal integers because doing it as floats
        # is easier to read.
        exp = [
            [0x04, 0x20, 0x3f, 0x6d, 0xad, 0xff],
            [0x04, 0x20, 0x3f, 0x6d, 0xad, 0xff],
            [0x04, 0x20, 0x3f, 0x6d, 0xad, 0xff],
            [0x04, 0x20, 0x3f, 0x6d, 0xad, 0xff],
            [0x04, 0x20, 0x3f, 0x6d, 0xad, 0xff],
        ]

        # Set up test data and state.
        # Again, I'm using hexadecimal integers here to make it easier
        # to read.
        img = np.array([
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        ])
        img = to_percent(img)

        # Run the test.
        result = ease.in_cubic(img)

        # Extract the actual value for comparison.
        result = result * 0xff
        result = np.around(result).astype(int)
        act = result.tolist()

        # Determine whether the test passed.
        self.assertListEqual(exp, act)

    def test_in_quint(self):
        """Given a value between zero and one, ease.in_quint
        should perform the in quint easing function on the
        value and return it.
        """
        # The expected value.
        # This is in hexadecimal integers because doing it as floats
        # is easier to read.
        exp = [
            [0x00, 0x08, 0x19, 0x3e, 0x85, 0xff],
            [0x00, 0x08, 0x19, 0x3e, 0x85, 0xff],
            [0x00, 0x08, 0x19, 0x3e, 0x85, 0xff],
            [0x00, 0x08, 0x19, 0x3e, 0x85, 0xff],
            [0x00, 0x08, 0x19, 0x3e, 0x85, 0xff],
        ]

        # Set up test data and state.
        # Again, I'm using hexadecimal integers here to make it easier
        # to read.
        img = np.array([
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        ])
        img = to_percent(img)

        # Run the test.
        result = ease.in_quint(img)

        # Extract the actual value for comparison.
        result = result * 0xff
        result = np.around(result).astype(int)
        act = result.tolist()

        # Determine whether the test passed.
        self.assertListEqual(exp, act)
