"""
test_ease
~~~~~~~~~

Unit tests for easing functions in pjinoise.
"""
import unittest as ut

import numpy as np

from pjinoise import ease
from pjinoise.common import grayscale_to_ints_list


# Utility functions.
def to_percent(value:float, scale:float = 0xff) -> float:
    return value / scale


def ease_test(obj, exp, e):
    a = np.array([
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
    ], dtype=float)
    a = a / 0xff
    result = e(a)
    act = grayscale_to_ints_list(result, int)
    obj.assertListEqual(exp, act)


def overflows_test(obj, exp, a, action):
    @ease.overflows
    def spam(a):
        return a

    a = a / 0xff
    result = spam(a, action)
    act = grayscale_to_ints_list(result, int)
    obj.assertListEqual(exp, act)


# Test cases.
class DecoratorsTestCase(ut.TestCase):
    def test_clip_clips_results(self):
        """When passed an array and clip(), the overflows decorator
        should return an array that has been clipped to be between
        zero and one. This decorator should be using on easing
        functions that can result results that exceed zero or one,
        allowing the results to be rescaled to be within the expected
        values.
        """
        # Expected value.
        exp = [
            [
                [0x00, 0x00, 0x80, 0xff, 0xff,],
                [0x00, 0x00, 0x80, 0xff, 0xff,],
                [0x00, 0x00, 0x80, 0xff, 0xff,],
                [0x00, 0x00, 0x80, 0xff, 0xff,],
                [0x00, 0x00, 0x80, 0xff, 0xff,],
            ],
            [
                [0x00, 0x00, 0x80, 0xff, 0xff,],
                [0x00, 0x00, 0x80, 0xff, 0xff,],
                [0x00, 0x00, 0x80, 0xff, 0xff,],
                [0x00, 0x00, 0x80, 0xff, 0xff,],
                [0x00, 0x00, 0x80, 0xff, 0xff,],
            ],
        ]

        # Set up test data and state.
        a = np.array([
            [
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
            ],
            [
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
            ],
        ], dtype=float)
        action = ease.clip

        # Run test and determine if passed.
        overflows_test(self, exp, a, action)

    def test_nochange_does_not_change_results(self):
        """When passed an array and nochange(), the overflows decorator
        should return an unchanged array. This decorator should be
        used on easing functions that can result results that exceed
        zero or one, allowing the results to be rescaled to be within
        the expected values.
        """
        # Expected value.
        exp = [
            [
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
            ],
            [
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
            ],
        ]

        # Set up test data and state.
        a = np.array([
            [
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
            ],
            [
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
            ],
        ], dtype=float)
        action = ease.nochange

        # Run test and determine if passed.
        overflows_test(self, exp, a, action)

    def test_rescale_rescales_results(self):
        """When passed an array and rescale(), the overflows decorator
        should return an array that has been rescales to be between
        zero and one. This decorator should be using on easing
        functions that can result results that exceed zero or one,
        allowing the results to be rescaled to be within the expected
        values.
        """
        # Expected value.
        exp = [
            [
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
            ],
            [
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
                [0x00, 0x40, 0x80, 0xbf, 0xff,],
            ],
        ]

        # Set up test data and state.
        a = np.array([
            [
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
            ],
            [
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
                [-0x80, 0x00, 0x80, 0xff, 0x180,],
            ],
        ], dtype=float)
        action = ease.rescale

        # Run test and determine if passed.
        overflows_test(self, exp, a, action)


class EasingFunctionsTestCase(ut.TestCase):
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

    def test_in_back(self):
        """Perform the in-back easing function."""
        exp = [
            [0x17, 0x12, 0x08, 0x00, 0x02, 0x16, 0x42, 0x8f, 0xff],
            [0x17, 0x12, 0x08, 0x00, 0x02, 0x16, 0x42, 0x8f, 0xff],
            [0x17, 0x12, 0x08, 0x00, 0x02, 0x16, 0x42, 0x8f, 0xff],
            [0x17, 0x12, 0x08, 0x00, 0x02, 0x16, 0x42, 0x8f, 0xff],
            [0x17, 0x12, 0x08, 0x00, 0x02, 0x16, 0x42, 0x8f, 0xff],
        ]
        e = ease.in_back
        ease_test(self, exp, e)

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
