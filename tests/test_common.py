"""
test_common
-----------

Unit tests for the pjinoise.common.py module.
"""
import unittest as ut

import numpy as np

from pjinoise import common as c


class CommonTestCase(ut.TestCase):
    def test_convert_color_to_pjinoise_grayscale(self):
        """Given image data, the current color space of that
        image data, and an empty string for the destination
        space, common.convert_color_space() should convert
        the image data into a single float per pixel with a
        value between zero and one.
        """
        # Expected value.
        exp = [
            [
                [0x00, 0x7f, 0xff],
                [0x00, 0x7f, 0xff],
                [0x00, 0x7f, 0xff],
            ],
        ]

        # Set up test data and state.
        a = np.array([
            [
                [0x00, 0x7f, 0xff],
                [0x00, 0x7f, 0xff],
                [0x00, 0x7f, 0xff],
            ],
        ], dtype=np.uint8)
        src_space = 'L'
        dst_space = ''

        # Run test.
        result = c.convert_color_space(a, src_space, dst_space)

        # Extract actual result.
        act = c.grayscale_to_ints_list(result)

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_linear_interpolation(self):
        """Given three arrays that represent a set of points on a line,
        a second set of points on a line, and a set of the relative
        position of a third point between the two other points, return
        an array that is the set of the third point on the line.
        """
        # Expected value.
        exp = [
            [
                [0x00, 0x40, 0x80, 0xc0, 0xff],
                [0x40, 0x80, 0xc0, 0xff, 0xc0],
                [0x80, 0xc0, 0xff, 0xc0, 0x80],
                [0xc0, 0xff, 0xc0, 0x80, 0x40],
                [0xff, 0xc0, 0x80, 0x40, 0x00],
            ],
            [
                [0xff, 0xbf, 0x7f, 0x3f, 0x00],
                [0xbf, 0x7f, 0x3f, 0x00, 0x3f],
                [0x7f, 0x3f, 0x00, 0x3f, 0x7f],
                [0x3f, 0x00, 0x3f, 0x7f, 0xbf],
                [0x00, 0x3f, 0x7f, 0xbf, 0xff],
            ],
        ]

        # Set up test data and state.
        a = np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
            ],
            [
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
            ],
        ], float)
        b = np.array([
            [
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
            ],
        ], float)
        x = np.array([
            [
                [0x00, 0x40, 0x80, 0xc0, 0xff],
                [0x40, 0x80, 0xc0, 0xff, 0xc0],
                [0x80, 0xc0, 0xff, 0xc0, 0x80],
                [0xc0, 0xff, 0xc0, 0x80, 0x40],
                [0xff, 0xc0, 0x80, 0x40, 0x00],
            ],
            [
                [0x00, 0x40, 0x80, 0xc0, 0xff],
                [0x40, 0x80, 0xc0, 0xff, 0xc0],
                [0x80, 0xc0, 0xff, 0xc0, 0x80],
                [0xc0, 0xff, 0xc0, 0x80, 0x40],
                [0xff, 0xc0, 0x80, 0x40, 0x00],
            ],
        ], float)
        a /= 0xff
        b /= 0xff
        x /= 0xff

        # Run test.
        result = c.lerp(a, b, x)

        # Extract actual result.
        act = c.grayscale_to_ints_list(result)

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_remove_private_attrs(self):
        """Given an object's attributes serialized as a dictionary,
        remove any private attributes from the dictionary and return
        the dictionary.
        """
        # Expected value.
        exp = {
            'spam': 1,
            'eggs': 3,
        }

        # Set up test data and state.
        attrs = {
            'spam': 1,
            '_bacon': 2,
            'eggs': 3,
            '_baked_beans': 4,
        }

        # Run test.
        act = c.remove_private_attrs(attrs)

        # Determine whether test passed.
        self.assertDictEqual(exp, act)


class TerpTestCase(ut.TestCase):
    def test_increase_size(self):
        """Given an array and a scaling factor greater than one,
        interpolate the values in the array at the increased size
        of the array and return the array.
        """
        # Expected values.
        exp = [
            [
                [0x00, 0x55, 0xaa, 0xff],
                [0x00, 0x55, 0xaa, 0xff],
                [0x00, 0x55, 0xaa, 0xff],
                [0x00, 0x55, 0xaa, 0xff],
            ],
            [
                [0x55, 0xaa, 0xc7, 0xaa],
                [0x55, 0xaa, 0xc7, 0xaa],
                [0x55, 0xaa, 0xc7, 0xaa],
                [0x55, 0xaa, 0xc7, 0xaa],
            ],
            [
                [0xaa, 0xc7, 0xaa, 0x55],
                [0xaa, 0xc7, 0xaa, 0x55],
                [0xaa, 0xc7, 0xaa, 0x55],
                [0xaa, 0xc7, 0xaa, 0x55],
            ],
            [
                [0xff, 0xaa, 0x55, 0x00],
                [0xff, 0xaa, 0x55, 0x00],
                [0xff, 0xaa, 0x55, 0x00],
                [0xff, 0xaa, 0x55, 0x00],
            ],
        ]

        # Set up test data and state.
        a = np.array([
            [
                [0x00, 0x80, 0xff,],
                [0x00, 0x80, 0xff,],
                [0x00, 0x80, 0xff,],
            ],
            [
                [0x80, 0xff, 0x80,],
                [0x80, 0xff, 0x80,],
                [0x80, 0xff, 0x80,],
            ],
            [
                [0xff, 0x80, 0x00,],
                [0xff, 0x80, 0x00,],
                [0xff, 0x80, 0x00,],
            ],
        ], float)
        a /= 0xff
        factor = 1.5

        # Run test.
        result = c.trilinear_interpolation(a, factor)

        # Extract actual result from test.
        act = c.grayscale_to_ints_list(result)

        # Determine if test passed.
        self.assertListEqual(exp, act)
