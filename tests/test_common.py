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
