"""
test_operations
~~~~~~~~~~~~~~~

Unit tests for the pjinoise.operations module.
"""
import unittest as ut
from typing import Callable, List

import numpy as np

from pjinoise import operations as op


# Utility functions.
def run_test(operation: Callable, amount: float = 1.0) -> List[int]:
    a = np.array([
        [0x00, 0x40, 0x80, 0xc0, 0xff,],
        [0x00, 0x40, 0x80, 0xc0, 0xff,],
        [0x00, 0x40, 0x80, 0xc0, 0xff,],
        [0x00, 0x40, 0x80, 0xc0, 0xff,],
        [0x00, 0x40, 0x80, 0xc0, 0xff,],
    ])
    b = np.array([
        [0xff, 0xc0, 0x80, 0x40, 0x00,],
        [0xff, 0xc0, 0x80, 0x40, 0x00,],
        [0xff, 0xc0, 0x80, 0x40, 0x00,],
        [0xff, 0xc0, 0x80, 0x40, 0x00,],
        [0xff, 0xc0, 0x80, 0x40, 0x00,],
    ])
    a = a / 0xff
    b = b / 0xff
    result = operation(a, b, amount)
    result = result * 0xff
    result = np.around(result).astype(int)
    return result.tolist()


# Test classes.
class MaskedTestCase(ut.TestCase):
    def test_masked_replace(self):
        """Given two arrays, an amount, and a mask, the replace
        operation should blend the two arrays. Values that are
        white in the mask should be the second array. Values that
        are black in the mask should be first array. The lighter
        the gray, the more the result should be the second array
        than the first.
        """
        # Expected values.
        exp = [
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x80, 0x80, 0x80, 0x80, 0x7f,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
        ]

        # Set up test data and state.
        a = np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ], dtype=float)
        b = np.array([
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
        ], dtype=float)
        mask = np.array([
            [0x00, 0x00, 0x00, 0x00, 0x00,],
            [0x00, 0x00, 0x00, 0x00, 0x00,],
            [0x80, 0x80, 0x80, 0x80, 0x80,],
            [0xff, 0xff, 0xff, 0xff, 0xff,],
            [0xff, 0xff, 0xff, 0xff, 0xff,],
        ], dtype=float)
        a = a / 0xff
        b = b / 0xff
        mask = mask / 0xff
        amount = 1

        # Run test.
        result = op.replace(a, b, amount, mask)

        # Extract actual values from result.
        result = np.around(result * 0xff).astype(int)
        act = result.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)


class OperationsTestCase(ut.TestCase):
    def test_all_operations_handle_color(self):
        """The result of an operation on two RGB images should
        return unsigned integers from zero to 255.
        """
        # Expected value.
        exp = []

        # Set up test data and state.
        a = np.array([
            [
                [
                    [0x00, 0x00, 0x00],
                    [0x40, 0x40, 0x40],
                    [0x80, 0x80, 0x80],
                    [0xc0, 0xc0, 0xc0],
                    [0xff, 0xff, 0xff],
                ],
                [
                    [0x00, 0x00, 0x00],
                    [0x40, 0x40, 0x40],
                    [0x80, 0x80, 0x80],
                    [0xc0, 0xc0, 0xc0],
                    [0xff, 0xff, 0xff],
                ],
                [
                    [0x00, 0x00, 0x00],
                    [0x40, 0x40, 0x40],
                    [0x80, 0x80, 0x80],
                    [0xc0, 0xc0, 0xc0],
                    [0xff, 0xff, 0xff],
                ],
                [
                    [0x00, 0x00, 0x00],
                    [0x40, 0x40, 0x40],
                    [0x80, 0x80, 0x80],
                    [0xc0, 0xc0, 0xc0],
                    [0xff, 0xff, 0xff],
                ],
                [
                    [0x00, 0x00, 0x00],
                    [0x40, 0x40, 0x40],
                    [0x80, 0x80, 0x80],
                    [0xc0, 0xc0, 0xc0],
                    [0xff, 0xff, 0xff],
                ],
            ],
        ], dtype=np.uint8)
        b = np.array([
            [
                [
                    [0x00, 0xaa, 0xff],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x00, 0x00],
                ],
                [
                    [0x00, 0xaa, 0xff],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x00, 0x00],
                ],
                [
                    [0x00, 0xaa, 0xff],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x00, 0x00],
                ],
                [
                    [0x00, 0x00, 0x00],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0xaa, 0xff],
                ],
                [
                    [0x00, 0x00, 0x00],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0xaa, 0xff],
                ],
            ],
        ], dtype=np.uint8)
        ops = op.registered_ops.values()
        act = []
        for o in ops:

            # Run test.
            result = o(a, b)

            # Extract actual result.
            result = np.array(result)
            if result.dtype != np.uint8:
                act.append(o)
            elif np.max(result) > 0xff:
                act.append(o)
            elif np.min(result) < 0x00:
                act.append(o)

        # Determine if test passed.
        self.assertEqual(exp, act)

    def test_all_operations_handle_grayscale(self):
        """The result of an operation on two grayscale images should
        not return any values greater than one or less than zero.
        """
        # Expected value.
        exp = []

        # Set up test data and state.
        a = np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ], dtype=float)
        b = np.array([
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ], dtype=float)
        a = a / 0xff
        b = b / 0xff
        ops = op.registered_ops.values()
        act = []
        for o in ops:
            if o in op.color_only_ops:
                continue

            # Run test.
            result = o(a, b)

            # Extract actual result.
            result = np.array(result)
            if np.max(result) > 1.0:
                act.append(o)
            elif np.min(result) < 0.0:
                act.append(o)

        # Determine if test passed.
        self.assertEqual(exp, act)

    def test_all_operations_handle_mixed(self):
        """The result of an operation on a grayscale image and an
        RGB image should return unsigned integers from zero to 255.
        """
        # Expected value.
        exp = []

        # Set up test data and state.
        a = np.array([
            [
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
            ],
        ], dtype=float)
        a = a / 0xff
        b = np.array([
            [
                [
                    [0x00, 0xaa, 0xff],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x00, 0x00],
                ],
                [
                    [0x00, 0xaa, 0xff],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x00, 0x00],
                ],
                [
                    [0x00, 0xaa, 0xff],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x00, 0x00],
                ],
                [
                    [0x00, 0x00, 0x00],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0xaa, 0xff],
                ],
                [
                    [0x00, 0x00, 0x00],
                    [0x00, 0x2a, 0x40],
                    [0x00, 0x55, 0x80],
                    [0x00, 0x80, 0xc0],
                    [0x00, 0xaa, 0xff],
                ],
            ],
        ], dtype=np.uint8)
        ops = op.registered_ops.values()
        act = []
        for o in ops:

            # Run test.
            result = o(a, b)

            # Extract actual result.
            result = np.array(result)
            if result.dtype != np.uint8:
                act.append(o)
            elif np.max(result) > 0xff:
                act.append(o)
            elif np.min(result) < 0x00:
                act.append(o)

        # Determine if test passed.
        self.assertEqual(exp, act)

    def test_difference(self):
        """Given two arrays, operations.difference should return the
        absolute value of the difference between each point in the
        arrays.
        """
        exp = [
            [0xff, 0x80, 0x00, 0x80, 0xff,],
            [0xff, 0x80, 0x00, 0x80, 0xff,],
            [0xff, 0x80, 0x00, 0x80, 0xff,],
            [0xff, 0x80, 0x00, 0x80, 0xff,],
            [0xff, 0x80, 0x00, 0x80, 0xff,],
        ]
        act = run_test(op.difference)
        self.assertListEqual(exp, act)

    def test_multiply(self):
        """Given two arrays, operations.multiply should return the
        value of the product of each point in the arrays. Since each
        value in the arrays is between zero and one, this results in
        a darkening of the image.
        """
        exp = [
            [0x00, 0x30, 0x40, 0x30, 0x00,],
            [0x00, 0x30, 0x40, 0x30, 0x00,],
            [0x00, 0x30, 0x40, 0x30, 0x00,],
            [0x00, 0x30, 0x40, 0x30, 0x00,],
            [0x00, 0x30, 0x40, 0x30, 0x00,],
        ]
        act = run_test(op.multiply)
        self.assertListEqual(exp, act)

    def test_multiply_with_half_effect(self):
        """Given two arrays and an amount the operation should affect
        the first array, operations.multiply should return the value
        of the product of each point in the arrays. Since each value
        in the arrays is between zero and one, this results in a
        darkening of the image.
        """
        exp = [
            [0x00, 0x38, 0x60, 0x78, 0x80,],
            [0x00, 0x38, 0x60, 0x78, 0x80,],
            [0x00, 0x38, 0x60, 0x78, 0x80,],
            [0x00, 0x38, 0x60, 0x78, 0x80,],
            [0x00, 0x38, 0x60, 0x78, 0x80,],
        ]
        act = run_test(op.multiply, .5)
        self.assertListEqual(exp, act)

    def test_overlay(self):
        """Given two arrays, operations.overlay should perform the
        overlay operation on the two arrays and return the result.
        The overlay operation is a function that darkens points that
        were darker than mid-gray in the a image and brightens points
        that were brighter than mid-gray. This is similar to running
        a multiply operation on the dark parts and a screen operation
        on the bright parts.
        """
        exp = [
            [0x00, 0x60, 0x80, 0xa1, 0xff,],
            [0x00, 0x60, 0x80, 0xa1, 0xff,],
            [0x00, 0x60, 0x80, 0xa1, 0xff,],
            [0x00, 0x60, 0x80, 0xa1, 0xff,],
            [0x00, 0x60, 0x80, 0xa1, 0xff,],
        ]
        act = run_test(op.overlay)
        self.assertListEqual(exp, act)

    def test_replace(self):
        """Given two arrays, operations.replace should replace the
        value of the first image with the value of the second image
        for every point in the image, returning the result.
        """
        exp = [
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
            [0xff, 0xc0, 0x80, 0x40, 0x00,],
        ]
        act = run_test(op.replace)
        self.assertListEqual(exp, act)

    def test_screen(self):
        """Given two arrays, operations.screen should invert the value
        of each point of each array, multiply those values, and then
        invert the result before returning it. This results in the
        opposite of the multiply operation: a brightening of the image.
        """
        exp = [
            [0xff, 0xd0, 0xc0, 0xd0, 0xff,],
            [0xff, 0xd0, 0xc0, 0xd0, 0xff,],
            [0xff, 0xd0, 0xc0, 0xd0, 0xff,],
            [0xff, 0xd0, 0xc0, 0xd0, 0xff,],
            [0xff, 0xd0, 0xc0, 0xd0, 0xff,],
        ]
        act = run_test(op.screen)
        self.assertListEqual(exp, act)
