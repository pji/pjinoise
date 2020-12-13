"""
test_sources
~~~~~~~~~~~~

Unit tests for the pjinoise.generator module.
"""
from copy import deepcopy
import unittest as ut
from unittest.mock import call, patch

import numpy as np

from pjinoise import sources as s
from pjinoise.common import grayscale_to_ints_list, print_array
from pjinoise.constants import P


# Common test functions.
def source_fill_test(test, exp, src_cls, src_kwargs, size, loc=(0, 0, 0)):
    src = src_cls(**src_kwargs)
    result = src.fill(size, loc)
    act = grayscale_to_ints_list(result)
    test.assertListEqual(exp, act)


# Test cases.
class CachingTestCase(ut.TestCase):
    def test_cache_fill(self):
        """The first time a fill is generated from a caching source,
        that fill should be cached and returned every time an instance
        of that class with the same key generates a fill of the same
        size.
        """
        # Expected value.
        exp = [
            [
                [0x40, 0x40, 0x40,],
                [0x40, 0x40, 0x40,],
                [0x40, 0x40, 0x40,],
            ],
            [
                [0x40, 0x40, 0x40,],
                [0x40, 0x40, 0x40,],
                [0x40, 0x40, 0x40,],
            ],
        ]

        # Set up test data and state.
        class Source(s.ValueSource):
            def __init__(self, value):
                self.value = value

            def fill(self, size, _):
                a = np.zeros(size, dtype=float)
                a.fill(self.value)
                return a

        class CachingSource(s.CachingMixin, Source):
            _cache = {}

        src1 = CachingSource('spam', 0.25)
        src2 = CachingSource('spam', 0.75)
        size = (2, 3, 3)
        _ = src1.fill(size)

        # Run test.
        result = src2.fill(size)

        # Extract actual result from test.
        act = grayscale_to_ints_list(result)

        # Determine if test passed.
        self.assertListEqual(exp, act)


class OctaveTestCases(ut.TestCase):
    def test_octavecosinecurtains_fill(self):
        """Given the size of a volume to generate, fill the space
        with octave cosine curtain noise and return it.
        """
        # Expected values.
        exp = [
            [
                [0x11, 0xad, 0xc7, 0x87, 0xe5, 0x5d, 0x5e, 0x68,],
                [0x11, 0xad, 0xc7, 0x87, 0xe5, 0x5d, 0x5e, 0x68,],
                [0x11, 0xad, 0xc7, 0x87, 0xe5, 0x5d, 0x5e, 0x68,],
                [0x11, 0xad, 0xc7, 0x87, 0xe5, 0x5d, 0x5e, 0x68,],
                [0x11, 0xad, 0xc7, 0x87, 0xe5, 0x5d, 0x5e, 0x68,],
                [0x11, 0xad, 0xc7, 0x87, 0xe5, 0x5d, 0x5e, 0x68,],
                [0x11, 0xad, 0xc7, 0x87, 0xe5, 0x5d, 0x5e, 0x68,],
                [0x11, 0xad, 0xc7, 0x87, 0xe5, 0x5d, 0x5e, 0x68,],
            ],
            [
                [0x78, 0x9f, 0xbb, 0xc6, 0x6c, 0x77, 0xd7, 0x30,],
                [0x78, 0x9f, 0xbb, 0xc6, 0x6c, 0x77, 0xd7, 0x30,],
                [0x78, 0x9f, 0xbb, 0xc6, 0x6c, 0x77, 0xd7, 0x30,],
                [0x78, 0x9f, 0xbb, 0xc6, 0x6c, 0x77, 0xd7, 0x30,],
                [0x78, 0x9f, 0xbb, 0xc6, 0x6c, 0x77, 0xd7, 0x30,],
                [0x78, 0x9f, 0xbb, 0xc6, 0x6c, 0x77, 0xd7, 0x30,],
                [0x78, 0x9f, 0xbb, 0xc6, 0x6c, 0x77, 0xd7, 0x30,],
                [0x78, 0x9f, 0xbb, 0xc6, 0x6c, 0x77, 0xd7, 0x30,],
            ],
        ]

        # Set up test data and state.
        kwargs = {
            'octaves': 4,
            'persistence': 8,
            'amplitude': 8,
            'frequency': 2,
            'unit': (4, 4, 4),
            'ease': 'l',
            'table': P,
        }
        obj = s.OctaveCosineCurtains(**kwargs)

        # Run test.
        result = obj.fill((2, 8, 8))

        # Extract actual values.
        result = np.around(result * 0xff).astype(int)
        act = result.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_octaveperlinnoise_fill(self):
        """Given the size of a space to fill, PerlinNoise.fill should
        return a np.array of that shape filled with noise.
        """
        # Expected data.
        exp = [
            [
                [0x80, 0x70, 0x7c, 0x8b,],
                [0x78, 0x79, 0x7e, 0x82,],
                [0x7c, 0x89, 0x80, 0x7e,],
                [0x76, 0x80, 0x86, 0x7f,],
            ],
        ]

        # Set up test data and state.
        size = (1, 4, 4)
        start = (4, 0, 0)
        kwargs = {
            'octaves': 4,
            'persistence': 8,
            'amplitude': 8,
            'frequency': 2,
            'unit': (8, 8, 8),
            'ease': '',
            'table': P,
        }
        cls = s.OctavePerlin

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size, start)


class PatternTestCase(ut.TestCase):
    def test_gradient_fill(self):
        """Given the size of a space to fill with noise, return an
        array of that size filled with noise.
        """
        # Expected values.
        exp = [
            [
                [0x00, 0x00, 0x00, 0x00],
                [0x80, 0x80, 0x80, 0x80],
                [0xff, 0xff, 0xff, 0xff],
                [0x80, 0x80, 0x80, 0x80],
                [0x00, 0x00, 0x00, 0x00],
            ],
            [
                [0x00, 0x00, 0x00, 0x00],
                [0x80, 0x80, 0x80, 0x80],
                [0xff, 0xff, 0xff, 0xff],
                [0x80, 0x80, 0x80, 0x80],
                [0x00, 0x00, 0x00, 0x00],
            ],
        ]

        # Set up test data and state.
        kwargs = {
            'direction': 'v',
            'stops': [0., 0., .5, 1., 1., 0.],
            'ease': 'l',
        }
        cls = s.Gradient
        size = (2, 5, 4)

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size)

    def test_lines_class(self):
        """An instance of noise.LineNoise should be initiated with
        the given attributes.
        """
        # Expected values.
        exp_cls = s.Lines
        exp_attrs = {
            'type': 'lines',
            'direction': 'h',
            'length': 10,
            'ease': 'i5',
        }

        # Set up test data and state.
        attrs = deepcopy(exp_attrs)
        del attrs['type']

        # Perform test.
        act_obj = exp_cls(**attrs)
        act_attrs = act_obj.asdict()

        # Determine if test passed.
        self.assertIsInstance(act_obj, exp_cls)
        self.assertDictEqual(exp_attrs, act_attrs)

    def test_lines_fill(self):
        """Given the size of a space to fill with noise, return an
        array of that size filled with noise.
        """
        # Expected values.
        exp = [
            [
                [0x00, 0x00, 0x00, 0x00],
                [0x80, 0x80, 0x80, 0x80],
                [0xff, 0xff, 0xff, 0xff],
                [0x80, 0x80, 0x80, 0x80],
            ],
            [
                [0x80, 0x80, 0x80, 0x80],
                [0xff, 0xff, 0xff, 0xff],
                [0x80, 0x80, 0x80, 0x80],
                [0x00, 0x00, 0x00, 0x00],
            ],
        ]

        # Set up test data and state.
        kwargs = {
            'direction': 'h',
            'length': 5,
            'ease': 'io3',
        }
        cls = s.Lines
        size = (2, 4, 4)

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size)

    def test_rays_fill(self):
        """Given a size and location, Ray.fill should return a
        volume filled with rays emanating from a central point.
        """
        # Expected value.
        exp = [
            [
                [0x8f, 0x51, 0x13, 0x04, 0x45, 0xa7, 0xe8, 0xfe],
                [0xc9, 0x8f, 0x36, 0x00, 0x58, 0xd3, 0xfe, 0xf6],
                [0xf9, 0xe0, 0x8f, 0x06, 0x87, 0xfe, 0xe8, 0xc2],
                [0xf1, 0xf9, 0xff, 0x8f, 0xfe, 0xa5, 0x76, 0x61],
                [0x9e, 0x89, 0x5a, 0x01, 0x70, 0x00, 0x06, 0x0e],
                [0x3d, 0x17, 0x01, 0x78, 0xf9, 0x70, 0x1f, 0x06],
                [0x09, 0x01, 0x2c, 0xa7, 0xff, 0xc9, 0x70, 0x36],
                [0x01, 0x17, 0x58, 0xba, 0xfb, 0xec, 0xae, 0x70],
            ],
        ]

        # Set up test data and state.
        kwargs = {
            'count': 3,
            'offset': np.pi / 2,
            'ease': 'ios',
        }
        cls = s.Rays
        size = (1, 8, 8)

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size)

    def test_ring_fill(self):
        """Given a size and location, Ring.fill should return a
        volume filled with concentric rings.
        """
        # Expected value.
        exp = [
            [
                [0x50, 0x00, 0x0e, 0xc0, 0xff, 0xc0, 0x0e, 0x00],
                [0x00, 0x83, 0x36, 0x00, 0x00, 0x00, 0x36, 0x83],
                [0x0e, 0x36, 0x00, 0x87, 0xff, 0x87, 0x00, 0x36],
                [0xc0, 0x00, 0x87, 0x00, 0x00, 0x00, 0x87, 0x00],
                [0xff, 0x00, 0xff, 0x00, 0x00, 0x00, 0xff, 0x00],
                [0xc0, 0x00, 0x87, 0x00, 0x00, 0x00, 0x87, 0x00],
                [0x0e, 0x36, 0x00, 0x87, 0xff, 0x87, 0x00, 0x36],
                [0x00, 0x83, 0x36, 0x00, 0x00, 0x00, 0x36, 0x83],
            ],
        ]

        # Set up test data and state.
        kwargs = {
            'radius': 2,
            'width': 1,
            'gap': 2,
            'count': 3,
            'ease': 'l'
        }
        cls = s.Ring
        size = (1, 8, 8)

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size)

    def test_solid_fill(self):
        """Given a size and location, Solid.fill should return a
        volume filled with a single color.
        """
        # Expected values.
        exp = [
            [
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
            ],
            [
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
                [0x40, 0x40, 0x40, 0x40],
            ],
        ]

        # Set up test data and state.
        kwargs = {
            'color': .25,
        }
        cls = s.Solid
        size = (2, 4, 4)

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size)

    def test_spheres_fill_x(self):
        """Given a size and location, Spheres.fill should return a
        volume filled a radial gradient.
        """
        # Expected values.
        exp = [
            [
                [0x2f, 0x42, 0x53, 0x60, 0x69, 0x6c, 0x69, 0x60,],
                [0x42, 0x58, 0x6c, 0x7b, 0x86, 0x89, 0x86, 0x7b,],
                [0x53, 0x6c, 0x82, 0x95, 0xa2, 0xa7, 0xa2, 0x95,],
                [0x60, 0x7b, 0x95, 0xac, 0xbd, 0xc4, 0xbd, 0xac,],
                [0x69, 0x86, 0xa2, 0xbd, 0xd5, 0xe2, 0xd5, 0xbd,],
                [0x6c, 0x89, 0xa7, 0xc4, 0xe2, 0xff, 0xe2, 0xc4,],
                [0x69, 0x86, 0xa2, 0xbd, 0xd5, 0xe2, 0xd5, 0xbd,],
                [0x60, 0x7b, 0x95, 0xac, 0xbd, 0xc4, 0xbd, 0xac,],
            ],
            [
                [0x2d, 0x40, 0x51, 0x5e, 0x66, 0x69, 0x66, 0x5e,],
                [0x40, 0x56, 0x69, 0x78, 0x82, 0x86, 0x82, 0x78,],
                [0x51, 0x69, 0x7f, 0x91, 0x9d, 0xa2, 0x9d, 0x91,],
                [0x5e, 0x78, 0x91, 0xa7, 0xb7, 0xbd, 0xb7, 0xa7,],
                [0x66, 0x82, 0x9d, 0xb7, 0xcc, 0xd5, 0xcc, 0xb7,],
                [0x69, 0x86, 0xa2, 0xbd, 0xd5, 0xe2, 0xd5, 0xbd,],
                [0x66, 0x82, 0x9d, 0xb7, 0xcc, 0xd5, 0xcc, 0xb7,],
                [0x5e, 0x78, 0x91, 0xa7, 0xb7, 0xbd, 0xb7, 0xa7,],
            ],
        ]

        # Set up test data and state.
        args = ['5', 'x', 'l']
        n = s.Spheres(*args)
        size = (2, 8, 8)

        # Run test.
        values = n.fill(size)

        # Extract actual values.
        values = np.around(values * 0xff).astype(int)
        act = values.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_spheres_fill_y(self):
        """Given a size and location, Spheres.fill should return a
        volume filled a radial gradient.
        """
        # Expected values.
        exp = [
            [
                [0x6c, 0x89, 0xa7, 0xc4, 0xe2, 0xff, 0xe2, 0xc4,],
                [0x69, 0x86, 0xa2, 0xbd, 0xd5, 0xe2, 0xd5, 0xbd,],
                [0x60, 0x7b, 0x95, 0xac, 0xbd, 0xc4, 0xbd, 0xac,],
                [0x53, 0x6c, 0x82, 0x95, 0xa2, 0xa7, 0xa2, 0x95,],
                [0x42, 0x58, 0x6c, 0x7b, 0x86, 0x89, 0x86, 0x7b,],
                [0x2f, 0x42, 0x53, 0x60, 0x69, 0x6c, 0x69, 0x60,],
                [0x42, 0x58, 0x6c, 0x7b, 0x86, 0x89, 0x86, 0x7b,],
                [0x53, 0x6c, 0x82, 0x95, 0xa2, 0xa7, 0xa2, 0x95,],
            ],
            [
                [0x69, 0x86, 0xa2, 0xbd, 0xd5, 0xe2, 0xd5, 0xbd,],
                [0x66, 0x82, 0x9d, 0xb7, 0xcc, 0xd5, 0xcc, 0xb7,],
                [0x5e, 0x78, 0x91, 0xa7, 0xb7, 0xbd, 0xb7, 0xa7,],
                [0x51, 0x69, 0x7f, 0x91, 0x9d, 0xa2, 0x9d, 0x91,],
                [0x40, 0x56, 0x69, 0x78, 0x82, 0x86, 0x82, 0x78,],
                [0x2d, 0x40, 0x51, 0x5e, 0x66, 0x69, 0x66, 0x5e,],
                [0x40, 0x56, 0x69, 0x78, 0x82, 0x86, 0x82, 0x78,],
                [0x51, 0x69, 0x7f, 0x91, 0x9d, 0xa2, 0x9d, 0x91,],
            ],
        ]

        # Set up test data and state.
        args = ['5', 'y', 'l']
        n = s.Spheres(*args)
        size = (2, 8, 8)

        # Run test.
        values = n.fill(size)

        # Extract actual values.
        values = np.around(values * 0xff).astype(int)
        act = values.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_spot_fill(self):
        """Given a size and location, Spot.fill should return a
        volume filled with a spot of color.
        """
        # Expected values.
        exp = [
            [
                [0x18, 0x32, 0x4c, 0x5f, 0x65, 0x5f, 0x4c, 0x32],
                [0x32, 0x58, 0x7c, 0x95, 0x9e, 0x95, 0x7c, 0x58],
                [0x4c, 0x7c, 0xa7, 0xc5, 0xd0, 0xc5, 0xa7, 0x7c],
                [0x5f, 0x95, 0xc5, 0xe7, 0xf3, 0xe7, 0xc5, 0x95],
                [0x65, 0x9e, 0xd0, 0xf3, 0xff, 0xf3, 0xd0, 0x9e],
                [0x5f, 0x95, 0xc5, 0xe7, 0xf3, 0xe7, 0xc5, 0x95],
                [0x4c, 0x7c, 0xa7, 0xc5, 0xd0, 0xc5, 0xa7, 0x7c],
                [0x32, 0x58, 0x7c, 0x95, 0x9e, 0x95, 0x7c, 0x58],
            ],
        ]

        # Set up test data and state.
        kwargs = {
            'radius': 5,
            'ease': 'ios'
        }
        cls = s.Spot
        size = (1, 8, 8)

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size)

    def test_waves_fill(self):
        """Waves.fill should return a series of concentric rings."""
        # Expected value.
        exp = [
            [
                [0x4c, 0x22, 0x76, 0xa4, 0xa4, 0x76, 0x22, 0x4c],
                [0x22, 0xa4, 0xf1, 0xb2, 0xb2, 0xf1, 0xa4, 0x22],
                [0x76, 0xf1, 0x6a, 0x0e, 0x0e, 0x6a, 0xf1, 0x76],
                [0xa4, 0xb2, 0x0e, 0x87, 0x87, 0x0e, 0xb2, 0xa4],
                [0xa4, 0xb2, 0x0e, 0x87, 0x87, 0x0e, 0xb2, 0xa4],
                [0x76, 0xf1, 0x6a, 0x0e, 0x0e, 0x6a, 0xf1, 0x76],
                [0x22, 0xa4, 0xf1, 0xb2, 0xb2, 0xf1, 0xa4, 0x22],
                [0x4c, 0x22, 0x76, 0xa4, 0xa4, 0x76, 0x22, 0x4c],
            ],
            [
                [0x4c, 0x22, 0x76, 0xa4, 0xa4, 0x76, 0x22, 0x4c],
                [0x22, 0xa4, 0xf1, 0xb2, 0xb2, 0xf1, 0xa4, 0x22],
                [0x76, 0xf1, 0x6a, 0x0e, 0x0e, 0x6a, 0xf1, 0x76],
                [0xa4, 0xb2, 0x0e, 0x87, 0x87, 0x0e, 0xb2, 0xa4],
                [0xa4, 0xb2, 0x0e, 0x87, 0x87, 0x0e, 0xb2, 0xa4],
                [0x76, 0xf1, 0x6a, 0x0e, 0x0e, 0x6a, 0xf1, 0x76],
                [0x22, 0xa4, 0xf1, 0xb2, 0xb2, 0xf1, 0xa4, 0x22],
                [0x4c, 0x22, 0x76, 0xa4, 0xa4, 0x76, 0x22, 0x4c],
            ],
        ]

        # Set up test data and state.
        cls = s.Waves
        kwargs = {
            'length': 3,
            'growth': 'l',
            'ease': '',
        }
        size = (2, 8, 8)

        # Run test and determine if passed.
        source_fill_test(self, exp, cls, kwargs, size)


class RandomTestCase(ut.TestCase):
    def test_random_fill(self):
        """Given a size and a location, Random.fill should return a
        space filled with random noise that is centered around a given
        midpoint.
        """
        # Expected value.
        exp = [
            [
                [0x8b, 0x93, 0x8d, 0x84, 0x73, 0x68, 0x8e, 0x82],
                [0x8a, 0x6a, 0x6c, 0x6c, 0x72, 0x6f, 0x68, 0x83],
                [0x88, 0x8a, 0x85, 0x77, 0x97, 0x8b, 0x8b, 0x6c],
                [0x76, 0x82, 0x94, 0x92, 0x7f, 0x8e, 0x76, 0x6f],
                [0x6a, 0x8d, 0x8b, 0x74, 0x8e, 0x74, 0x66, 0x89],
                [0x75, 0x96, 0x7a, 0x82, 0x97, 0x8c, 0x87, 0x68],
                [0x82, 0x6d, 0x97, 0x7d, 0x85, 0x6f, 0x88, 0x82],
                [0x85, 0x88, 0x8b, 0x92, 0x68, 0x91, 0x81, 0x8e],
            ],
        ]

        # Set up test data and state.
        kwargs = {
            'mid': .5,
            'scale': .1,
            'seed': 'spam',
            'ease': 'l',
        }
        cls = s.Random
        size = (1, 8, 8)

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size)

    def test_seededrandom_fill(self):
        """When given the size of an array, return an array that
        contains randomly generated noise.
        """
        exp = [
            [
                [0xb7, 0xe0, 0xc5, 0x95, 0x3f, 0x0b, 0xc7, 0x8d],
                [0xb2, 0x15, 0x20, 0x1f, 0x3e, 0x2f, 0x09, 0x92],
                [0xaa, 0xb6, 0x9d, 0x57, 0xf5, 0xb7, 0xbb, 0x1d],
                [0x52, 0x8a, 0xe5, 0xdb, 0x7d, 0xc7, 0x52, 0x2c],
                [0x16, 0xc5, 0xb9, 0x46, 0xca, 0x45, 0x02, 0xaf],
                [0x49, 0xef, 0x63, 0x8b, 0xf7, 0xbd, 0xa5, 0x0a],
                [0x8d, 0x21, 0xf7, 0x72, 0x9a, 0x2c, 0xaa, 0x8b],
                [0x9a, 0xa8, 0xba, 0xda, 0x0b, 0xd8, 0x86, 0xc9],
            ],
            [
                [0xe4, 0x10, 0xc0, 0xf4, 0xf6, 0x18, 0xf4, 0x94],
                [0xd7, 0x73, 0x80, 0xd2, 0x6b, 0xc8, 0x5d, 0xee],
                [0xb8, 0xcf, 0x10, 0x28, 0x7e, 0x7f, 0xe5, 0xfd],
                [0x5d, 0x91, 0xb5, 0x01, 0x78, 0x02, 0x5e, 0x1c],
                [0x05, 0x20, 0xb8, 0x23, 0x51, 0xc3, 0x67, 0x45],
                [0x94, 0x13, 0x72, 0x00, 0x68, 0x22, 0x63, 0xa5],
                [0x67, 0x7a, 0x77, 0xa6, 0xf9, 0xcf, 0x47, 0xc2],
                [0xe7, 0x73, 0xa0, 0xa6, 0xb5, 0x17, 0x05, 0x4c],
            ],
        ]
        src_class = s.SeededRandom
        src_kwargs = {'seed': 'spam'}
        size = (2, 8, 8)
        source_fill_test(self, exp, src_class, src_kwargs, size)

    def test_seededrandom_with_seed_repeats_noise(self):
        """When given the same seed, two instances of SeededRandom
        should return the same noise.
        """
        # Set up for expected values.
        seed = 'spam'
        size = (2, 8, 8)
        src_a = s.SeededRandom(seed)
        result = src_a.fill(size)

        # Expected value.
        exp = grayscale_to_ints_list(result)

        # Set up test data and state.
        src_b = s.SeededRandom(seed)

        # Run test.
        result = src_b.fill(size)

        # Extract actual test results.
        act = grayscale_to_ints_list(result)

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_seededrandom_without_seed_not_repeat_noise(self):
        """When given the same seed, two instances of SeededRandom
        should return the same noise.
        """
        # Set up for expected values.
        seed_exp = 'spam'
        size = (2, 8, 8)
        src_a = s.SeededRandom(seed_exp)
        result = src_a.fill(size)

        # Expected value.
        exp = grayscale_to_ints_list(result)

        # Set up test data and state.
        seed_act = 'eggs'
        src_b = s.SeededRandom(seed_act)

        # Run test.
        result = src_b.fill(size)

        # Extract actual test results.
        act = grayscale_to_ints_list(result)

        # Determine if test passed.
        self.assertNotEqual(exp, act)

    def test_embers_fill(self):
        """Given a size and a location, Embers.fill should fill the
        space with an amount of points or small dots of color that
        look like burning embers or stars.
        """
        # Expected value.
        exp = [
            [
                [0x00, 0x00, 0x00, 0x00, 0x3c, 0xb4, 0xc0, 0xc0],
                [0x00, 0x25, 0x4f, 0x1a, 0x22, 0x65, 0x6c, 0x6c],
                [0x00, 0x4f, 0xa9, 0x38, 0xc0, 0x00, 0x00, 0x00],
                [0x00, 0x1a, 0x38, 0x13, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x00],
                [0x00, 0x00, 0xc1, 0x00, 0x22, 0x66, 0x30, 0x00],
                [0x00, 0x00, 0x00, 0x00, 0x3d, 0xb6, 0x55, 0x00],
            ],
        ]

        # Set up test data and state.
        kwargs = {
            'depth': 2,
            'threshold': .95,
            'blend': 'lighter',
            'seed': 'spam',
            'ease': 'l',
        }
        cls = s.Embers
        size = (1, 8, 8)

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size)


class UnitNoiseTestCase(ut.TestCase):
    def test_curtains_makes_noise(self):
        """Given the size of each dimension of the noise,
        generators.Value.fill should return a space of that size
        filled with noise.
        """
        # Expected values.
        exp = [
            [
                [0x11, 0x2a, 0x44, 0x5e, 0x77, 0x7e, 0x85, 0x8c,],
                [0x11, 0x2a, 0x44, 0x5e, 0x77, 0x7e, 0x85, 0x8c,],
                [0x11, 0x2a, 0x44, 0x5e, 0x77, 0x7e, 0x85, 0x8c,],
                [0x11, 0x2a, 0x44, 0x5e, 0x77, 0x7e, 0x85, 0x8c,],
                [0x11, 0x2a, 0x44, 0x5e, 0x77, 0x7e, 0x85, 0x8c,],
                [0x11, 0x2a, 0x44, 0x5e, 0x77, 0x7e, 0x85, 0x8c,],
                [0x11, 0x2a, 0x44, 0x5e, 0x77, 0x7e, 0x85, 0x8c,],
                [0x11, 0x2a, 0x44, 0x5e, 0x77, 0x7e, 0x85, 0x8c,],
            ],
            [
                [0x3a, 0x52, 0x69, 0x80, 0x97, 0x94, 0x92, 0x8f,],
                [0x3a, 0x52, 0x69, 0x80, 0x97, 0x94, 0x92, 0x8f,],
                [0x3a, 0x52, 0x69, 0x80, 0x97, 0x94, 0x92, 0x8f,],
                [0x3a, 0x52, 0x69, 0x80, 0x97, 0x94, 0x92, 0x8f,],
                [0x3a, 0x52, 0x69, 0x80, 0x97, 0x94, 0x92, 0x8f,],
                [0x3a, 0x52, 0x69, 0x80, 0x97, 0x94, 0x92, 0x8f,],
                [0x3a, 0x52, 0x69, 0x80, 0x97, 0x94, 0x92, 0x8f,],
                [0x3a, 0x52, 0x69, 0x80, 0x97, 0x94, 0x92, 0x8f,],
            ],
        ]

        # Set up test data and state.
        args = ['4,4,4', P, 'l']
        obj = s.Curtains(*args)

        # Run test.
        result = obj.fill((2, 8, 8))

        # Extract actual values.
        result = np.around(result * 0xff).astype(int)
        act = result.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_perlin_fill(self):
        """Given the size of a space to fill, Perlin.fill should
        return a np.array of that shape filled with noise.
        """
        # Expected value.
        exp = [[
            [0x9f, 0x8e, 0x77, 0x60],
            [0xa5, 0x94, 0x7d, 0x65],
            [0x9f, 0x90, 0x7c, 0x68],
            [0x8b, 0x81, 0x74, 0x67],
        ],]

        # Set up test data and state.
        size = (1, 4, 4)
        start = (4, 0, 0)
        kwargs = {
            'unit': (8, 8, 8),
            'ease': '',
            'table': P,
        }
        cls = s.Perlin

        # Run test.
        source_fill_test(self, exp, cls, kwargs, size, start)

    def test_unitnoise_seeds_table_creation(self):
        """When initialized with a seed value, UnitNoise should use
        that value to seed the random generation of its table.
        """
        # Set up expected values.
        class Spam(s.UnitNoise):
            def fill(*args, **kwargs):
                return None

        kwargs = {
            'unit': (1024, 1024, 1024),
            'seed': 'spam',
        }
        exp_obj = Spam(**kwargs)

        # Expected value.
        exp = exp_obj.table.tolist()

        # Run test.
        act_obj = Spam(**kwargs)

        # Extract actual data.
        act = act_obj.table.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_unitnoise_diff_seeds_diff_table(self):
        """If you pass different seeds to two different UnitNoise
        objects, their tables will be different.
        """
        # Set up expected values.
        class Spam(s.UnitNoise):
            def fill(*args, **kwargs):
                return None

        kwargs = {
            'unit': (1024, 1024, 1024),
            'seed': 'spam',
        }
        exp_obj = Spam(**kwargs)

        # Expected value.
        exp = exp_obj.table.tolist()

        # Set up test data and state.
        kwargs_act = {
            'unit': kwargs['unit'],
            'seed': 'eggs',
        }

        # Run test.
        act_obj = Spam(**kwargs_act)

        # Extract actual data.
        act = act_obj.table.tolist()

        # Determine if test passed.
        self.assertNotEqual(exp, act)

    def test_unitnoise_serializes_seed_not_table(self):
        """If the UnitNoise object was given a seed,
        UnitNoise.asdict() should serialize the seed
        rather than the entire table.
        """
        # Expected value.
        exp = {
            'ease': 'l',
            'unit': (1024, 1024, 1024),
            'seed': 'spam',
            'type': 'spam',
        }

        # Set up test data and state.
        class Spam(s.UnitNoise):
            def fill(*args, **kwargs):
                return None

        attrs = {
            'unit': exp['unit'],
            'seed': exp['seed']
        }
        obj = Spam(**attrs)

        # Run test.
        act = obj.asdict()

        # Determine if test passed.
        self.assertDictEqual(exp, act)

    def test_values_fill_with_noise(self):
        """Given the size of each dimension of the noise,
        Values.fill should return an array that contains
        the expected noise.
        """
        # Expected values.
        exp = [[
            [0x00, 0x40, 0x7f, 0xbf, 0xff,],
            [0x00, 0x40, 0x7f, 0xbf, 0xff,],
            [0x00, 0x40, 0x7f, 0xbf, 0xff,],
        ],]

        # Set up test data and state.
        size = (1, 3, 5)
        table = [
            [
                [0x00, 0x7f, 0xff, 0xff,],
                [0x00, 0x7f, 0xff, 0xff,],
                [0x00, 0x7f, 0xff, 0xff,],
                [0x00, 0x7f, 0xff, 0xff,],
            ],
            [
                [0x00, 0x7f, 0xff, 0xff,],
                [0x00, 0x7f, 0xff, 0xff,],
                [0x00, 0x7f, 0xff, 0xff,],
                [0x00, 0x7f, 0xff, 0xff,],
            ],
            [
                [0x00, 0x7f, 0xff, 0xff,],
                [0x00, 0x7f, 0xff, 0xff,],
                [0x00, 0x7f, 0xff, 0xff,],
                [0x00, 0x7f, 0xff, 0xff,],
            ],
        ]
        unit = (2, 2, 2)
        obj = s.Values(table=table, ease='l', unit=unit)

        # Perform test.
        result = obj.fill(size)

        # Extract actual values.
        result = np.around(result * 0xff).astype(int)
        act = result.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)
