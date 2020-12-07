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
        octaves = 4
        persistence = 8
        amplitude = 8
        frequency = 2
        unit = (4, 4, 4)
        ease = 'l'
        table = P
        args = [octaves, persistence, amplitude, frequency, unit, table, ease]
        obj = s.OctaveCosineCurtains(*args)

        # Run test.
        result = obj.fill((2, 8, 8))

        # Extract actual values.
        result = np.around(result * 0xff).astype(int)
        act = result.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)


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
        args = ['v', [0., 0., .5, 1., 1., 0.], 'l']
        n = s.Gradient(*args)
        size = (2, 5, 4)

        # Run test.
        values = n.fill(size)

        # Extract actual values.
        values = np.around(values * 0xff).astype(int)
        act = values.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

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
        attrs = {
            'direction': 'h',
            'length': 5,
            'ease': 'io3',
        }
        n = s.Lines(**attrs)
        size = (2, 4, 4)

        # Run test.
        values = n.fill(size)

        # Extract actual values.
        values = np.around(values * 0xff).astype(int)
        act = values.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

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
        attrs = {
            'color': .25,
        }
        n = s.Solid(**attrs)
        size = (2, 4, 4)

        # Run test.
        values = n.fill(size)

        # Extract actual values.
        values = np.around(values * 0xff).astype(int)
        act = values.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

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

    @patch('random.shuffle')
    def test_curtains_makes_table(self, mock_random):
        """If not given a permutations table, generators.Value will
        create one on initialization.
        """
        # Set up for expected values.
        table = [n for n in range(0xff)]
        table.extend(table)

        # Expected values.
        exp = table
        exp_random = [
            call(exp),
        ]

        # Set up test data and state.
        n = s.Curtains(unit=[32, 32])

        # Run test.
        act = n.table.tolist()
        act_random = mock_random.mock_calls

        # Determine if test passed.
        self.assertListEqual(exp, act)
        self.assertListEqual(exp_random, act_random)

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
