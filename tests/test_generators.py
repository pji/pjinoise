"""
test_generators
~~~~~~~~~~~~~~~

Unit tests for the pjinoise.generator module.
"""
from copy import deepcopy
import unittest as ut
from unittest.mock import call, patch

import numpy as np

from pjinoise.constants import P
from pjinoise import generators as g


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
        args = [octaves, persistence, amplitude, frequency, unit, ease, table]
        obj = g.OctaveCosineCurtains(*args)
        
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
        exp = [[
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
        args = ['v', 'l', 0., 0., .5, 1., 1., 0.]
        n = g.Gradient(*args)
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
        exp_cls = g.Lines
        exp_attrs = {
            'type': 'Lines',
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
        exp = [[
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
        n = g.Lines(**attrs)
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
        exp = [[
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
        n = g.Solid(**attrs)
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
        exp = [[
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
        args = ['5', 'l', 'x']
        n = g.Spheres(*args)
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
        exp = [[
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
        args = ['5', 'l', 'y']
        n = g.Spheres(*args)
        size = (2, 8, 8)
        
        # Run test.
        values = n.fill(size)
        
        # Extract actual values.
        values = np.around(values * 0xff).astype(int)
        act = values.tolist()
        
        # Determine if test passed.
        self.assertListEqual(exp, act)


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
        args = ['4,4,4', 'l', P]
        obj = g.Curtains(*args)
        
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
        n = g.Curtains(unit=[32, 32])
        
        # Run test.
        act = n.table.tolist()
        act_random = mock_random.mock_calls
        
        # Determine if test passed.
        self.assertListEqual(exp, act)
        self.assertListEqual(exp_random, act_random)
    
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
        obj = g.Values(table=table, ease='l', unit=unit)
        
        # Perform test.
        result = obj.fill(size)
        
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
            [0x80, 0x70, 0x7c, 0x8b,],
            [0x78, 0x79, 0x7e, 0x82,],
            [0x7c, 0x89, 0x80, 0x7e,],
            [0x76, 0x80, 0x86, 0x7f,],
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
        obj = g.OctavePerlin(**kwargs)
        
        # Run test.
        result = obj.fill(size, start)
        
        # Extract actual result.
        result = np.around(result * 0xff).astype(int)
        act = result.tolist()[0]
        
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
        obj = g.Perlin(**kwargs)
        
        # Run test.
        result = obj.fill(size, start)
        
        # Extract actual result.
        result = np.around(result * 0xff).astype(int)
        act = result.tolist()
        
        # Determine if test passed.
        self.assertListEqual(exp, act)


if __name__ == '__main__':
    raise NotImplementedError