"""
test_generators
~~~~~~~~~~~~~~~

Unit tests for the pjinoise.generator module.
"""
from copy import deepcopy
import unittest as ut
from unittest.mock import call, patch

import numpy as np

from pjinoise import generators as g


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


class RandomTestCase(ut.TestCase):
    def test_curtains_makes_noise(self):
        """Given the size of each dimension of the noise, 
        generators.Value.fill should return a space of that size 
        filled with noise.
        """
        exp = [[
            [255.0, 255.0, 255.0],
            [255.0, 255.0, 255.0],
            [255.0, 255.0, 255.0],
        ],]
        
        args = ['2,2,1', 'l', [255 for _ in range(512)]]
        obj = g.Curtains(*args)
        array = obj.fill((3, 3))
        act = array.tolist()
        
        self.assertListEqual(exp, act)
    
    @patch('random.shuffle')
    def test_curtains_makes_table(self, mock_random):
        """If not given a permutations table, generators.Value will 
        create one on initialization.
        """
        exp = [n for n in range(256)]
        exp.extend([n for n in range(256)])
        exp_random = [
            call(exp),
        ]
        
        n = g.Curtains(unit=[32, 32])
        act = n.table.tolist()
        act_random = mock_random.mock_calls
        
        self.assertListEqual(exp, act)
        self.assertListEqual(exp_random, act_random)
    
    def test_values_fill_with_noise(self):
        """Given the size of each dimension of the noise, 
        Values.fill should return an array that contains 
        the expected noise.
        """
        exp = [[
            [0., 63.5, 127., 191., 255.],
            [0., 63.5, 127., 191., 255.],
            [0., 63.5, 127., 191., 255.],
        ],]
        
        size = (1, 3, 5)
        table = [
                [
                    [0, 127, 255, 255],
                    [0, 127, 255, 255],
                    [0, 127, 255, 255],
                    [0, 127, 255, 255],
                ],
                [
                    [0, 127, 255, 255],
                    [0, 127, 255, 255],
                    [0, 127, 255, 255],
                    [0, 127, 255, 255],
                ],
                [
                    [0, 127, 255, 255],
                    [0, 127, 255, 255],
                    [0, 127, 255, 255],
                    [0, 127, 255, 255],
                ],
            ]
        unit = (2, 2, 2)
        obj = g.Values(table=table, ease='l', unit=unit)
        act = obj.fill(size).tolist()
        
        self.assertListEqual(exp, act)
    



if __name__ == '__main__':
    raise NotImplementedError