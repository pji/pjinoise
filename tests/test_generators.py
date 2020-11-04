"""
test_generators
~~~~~~~~~~~~~~~

Unit tests for the pjinoise.generator module.
"""
from copy import deepcopy
import unittest as ut

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


if __name__ == '__main__':
    raise NotImplementedError