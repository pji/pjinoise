"""
test_noise
~~~~~~~~~~

Unit tests for the pjinoise.noise module.
"""
from copy import deepcopy
import unittest as ut
from unittest.mock import call, patch

import numpy as np

from pjinoise import constants
from pjinoise import ease
from pjinoise import noise


class PointTestCase(ut.TestCase):
    def test_gaussnoise_class(self):
        """An instance of noise.GaussNoise should be initiated with 
        the given attributes.
        """
        exp_cls = noise.GaussNoise
        exp_attrs = {
            'location': 0,
            'scale': 5,
        }
        act_obj = noise.GaussNoise(**exp_attrs)
        act_attrs = {
            'location': act_obj.location,
            'scale': act_obj.scale,
        }
        self.assertIsInstance(act_obj, exp_cls)
        self.assertDictEqual(exp_attrs, act_attrs)
    
    def test_gaussnoise_fill_with_noise(self):
        """Given the size of each dimension of the noise, 
        GaussNoise.fill should return an array that contains 
        the expected noise.
        """
        exp_shape = (3, 3)
        
        attrs = {
            'location': 127,
            'scale': 2,
        }
        n = noise.GaussNoise(**attrs)
        act_noise = n.fill(exp_shape)
        act_shape = act_noise.shape
        
        self.assertTupleEqual(exp_shape, act_shape)


class SolidTestCase(ut.TestCase):
    def test_solidnoise_class(self):
        """An instance of noise.SolidNoise should be initiated with 
        the given attributes.
        """
        exp_cls = noise.SolidNoise
        exp_attrs = {
            'color': 0,
        }
        act_obj = noise.SolidNoise(**exp_attrs)
        act_attrs = {
            'color': act_obj.color,
        }
        self.assertTrue(isinstance(act_obj, exp_cls))
        self.assertDictEqual(exp_attrs, act_attrs)
    
    def test_solidnoise_makes_noise(self):
        """Given a set of coordinates within a multidimensional 
        space, SolidNoise.noise should return the color defined 
        for the SolidNoise object.
        """
        exp = 128
        coords = (1, 1, 3, 8)
        obj = noise.SolidNoise(exp)
        act = obj.noise(coords)
        self.assertEqual(exp, act)
    
    def test_solidnoise_serializes_to_dict(self):
        """SolidNoise.asdict should return a dictionary representation 
        of itself.
        """
        exp = {
            'type': 'SolidNoise',
            'color': 128,
            'scale': 255,
        }
        obj = noise.SolidNoise(exp['color'])
        act = obj.asdict()
        self.assertDictEqual(exp, act)
    
    def test_gradientnoise_class(self):
        """An instance of noise.GradientNoise should be initiated with 
        the given attributes.
        """
        exp_cls = noise.GradientNoise
        exp_attrs = {
            'table': [
                [
                    [0, 127, 255],
                    [0, 127, 255],
                    [0, 127, 255],
                ],
                [
                    [0, 127, 255],
                    [0, 127, 255],
                    [0, 127, 255],
                ],
                [
                    [0, 127, 255],
                    [0, 127, 255],
                    [0, 127, 255],
                ],
            ],
            'unit': (2, 2, 2),
            'scale': 255,
        }
        act_obj = noise.GradientNoise(**exp_attrs)
        act_attrs = act_obj.asdict()
        act_attrs.pop('type', None)
        self.assertTrue(isinstance(act_obj, exp_cls))
        self.assertDictEqual(exp_attrs, act_attrs)
    
    def test_gradientnoise_noise(self):
        """Given a coordinate, noise.GradientNoise should return the 
        correct value for that coordinate."""
        exp = 63.5
        
        attrs = {
            'table': [
                [
                    [0, 127, 255],
                    [0, 127, 255],
                    [0, 127, 255],
                ],
                [
                    [0, 127, 255],
                    [0, 127, 255],
                    [0, 127, 255],
                ],
                [
                    [0, 127, 255],
                    [0, 127, 255],
                    [0, 127, 255],
                ],
            ],
            'unit': (2, 2, 2),
        }
        obj = noise.GradientNoise(**attrs)
        act = obj.noise((1, 1, 1))
        
        self.assertEqual(exp, act)
    
    def test_gradientnoise_fill_with_noise(self):
        """Given the size of each dimension of the noise, 
        GradientNoise.fill should return an array that contains 
        the expected noise.
        """
        exp = [
            [0, 63.5, 127, 191, 255],
            [0, 63.5, 127, 191, 255],
            [0, 63.5, 127, 191, 255],
        ]
        
        size = (3, 5)
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
        obj = noise.GradientNoise(table=table, unit=unit)
        act = obj.fill(size).tolist()
        
        self.assertListEqual(exp, act)
    
    @patch('random.randrange', return_value=127)
    def test_gradientnosie_make_table(self, _):
        """If not passed a color table for its vertices, 
        noise.GradientNoise should create one.
        """
        exp = [
            [127, 127, 127, 127],
            [127, 127, 127, 127],
            [127, 127, 127, 127],
            [127, 127, 127, 127],
        ]
        kwargs = {
            'size': (5, 5),
            'unit': (2, 2),
        }
        obj = noise.GradientNoise(**kwargs)
        act = obj.table.tolist()
        self.assertListEqual(exp, act)
    
    def test_linenoise_class(self):
        """An instance of noise.LineNoise should be initiated with 
        the given attributes.
        """
        # Expected values.
        exp_cls = noise.LineNoise
        exp_attrs = {
            'type': 'LineNoise',
            'max': 0xff,
            'min': 0x00,
            'direction': 'h',
            'length': 10,
            'ease': 'iq',
            'scale': 0xff,
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
    
    def test_linenoise_fill_with_noise(self):
        """Given the size of a space to fill with noise, return an 
        array of that size filled with noise.
        """
        # Expected values.
        exp = [
            [0x00, 0x00, 0x00, 0x00],
            [0x80, 0x80, 0x80, 0x80],
            [0xff, 0xff, 0xff, 0xff],
            [0x80, 0x80, 0x80, 0x80],
        ]
        
        # Set up test data and state.
        attrs = {
            'max': 0xff,
            'min': 0x00,
            'direction': 'h',
            'length': 5,
            'ease': 'ioc',
        }
        n = noise.LineNoise(**attrs)
        size = (4, 4)
        
        # Run test.
        values = n.fill(size)
        
        # Extract actual values.
        act = values.tolist()
        
        # Determine if test passed.
        self.assertListEqual(exp, act)


class ValueTestCase(ut.TestCase):
    def test_valuenoise_makes_noise(self):
        """Given the size of each dimension of the noise, 
        ValueNoise.fill should return a space of that size 
        filled with noise.
        """
        exp = [[
            [255.0, 255.0, 255.0],
            [255.0, 255.0, 255.0],
            [255.0, 255.0, 255.0],
        ],]
        
        kwargs = {
            'table': [255 for _ in range(512)],
            'unit': (2, 2),
        }
        obj = noise.ValueNoise(**kwargs)
        array = obj.fill((3, 3))
        act = array.tolist()
        
        self.assertListEqual(exp, act)
    
    @patch('random.shuffle')
    def test_valuenoise_makes_table(self, mock_random):
        """If not given a permutations table, noise.ValueNoise will 
        create one on initialization.
        """
        exp = [n for n in range(256)]
        exp.extend([n for n in range(256)])
        exp_random = [
            call(exp),
        ]
        
        n = noise.ValueNoise(unit=[32, 32])
        act = n.table.tolist()
        act_random = mock_random.mock_calls
        
        self.assertListEqual(exp, act)
        self.assertListEqual(exp_random, act_random)


class PerlinTestCase(ut.TestCase):
    def test_perlinnoise_class(self):
        """An instance of noise.PerlinNoise should be initiated with 
        the given attributes.
        """
        exp = {
            'scale': 256,
            'type': 'PerlinNoise',
            'table': constants.P,
            'unit': (1024, 1024, 1024),
        }
        n = noise.PerlinNoise(exp['unit'], exp['table'])
        act = n.asdict()
        for key in exp:
            self.assertEqual(exp[key], act[key])
    
    def test_perlinnoise_fill(self):
        """Given the size of a space to fill, PerlinNoise.fill should 
        return a np.array of that shape filled with noise.
        """
        exp = [[
            [0xa0, 0x8e, 0x78, 0x61],
            [0xa5, 0x94, 0x7d, 0x66],
            [0x9f, 0x90, 0x7d, 0x68],
            [0x8c, 0x82, 0x75, 0x67],
        ],]
        
        size = (1, 4, 4)
        start = (4, 0, 0)
        unit = (8, 8, 8)
        table = constants.P
        n = noise.PerlinNoise(unit, table)
        act = n.fill(size, start).tolist()
        
        self.assertListEqual(exp, act)
    
    def test_octacveperlinnoise_class(self):
        """An instance of noise.OctavePerlinNoise should be initiated 
        with the given attributes.
        """
        exp = {
            'scale': 256,
            'type': 'OctavePerlinNoise',
            'table': constants.P,
            'unit': (1024, 1024, 1024),
            'octaves': 8,
            'persistence': -5,
            'amplitude': 20,
            'frequency': 3,
        }
        
        attrs = exp.copy()
        attrs.pop('type')
        n = noise.OctavePerlinNoise(**attrs)
        act = n.asdict()
        for key in exp:
            self.assertEqual(exp[key], act[key])
    
    def test_octaveperlinnoise_fill(self):
        """Given the size of a space to fill, PerlinNoise.fill should 
        return a np.array of that shape filled with noise.
        """
        exp = [[
            [0x80, 0x6e, 0x80, 0x92,],
            [0x77, 0x7b, 0x80, 0x80,],
            [0x80, 0x92, 0x80, 0x77,],
            [0x77, 0x85, 0x89, 0x7b,],
        ],]
        
        size = (1, 4, 4)
        start = (4, 0, 0)
        unit = (8, 8, 8)
        table = constants.P
        n = noise.OctavePerlinNoise(unit=unit, table=table)
        act = n.fill(size, start).tolist()
        
        self.assertListEqual(exp, act)
    
    
if __name__ == '__main__':
    raise NotImplementedError