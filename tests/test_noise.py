"""
test_noise
~~~~~~~~~~

Unit tests for the pjinoise.noise module.
"""
import numpy as np
import unittest as ut
from unittest.mock import call, patch

from pjinoise import constants
from pjinoise import noise


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


class ValueTestCase(ut.TestCase):
    def test_valuenoise_makes_noise(self):
        """Given the size of each dimension of the noise, 
        ValueNoise.fill should return a space of that size 
        filled with noise.
        """
        exp = [
            [255, 255, 255],
            [255, 255, 255],
            [255, 255, 255],
        ]
        
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
    def test_perlin_class(self):
        """Given an x, y, and z coordinates; a permutations table, 
        and a repeat period, pjinoise.perlin should return the color 
        value for that x, y, z coordinate.
        """
        exp = 128
        p = noise.Perlin(unit=(1024, 1024, 1024), table=constants.P)
        x, y, z = 3, 3, 0
        coords = (z, y, x)
        act = p.noise(coords)
        self.assertEqual(exp, act)
    
    @ut.skip
    def test_octave_perlin_class(self):
        """Given x, y, and z coordinates; a permutations table; 
        pjinoise.OctavePerlin.octave_perlin should return the color 
        value for that x, y, z coordinate.
        """
        exp = 132
        
        unit_cube = 1024
        octaves = 6
        persist = -4
        amplitude = 24
        frequency = 4
        p = noise.OctavePerlin(unit_cube=unit_cube,
                               permutation_table=constants.P,
                               octaves=octaves,
                               persistence=persist,
                               amplitude=amplitude,
                               frequency=frequency)
        x, y, z = 3, 3, 0
        act = p.octave_perlin(x, y, z)
        
        self.assertEqual(exp, act)
    

if __name__ == '__main__':
    raise NotImplementedError