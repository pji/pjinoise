"""
test_noise
~~~~~~~~~~

Unit tests for the pjinoise.noise module.
"""
import unittest as ut

from pjinoise import constants
from pjinoise import noise


class PerlinTestCase(ut.TestCase):
    def test_perlin_class(self):
        """Given an x, y, and z coordinates; a permutations table, 
        and a repeat period, pjinoise.perlin should return the color 
        value for that x, y, z coordinate.
        """
        exp = 128
        p = noise.Perlin(permutation_table=constants.P)
        x, y, z = 3, 3, 0
        act = p.perlin(x, y, z)
        self.assertEqual(exp, act)
    
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
    

