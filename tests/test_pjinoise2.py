"""
test_pjinoise
~~~~~~~~~~~~~

Unit tests for pjinoise.pjinoise2.
"""
import unittest as ut
from unittest.mock import call, patch
import sys

import numpy as np

from pjinoise import noise
from pjinoise import pjinoise2 as pn


class CLITestCase(ut.TestCase):
    @patch('random.randrange', return_value=127)
    def test_configure_from_command_line(self, _):
        """When the script is invoked with command line arguments, 
        pjinoise.configure should update the script configuration 
        based on those arguments.
        """
        exp = {
            'filename': 'spam.tiff',
            'format': 'TIFF',
            'ntypes': [noise.GradientNoise,],
            'size': [3, 3],
            'unit': [2, 2],
        }
        exp['noises'] = [exp['ntypes'][0](unit=exp['unit'], size=exp['size']),]
        
        sys.argv = [
            'python3.8 -m pjinoise.pjinoise', 
            '-n',
            'GradientNoise',
            '-s',
            str(exp['size'][0]),
            str(exp['size'][1]),
            '-u',
            str(exp['unit'][0]),
            str(exp['unit'][1]),
            '-o',
            exp['filename'],
        ]
        pn.configure()
        act = pn.CONFIG
        
        self.assertDictEqual(exp, act)


class ImageFileTestCase(ut.TestCase):
    @patch('PIL.Image.Image.save')
    @patch('PIL.Image.fromarray')
    def test_save_image(self, mock_fromarray, mock_save):
        """Given a two dimensional numpy.array, pjinoise.save_image 
        should create PIL.Image for the array and save it to disk.
        """
        array = np.array([[0, 127, 255], [0, 127, 255]])
        filename = 'spam'
        format = 'TIFF'
        exp = [
            call(array), 
            call().save(filename, format),
        ]
        
        pn.CONFIG['filename'] = filename
        pn.CONFIG['format'] = format
        pn.save_image(array)
        act = mock_fromarray.mock_calls
                
        self.assertListEqual(exp, act)


class NoiseTestCase(ut.TestCase):
    """Test cases for the creation of noise."""
    def test_create_single_noise_volume(self):
        """Given a noise object and a size of the noise to generate, 
        pjinoise.make_noise should return a numpy.ndarray object 
        containing the noise.
        """
        exp = [
            [0, 63.5, 127, 190.5, 255],
            [0, 63.5, 127, 190.5, 255],
            [0, 63.5, 127, 190.5, 255],
        ]
        
        size = (3, 5)
        table = [
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
            ]
        unit = (2, 2, 2)
        obj = noise.GradientNoise(table=table, unit=unit)
        array = pn.make_noise(obj, size)
        act = array.tolist()
        
        self.assertListEqual(exp, act)


if __name__ == '__main__':
    raise NotImplementedError