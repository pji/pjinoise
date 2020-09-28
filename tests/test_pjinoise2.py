"""
test_pjinoise
~~~~~~~~~~~~~

Unit tests for pjinoise.pjinoise2.
"""
import unittest as ut
from unittest.mock import call, patch
import sys

import numpy as np
from PIL import Image

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
            'loops': 0,
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
            ['', [array.tolist(),], {'mode': 'L',}], 
            ['().save', [filename, format], {}],
        ]
        
        pn.CONFIG['filename'] = filename
        pn.CONFIG['format'] = format
        pn.save_image(array)
        calls = mock_fromarray.mock_calls
        act = [list(item) for item in calls]
        for item in act:
            item[1] = [thing for thing in item[1]]
        act[0][1][0] = act[0][1][0].tolist()
                
        self.assertListEqual(exp, act)
    
    
    @patch('PIL.Image.Image.save')
    @patch('PIL.Image.fromarray')
    def test_save_animation(self, mock_fromarray, mock_save):
        """Given a three dimensional numpy.array, pjinoise.save_image 
        should create PIL.Image that is an animation, with each two 
        dimensional slice being a frame, and save it to disk.
        """
        array = np.array([
            [
                [0, 127, 255], 
                [0, 127, 255]
            ],
            [
                [0, 127, 255], 
                [0, 127, 255]
            ],
        ]
        )
        filename = 'spam'
        format = 'GIF'
        loop = 0
        img = Image.fromarray(array[1])
        exp = [
            ['', [array.tolist()[0],], {}],     # Artifact from two lines up.
            ['', [array.tolist()[0],], {'mode': 'L',}], 
            ['', [array.tolist()[1],], {'mode': 'L',}], 
            ['().save', [filename], {
                'save_all': True,
                'append_images': [img,],
                'loop': loop,
            }],
        ]
        
        pn.CONFIG['filename'] = filename
        pn.CONFIG['format'] = format
        pn.CONFIG['loop'] = 0
        pn.save_image(array)
        calls = mock_fromarray.mock_calls
        act = [list(item) for item in calls]
        for item in act:
            item[1] = [thing for thing in item[1]]
        act[0][1][0] = act[0][1][0].tolist()
        act[1][1][0] = act[1][1][0].tolist()
        act[2][1][0] = act[2][1][0].tolist()
        
        self.assertListEqual(exp, act)


class NoiseTestCase(ut.TestCase):
    """Test cases for the creation of noise."""
    def test_create_single_noise_volume(self):
        """Given a noise object and a size of the noise to generate, 
        pjinoise.make_noise should return a numpy.ndarray object 
        containing the noise.
        """
        exp = [
            [0.0, 63.5, 127.0, 191.0, 255.0],
            [0.0, 63.5, 127.0, 191.0, 255.0],
            [0.0, 63.5, 127.0, 191.0, 255.0],
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
        array = pn.make_noise(obj, size)
        act = array.tolist()
        
        self.assertListEqual(exp, act)


if __name__ == '__main__':
    raise NotImplementedError