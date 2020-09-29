"""
test_pjinoise
~~~~~~~~~~~~~

Unit tests for pjinoise.pjinoise2.
"""
from copy import deepcopy
import json
import unittest as ut
from unittest.mock import call, mock_open, patch
import sys

import numpy as np
from PIL import Image

from pjinoise import noise
from pjinoise import pjinoise2 as pn


CONFIG = {
    'filename': 'spam.tiff',
    'format': 'TIFF',
    'loops': 0,
    'ntypes': [noise.ValueNoise,],
    'size': [3, 3],
    'unit': [2, 2],
}
CONFIG['noises'] = [CONFIG['ntypes'][0](unit=CONFIG['unit'], 
                                        table=[0 for _ in range(512)]),]


class CLITestCase(ut.TestCase):
    def test_configure_from_command_line(self):
        """When the script is invoked with command line arguments, 
        pjinoise.configure should update the script configuration 
        based on those arguments.
        """
        exp = CONFIG
        
        sys.argv = [
            'python3.8 -m pjinoise.pjinoise', 
            '-n',
            'ValueNoise',
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
        pn.CONFIG['noises'][0].table = np.array([0 for _ in range(512)])
        act = pn.CONFIG
        
        self.assertDictEqual(exp, act)


class FileTestCase(ut.TestCase):
    def test_save_configuration_file(self):
        """When called, pjinoise.save_config should write the 
        current configuration to a file.
        """
        namepart = CONFIG["filename"].split(".")[0]
        filename = f'{namepart}.conf'
        exp_conf = deepcopy(CONFIG)
        pn.CONFIG = deepcopy(CONFIG)
        exp_conf['ntypes'] = [cls.__name__ for cls in exp_conf['ntypes']]
        exp_conf['noises'][0] = exp_conf['noises'][0].asdict()
        exp_open = (filename, 'w')
        
        open_mock = mock_open()
        with patch('pjinoise.pjinoise2.open', open_mock, create=True):
            pn.save_config()
        
        open_mock.assert_called_with(*exp_open)
        
        text = open_mock.return_value.write.call_args[0][0]
        act_conf = json.loads(text)
        for key in exp_conf:
            self.assertEqual(exp_conf[key], act_conf[key])
        
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