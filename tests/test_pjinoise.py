"""
test_pjinoise
~~~~~~~~~~

Unit tests for the pjinoise.pjinoise module.
"""
import json
from operator import itemgetter
import unittest as ut
from unittest.mock import call, mock_open, patch

from PIL import Image

from pjinoise import filters
from pjinoise import noise
from pjinoise import pjinoise


class PerlinGenerationTestCase(ut.TestCase):
    def test_make_animation_frames(self):
        """Given a size, a Z coordinate, a direction, a length, 
        a number of diference layers, and a sequence of noise 
        generators, pjinoise.make_noise_volume should return a 
        sequence of noise slices that travel in the given direction 
        through the noise space for a number of frames 
        equal to length.
        """
        exp = [
            (0, [
                [128, 127],
                [130, 130],
            ]),
            (1, [
                [133, 132],
                [135, 134],
            ]),
            (2, [
                [137, 135],
                [137, 136],
            ]),
        ]
        
        kwargs = {
            'size': [2, 2],
            'z': 0,
            'direction': (1, 1, 1),
            'length': 3,
            'diff_layers': 3,
            'noises': [
                noise.OctavePerlin(permutation_table=pjinoise.P),
                noise.OctavePerlin(permutation_table=pjinoise.P),
                noise.OctavePerlin(permutation_table=pjinoise.P),
            ],
        }
        volume = [slice for slice in pjinoise.make_noise_volume(**kwargs)]
        act = sorted(volume, key=itemgetter(0))
        
        self.assertListEqual(exp, act)
    
    def test_make_diff_layer(self):
        """Given an image size, a Z coordinate, a number 
        of difference layers, and a list of Noise objects, 
        pjinoise.make_diff_layers should return a list of lists 
        that contain the color values for a slice of "marbled" 
        noise.
        """
        exp = [
            [128, 127, 126, 126],
            [130, 130, 129, 128],
            [132, 132, 131, 130],
            [133, 133, 133, 132],
        ]
        kwargs = {
            'size': (4, 4),
            'z': 0,
            'diff_layers': 7,
            'noises': [
                noise.OctavePerlin(permutation_table=pjinoise.P),
                noise.OctavePerlin(permutation_table=pjinoise.P),
                noise.OctavePerlin(permutation_table=pjinoise.P),
                noise.OctavePerlin(permutation_table=pjinoise.P),
                noise.OctavePerlin(permutation_table=pjinoise.P),
                noise.OctavePerlin(permutation_table=pjinoise.P),
                noise.OctavePerlin(permutation_table=pjinoise.P),
            ]
        }
        act = pjinoise.make_diff_layers(**kwargs)
        self.assertListEqual(exp, act)
    
    @patch('pjinoise.pjinoise.make_permutations', return_value=pjinoise.P)
    def test_make_diff_layer_no_noise(self, _):
        """Given an image size, a Z coordinate, a number 
        of difference layers, and no Noise objects, 
        pjinoise.make_diff_layers should create a list of 
        Noise objects to use.
        """
        exp = [
            [128, 127, 126, 126],
            [130, 130, 129, 128],
            [132, 132, 131, 130],
            [133, 133, 133, 132],
        ]
        kwargs = {
            'size': (4, 4),
            'z': 0,
            'diff_layers': 7,
            'noises': [],
            'permutation_table': pjinoise.P
        }
        act = pjinoise.make_diff_layers(**kwargs)
        self.assertListEqual(exp, act)

    def test_make_noise_slice(self):
        """Given a matrix size, a z coordinate, a permutations table, 
        a unit cube size, a number of octaves, a persistence value, 
        an amplitude value, a frequency value, a number of difference 
        levels, and a repeat value pjinoise.make_noise_slice should 
        return a two-dimensional slice of Perlin noise as a list of 
        lists.
        """
        exp = [
            [128, 127, 126, 126],
            [130, 130, 129, 128],
            [132, 132, 131, 130],
            [133, 133, 133, 132],
        ]
        
        obj_params = {
            'permutation_table': pjinoise.P,
            'unit_cube': 1024,
            'octaves': 6,
            'persistence': -4,
            'amplitude': 24,
            'frequency': 4,
            'repeat': 0,
        }
        obj = noise.OctavePerlin(**obj_params)
        params = {
            'size': (4, 4),
            'noise_gen': obj,
            'z': 0,
        }
        act = pjinoise.make_noise_slice(**params)
        
        self.assertListEqual(exp, act)
    

class UtilityTestCase(ut.TestCase):
    def test_file_extension_should_determine_format(self):
        """Given a filename, pjinoise.get_format should return the 
        image format.
        """
        exp = 'TIFF'
        value = 'spam.tiff'
        act = pjinoise.get_format(value)
        self.assertEqual(exp, act)
    
    @patch('pjinoise.pjinoise.print')
    def test_file_type_unsupported(self, mock_print):
        """Given a filename with an unsupported file extension, 
        pjinoise.get_format should print a message saying what types 
        are supported and exit.
        """
        type = 'spam'
        supported = ', '.join(pjinoise.SUPPORTED_FORMATS)
        exp_msg = [
            call(f'The file type {type} is not supported.'),
            call(f'The supported formats are: {supported}.'),
        ]
        exp_except = SystemExit
        filename = f'eggs.{type}'
        with self.assertRaises(exp_except):
            _ = pjinoise.get_format(filename)
        act_msg = mock_print.mock_calls
        self.assertListEqual(exp_msg, act_msg)
    
    def test_return_filter_list(self):
        """Given a comma separated list, pjinoise.parse_filter_list should 
        return a list of filters to run on the image.
        """
        exp = [
            filters.cut_shadow,
            filters.pixelate,
        ]
        text = 'cut_shadow,pixelate'
        act = pjinoise.parse_filter_list(text)
        self.assertListEqual(exp, act)
    
    def test_read_image_creation_details(self):
        """Given a filename, pjinoise.read_conf should read the 
        configuration details from the file and return those 
        details as a dictionary.
        """
        filename = 'spam.conf'
        exp_conf = {
            'mode': 'L',
            'size': [4, 4],
            'diff_layers': 3,
            'autocontrast': False,
            'z': 0,
            'filters': '',
            'save_conf': True,
            'frames': 1,
            'direction': [0, 0, 1],
            'loops': 1,
            'workers': 1,
            'noises': [
                {
                    'type': 'OctavePerlin',
                    'scale': 255,
                    'permutation_table': pjinoise.P,
                    'unit_cube': 1024,
                    'repeat': 0,
                    'octaves': 6,
                    'persistence': -4,
                    'amplitude': 24,
                    'frequency': 4,
                },
                {
                    'type': 'OctavePerlin',
                    'scale': 255,
                    'permutation_table': pjinoise.P,
                    'unit_cube': 1024,
                    'repeat': 0,
                    'octaves': 6,
                    'persistence': -4,
                    'amplitude': 24,
                    'frequency': 4,
                },
                {
                    'type': 'OctavePerlin',
                    'scale': 255,
                    'permutation_table': pjinoise.P,
                    'unit_cube': 1024,
                    'repeat': 0,
                    'octaves': 6,
                    'persistence': -4,
                    'amplitude': 24,
                    'frequency': 4,
                },
            ],
        }
        exp_open = (filename,)
        
        contents = json.dumps(exp_conf, indent=4)
        open_mock = mock_open()
        with patch("pjinoise.pjinoise.open", open_mock, create=True):
            open_mock.return_value.read.return_value = contents
            act_conf = pjinoise.read_config(filename)
        
        open_mock.assert_called_with(*exp_open)
        for key in exp_conf:
            self.assertEqual(exp_conf[key], act_conf[key])
    
    def test_save_image_creation_details(self):
        """Given a dictionary of configuration settings and a file 
        name, pjinoise.save_config should save the configuration settings 
        into a JSON file based on the given file name.
        """
        name = 'spam'
        exp_conf = {
            'mode': 'L',
            'size': [4, 4],
            'diff_layers': 3,
            'autocontrast': False,
            'z': 0,
            'filters': '',
            'save_conf': True,
            'frames': 1,
            'direction': [0, 0, 1],
            'loops': 1,
            'workers': 1,
            'noises': [
                {
                    'type': 'OctavePerlin',
                    'scale': 255,
                    'permutation_table': pjinoise.P,
                    'unit_cube': 1024,
                    'repeat': 0,
                    'octaves': 6,
                    'persistence': -4,
                    'amplitude': 24,
                    'frequency': 4,
                },
                {
                    'type': 'OctavePerlin',
                    'scale': 255,
                    'permutation_table': pjinoise.P,
                    'unit_cube': 1024,
                    'repeat': 0,
                    'octaves': 6,
                    'persistence': -4,
                    'amplitude': 24,
                    'frequency': 4,
                },
                {
                    'type': 'OctavePerlin',
                    'scale': 255,
                    'permutation_table': pjinoise.P,
                    'unit_cube': 1024,
                    'repeat': 0,
                    'octaves': 6,
                    'persistence': -4,
                    'amplitude': 24,
                    'frequency': 4,
                },
            ],
        }
        exp_open = (f'{name}.conf', 'w')
        
        filename = f'{name}.tiff'
        open_mock = mock_open()
        with patch("pjinoise.pjinoise.open", open_mock, create=True):
            pjinoise.save_config(filename, exp_conf)
        
        open_mock.assert_called_with(*exp_open)
        
        text = open_mock.return_value.write.call_args[0][0]
        act_conf = json.loads(text)
        for key in exp_conf:
            self.assertEqual(exp_conf[key], act_conf[key])


if __name__ == '__main__':
    raise NotImplementedError