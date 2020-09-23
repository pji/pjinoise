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

from pjinoise import pjinoise


class FilterTestCase(ut.TestCase):
    def test_cut_shadow_filter(self):
        """Given a 2D matrix of color values, pjinoise.cut_shadow should 
        set the bottom half of colors to black, then rebalance the 
        colors again.
        """
        exp = [
            [255, 255, 128, 2],
            [255, 128, 2, 0],
            [128, 2, 0, 0],
            [2, 0, 0, 0]
        ]
        test = [
            [255, 255, 191, 128],
            [255, 191, 128, 64],
            [191, 128, 64, 0],
            [128, 64, 0, 0]
        ]
        act = pjinoise.cut_shadow(test)
        self.assertListEqual(exp, act)
    
    def test_pixelate(self):
        """Given a 2D matrix of color values and a size, 
        pjinoise.average_square should create squares within 
        the matrix where the color is the average of the 
        original colors within the square.
        """
        exp = [
            [239, 239, 128, 128],
            [239, 239, 128, 128],
            [128, 128, 16, 16],
            [128, 128, 16, 16],
        ]
        matrix = [
            [255, 255, 191, 128],
            [255, 191, 128, 64],
            [191, 128, 64, 0],
            [128, 64, 0, 0]
        ]
        size = 2
        act = pjinoise.pixelate(matrix, size)
        self.assertListEqual(exp, act)


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
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
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
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
                pjinoise.OctavePerlin(permutation_table=pjinoise.P),
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
            'noises': []
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
        obj = pjinoise.OctavePerlin(**obj_params)
        params = {
            'size': (4, 4),
            'noise_gen': obj,
            'z': 0,
        }
        act = pjinoise.make_noise_slice(**params)
        
        self.assertListEqual(exp, act)
    

class PerlinTestCase(ut.TestCase):
    def test_perlin_class(self):
        """Given an x, y, and z coordinates; a permutations table, 
        and a repeat period, pjinoise.perlin should return the color 
        value for that x, y, z coordinate.
        """
        exp = 128
        p = pjinoise.Perlin(permutation_table=pjinoise.P)
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
        p = pjinoise.OctavePerlin(unit_cube=unit_cube,
                               permutation_table=pjinoise.P,
                               octaves=octaves,
                               persistence=persist,
                               amplitude=amplitude,
                               frequency=frequency)
        x, y, z = 3, 3, 0
        act = p.octave_perlin(x, y, z)
        
        self.assertEqual(exp, act)
    

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
            pjinoise.cut_shadow,
            pjinoise.pixelate,
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
