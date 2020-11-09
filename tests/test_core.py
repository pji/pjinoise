"""
test_core
~~~~~~~~~

Unit tests for the core image generation of the pjinoise module.
"""
from collections import namedtuple
import json
import unittest as ut
from unittest.mock import call, mock_open, patch
import sys

import cv2
import numpy as np
from PIL import Image

from pjinoise.constants import COLOR
from pjinoise import core
from pjinoise import generators as g
from pjinoise import operations as op


# Utility classes.
class Gen(g.ValueGenerator):
    def __init__(self, *args, **kwargs) -> None:
        self.image = np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ]).astype(int)
        self.scale = 0xff
    
    def fill(self, size, *args, **kwargs):
        image = self.image / self.scale
        if len(size) == 2:
            return image
        a = np.zeros(size)
        a[0] = image
        return a


# Test cases.
class ImageTestCase(ut.TestCase):
    def test_bake_image(self):
        """Given an array, a scale, and an image mode, return an array 
        of color values that represent the original array as an image. 
        """
        # Expected value.
        exp = [
            [
                [[0, 0x00, 0x00,], [141, 0xff, 0x80,], [141, 0xff, 0xff,],],
                [[0, 0x00, 0x00,], [141, 0xff, 0x80,], [141, 0xff, 0xff,],],
                [[0, 0x00, 0x00,], [141, 0xff, 0x80,], [141, 0xff, 0xff,],],
            ],
        ]
        
        # Set up test data and state.
        mode = 'HSV'
        scale = 0xff
        color = COLOR['b']
        a = np.array([
            [
                [0x00, 0x80, 0xff],
                [0x00, 0x80, 0xff],
                [0x00, 0x80, 0xff],
            ],
        ])
        a = a / scale
        
        # Run test.
        result = core.bake_image(a, scale, mode, color)
        
        # Extract actual value.
        result = result.astype(int)
        act = result.tolist()
        
        # Determine if test passed.
        self.assertListEqual(exp, act)
    
    def test_make_image(self):
        """Given image configuration, return the array representing 
        the image.
        """
        # Expected value.
        exp = [[
            [0x00, 0x00, 0x00, 0x00, 0x00,],
            [0x00, 0x10, 0x20, 0x30, 0x40,],
            [0x00, 0x20, 0x40, 0x60, 0x80,],
            [0x00, 0x30, 0x60, 0x91, 0xc0,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ],]
        
        # Build test data and state.
        size = (1, 5, 5)
        lfconf = [
            core.FilterConfig('rotate90', ['r',]),
        ]
        lconf = [
            core.LayerConfig('gen', {}, 'replace', (0, 0, 0), []),
            core.LayerConfig('gen', {}, 'multiply', (0, 0, 0), lfconf),
        ]
        ifconf = [
            core.FilterConfig('curve', ['l',]),
        ]
        iconf = core.ImageConfig(size, lconf, ifconf, [])
        with patch.dict(g.registered_generators, {'gen': Gen,}):
        
            # Run test.
            result = core.make_image(iconf)
        
        # Extract actual result.
        result = np.around(result * 0xff).astype(int)
        act = result.tolist()
        
        # Determine if test passed.
        self.assertListEqual(exp, act)
    
    @patch('PIL.Image.Image.save')
    def test_save_image_as_tiff(self, mock_save):
        """Given an image and save configuration, save the image as a 
        TIFF file.
        """
        # Expected value.
        filename = 'spam.tiff'
        format = 'TIFF'
        mode = 'L'
        exp = call(filename, format)
        
        # Set up test data and state.
        a = np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ]).astype(np.uint8)
        sconf = core.SaveConfig(filename, format, mode)
        
        # Run test.
        core.save_image(a, sconf)
        
        # Extract actual result.
        act = mock_save.call_args
        
        # Determine if test passed.
        self.assertTupleEqual(exp, act)
    
    @patch('cv2.VideoWriter')
    def test_save_image_as_mp4(self, mock_write):
        """Given an image and save configuration, save the image as a 
        MP4 file.
        """
        # Common data for expected and actual values.
        filename = 'spam.mp4'
        codec = 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        framerate = 12
        a = np.array([
            [
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
            ],
            [
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
            ],
        ]).astype(np.uint8)
        size = (a.shape[-1], a.shape[-2])
        
        # Expected value.
        a_exp = np.flip(a, -1)
        exp = [
            call(filename, fourcc, framerate, size, True),
            ['().write', (a_exp[0].tolist()),],
            ['().write', (a_exp[1].tolist()),],
            call().release()
        ]
        
        # Set up test data and state.
        format = 'MP4'
        mode = 'L'
        sconf = core.SaveConfig(filename, format, mode, framerate)
        
        # Run test.
        core.save_image(a, sconf)
        
        # Extract actual result.
        act = mock_write.mock_calls
        act[1] = [act[1][0], act[1][1][0].tolist()]
        act[2] = [act[2][0], act[2][1][0].tolist()]
        
        # Determine if test passed.
        self.assertListEqual(exp, act)    


class InterfaceTestCase(ut.TestCase):
    def test_load_config(self):
        """Given the name of a file containing serialized 
        configuration, load the file and deserialize the 
        configuration.
        """
        # Back up initial state.
        argv_bkp = sys.argv
        try:
        
            # Construct data for expected values.
            size = [1, 5, 5]
            lfconf = [
                core.FilterConfig('curve', ['',]),
            ]
            lconf = [
                core.LayerConfig(
                    'lines', 
                    ['v', 128, 'ioq'], 
                    'replace', 
                    [5, 0, 0],
                    []
                ),
                core.LayerConfig(
                    'lines', 
                    ['h', 128, 'ioc'], 
                    'difference', 
                    [5, 0, 0],
                    lfconf
                ),
            ]
            ifconf = [
                core.FilterConfig('rotate90', ['r',]),
            ]
            filename = 'spam.json'
        
            # Expected values.
            exp_iconf = core.ImageConfig(size, lconf, ifconf, COLOR['c'])
            exp_sconf = core.SaveConfig('spam.tiff', 'TIFF', 'RGB', None)
            exp_open = (filename, 'r')
        
            # Build test data and state.
            sys.argv = [
                'python3.8 -m pjinoise.core',
                '-c', filename,
                '-l', '0', '0', '5',
                '-o', 'spam.mp4',
            ]
            config = {
                'Version' : '0.0.1',
                'ImageConfig': {
                    'size': [1, 5, 5],
                    'layers': [
                        {
                            'generator': 'lines',
                            'args': ['v', 128, 'ioq'],
                            'mode': 'replace',
                            'location': [0, 0, 0],
                            'filters': [],
                        },
                        {
                            'generator': 'lines',
                            'args': ['h', 128, 'ioc'],
                            'mode': 'difference',
                            'location': [0, 0, 0],
                            'filters': [
                                {
                                    'filter': 'curve',
                                    'args': ['',],
                                },
                            ],
                        },
                    ],
                    'filters': [
                        {
                            'filter': 'rotate90',
                            'args': ['r',],
                        },
                    ],
                    'color': COLOR['c'],
                },
                'SaveConfig': {
                    'filename': 'spam.tiff',
                    'format': 'TIFF',
                    'mode': 'RGB',
                    'framerate': None
                },
            }
            text = json.dumps(config, indent=4)
            open_mock = mock_open()
            with patch('pjinoise.core.open', open_mock, create=True):
                open_mock.return_value.read.return_value = text
            
                # Run test.
                args = core.parse_cli_arguments()
                act_iconfs, act_sconf = core.load_config(filename, args)
            
            # Extract actual values.
            act_iconf = act_iconfs[0]
        
            # Determine if test passed.
            self.assertEqual(exp_iconf.size, act_iconf.size)
            self.assertEqual(exp_iconf.layers, act_iconf.layers)
            self.assertEqual(exp_iconf.filters, act_iconf.filters)
            self.assertEqual(exp_iconf.color, act_iconf.color)
            self.assertEqual(exp_iconf, act_iconf)
            self.assertEqual(exp_sconf, act_sconf)
            open_mock.assert_called_with(*exp_open)
        
        # Restore original state.
        finally:
            sys.argv = argv_bkp
    
    def test_parse_arguments(self):
        """Given command line arguments, create configuration to drive 
        the image generation.
        """
        # Back up initial state.
        argv_bkp = sys.argv
        try:
        
            # Set up test data for expected values.
            filter_kwargs = {
                'filter': 'rotate90',
                'args': ['r',]
            }
            filterconfig = [core.FilterConfig(**filter_kwargs),]
            layer_kwargs = [
                {
                    'generator': 'lines',
                    'args': [],
                    'mode': 'replace',
                    'location': (0, 0, 0),
                    'filters': [],
                },
                {
                    'generator': 'lines',
                    'args': [],
                    'mode': 'difference',
                    'location': (0, 0, 0),
                    'filters': filterconfig,
                },
            ]
            layerconfig = [core.LayerConfig(**kwargs) for kwargs in layer_kwargs]
            image_kwargs = {
                'size': (1, 5, 5),
                'layers': layerconfig,
                'filters': filterconfig,
                'color': COLOR['e'],
            }
            saveconfig_kwargs = {
                'filename': 'spam.mp4',
                'format': 'MP4',
                'mode': 'L',
            }
        
            # Expected values.
            exp = (
                core.ImageConfig(**image_kwargs),
                core.SaveConfig(**saveconfig_kwargs),
            )
            
            # Set up test data and state.
            sys.argv = [
                'python3.8 -m pjinoise.core',
                '-f', 'rotate90:r',
                '-k', 'e',
                '-n', 'lines__0:0:0__replace',
                '-n', 'lines__0:0:0_rotate90:r_difference',
                '-m', 'L',
                '-o', 'spam.mp4',
                '-s', '5', '5', '1',
            ]
        
            # Run test.
            args = core.parse_cli_arguments()
            result = core.make_config(args)
            
            # Extract actual value.
            act = [
                result[0][0],
                result[1],
            ]
        
            # Determine if test passed.
            self.assertEqual(exp[0].size, act[0].size)
            self.assertListEqual(exp[0].layers, act[0].layers)
            self.assertEqual(exp[0].filters, act[0].filters)
            self.assertListEqual(exp[0].color, act[0].color)
            self.assertEqual(exp[0], act[0])
            self.assertEqual(exp[1], act[1])
        
        # Restore original state.
        finally:
            sys.argv = argv_bkp
    
    def test_save_config(self):
        """Given configuration records, serialize those records to 
        disk as JSON.
        """
        # Expected value.
        exp_call = ('spam.json', 'w')
        exp_conf = {
            'Version': '0.0.2',
            'ImageConfig': [{
                'size': [1, 5, 5],
                'layers': [
                    {
                        'generator': 'lines',
                        'args': ['h', 128, 'ioq'],
                        'mode': 'replace',
                        'location': [0, 0, 0],
                        'filters': [],
                    },
                    {
                        'generator': 'lines',
                        'args': ['v', 128, 'ioc'],
                        'mode': 'replace',
                        'location': [0, 0, 0],
                        'filters': [
                            {
                                'filter': 'curve',
                                'args': ['',],
                            },
                        ],
                    },
                ],
                'filters': [
                    {
                        'filter': 'rotate90',
                        'args': ['r',],
                    },
                ],
                'color': COLOR['c'],
                'mode': 'difference',
            },],
            'SaveConfig': {
                'filename': 'spam.tiff',
                'format': 'TIFF',
                'mode': 'RGB',
                'framerate': None
            },
        }
        
        # Set up test data and state.
        lfconf = [
            core.FilterConfig(
                exp_conf['ImageConfig'][0]['layers'][1]['filters'][0]['filter'],
                exp_conf['ImageConfig'][0]['layers'][1]['filters'][0]['args'],
            ),
        ]
        lconf = [
            core.LayerConfig(
                exp_conf['ImageConfig'][0]['layers'][0]['generator'],
                exp_conf['ImageConfig'][0]['layers'][0]['args'],
                exp_conf['ImageConfig'][0]['layers'][0]['mode'],
                exp_conf['ImageConfig'][0]['layers'][0]['location'],
                exp_conf['ImageConfig'][0]['layers'][0]['filters']
            ),
            core.LayerConfig(
                exp_conf['ImageConfig'][0]['layers'][1]['generator'],
                exp_conf['ImageConfig'][0]['layers'][1]['args'],
                exp_conf['ImageConfig'][0]['layers'][1]['mode'],
                exp_conf['ImageConfig'][0]['layers'][1]['location'],
                lfconf
            ),
        ]
        ifconf = [
            core.FilterConfig(
                exp_conf['ImageConfig'][0]['filters'][0]['filter'],
                exp_conf['ImageConfig'][0]['filters'][0]['args'],
            ),
        ]
        iconfs = [core.ImageConfig(
            exp_conf['ImageConfig'][0]['size'], 
            lconf, 
            ifconf, 
            exp_conf['ImageConfig'][0]['color'], 
        ),]
        sconf = core.SaveConfig(
            exp_conf['SaveConfig']['filename'],
            exp_conf['SaveConfig']['format'],
            exp_conf['SaveConfig']['mode'],
            exp_conf['SaveConfig']['framerate'],
        )
        open_mock = mock_open()
        with patch('pjinoise.core.open', open_mock, create=True):
        
            # Run test.
            core.save_config(iconfs, sconf)
        
        # Extract actual values.
        result = open_mock.return_value.write.call_args[0][0] 
        act_conf = json.loads(result)
        
        # Determine if test passed.
        open_mock.assert_called_with(*exp_call)
        for key in exp_conf:
            self.assertEqual(exp_conf[key], act_conf[key])
    
    def test_set_image_location_with_command_line_args(self):
        """Given a location from the command line, update the location 
        of all the generators.
        """
        # Back up initial state.
        argv_bkp = sys.argv
        try:
        
            # Set up test data for expected values.
            filter_kwargs = {
                'filter': 'rotate90',
                'args': ['r',]
            }
            filterconfig = [core.FilterConfig(**filter_kwargs),]
            layer_kwargs = [
                {
                    'generator': 'lines',
                    'args': [],
                    'mode': 'replace',
                    'location': (5, 0, 0),
                    'filters': [],
                },
                {
                    'generator': 'lines',
                    'args': [],
                    'mode': 'difference',
                    'location': (5, 0, 0),
                    'filters': filterconfig,
                },
            ]
            layerconfig = [core.LayerConfig(**kwargs) for kwargs in layer_kwargs]
            image_kwargs = {
                'size': (1, 5, 5),
                'layers': layerconfig,
                'filters': filterconfig,
                'color': COLOR['e'],
            }
            saveconfig_kwargs = {
                'filename': 'spam.mp4',
                'format': 'MP4',
                'mode': 'L',
                'framerate': 29.95,
            }
        
            # Expected values.
            exp = (
                [core.ImageConfig(**image_kwargs),],
                core.SaveConfig(**saveconfig_kwargs),
            )
            
            # Set up test data and state.
            sys.argv = [
                'python3.8 -m pjinoise.core',
                '-f', 'rotate90:r',
                '-k', 'e',
                '-l', '0', '0', '5',
                '-m', 'L',
                '-n', 'lines__0:0:0__replace',
                '-n', 'lines__0:0:0_rotate90:r_difference',
                '-o', 'spam.mp4',
                '-r', '29.95',
                '-s', '5', '5', '1',
            ]
        
            # Run test.
            args = core.parse_cli_arguments()
            act = core.make_config(args)
        
            # Determine if test passed.
            self.assertEqual(exp[0][0].size, act[0][0].size)
            self.assertListEqual(exp[0][0].layers, act[0][0].layers)
            self.assertEqual(exp[0][0].filters, act[0][0].filters)
            self.assertListEqual(exp[0][0].color, act[0][0].color)
            self.assertListEqual(exp[0], act[0])
            self.assertEqual(exp[1], act[1])
        
        # Restore original state.
        finally:
            sys.argv = argv_bkp


class LayerTestCase(ut.TestCase):
    def test_make_layers(self):
        """Given layer configuration, return a list of Layer objects 
        that contain the mode and data for the layer.
        """
        # Expected value.
        image0 = np.array([[
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ],]).astype(int)
        image1 = np.array([[
            [0x00, 0x00, 0x00, 0x00, 0x00,],
            [0x40, 0x40, 0x40, 0x40, 0x40,],
            [0x80, 0x80, 0x80, 0x80, 0x80,],
            [0xc0, 0xc0, 0xc0, 0xc0, 0xc0,],
            [0xff, 0xff, 0xff, 0xff, 0xff,],
        ],]).astype(int)
        exp = [
            core.Layer(op.replace, image0 / 0xff),
            core.Layer(op.multiply, image1 / 0xff),
        ]
        
        # Build test data and state.
        fconf = [
            core.FilterConfig('rotate90', ['r']),
        ]
        lconf = [
            core.LayerConfig('gen', [], 'replace', (0, 0, 0), []),
            core.LayerConfig('gen', [], 'multiply', (0, 0, 0), fconf),
        ]
        size = (1, 5, 5)
        with patch.dict(g.registered_generators, {'gen': Gen,}):
        
            # Run test.
            layer_gen = core.make_layers(lconf, size)
            act = list(layer_gen)
        
        # Determine if test passed.
        for elayer, alayer in zip(exp, act):
            self.assertEqual(elayer.mode, alayer.mode)
            self.assertTrue(np.array_equal(elayer.data, alayer.data))
    
    def test_blend_layers(self):
        """Given a list of operations and layers, pjinoise should 
        return the result of blending those layers together.
        """
        # Expected value.
        exp = [
            [0x00, 0x20, 0x80, 0xe0, 0xff,],
            [0x00, 0x20, 0x80, 0xe0, 0xff,],
            [0x00, 0x20, 0x80, 0xe0, 0xff,],
            [0x00, 0x20, 0x80, 0xe0, 0xff,],
            [0x00, 0x20, 0x80, 0xe0, 0xff,],
        ]
        
        # Set up test data and state.
        image = np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ])
        image = image / 0xff
        layers = [
            core.Layer(op.replace, image.copy()),
            core.Layer(op.overlay, image.copy()),
        ]
        
        # Run test.
        result = core.blend_layers(layers)
        
        # Extract actual data.
        result = np.around(result * 0xff).astype(int)
        act = result.tolist()
        
        # Determine if test passed.
        self.assertListEqual(exp, act)


if __name__ == '__main__':
    raise NotImplementedError