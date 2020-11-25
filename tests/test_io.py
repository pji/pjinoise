"""
test_io
~~~~~~~

Unit tests for the pjinoise.io module.
"""
import json
import unittest as ut
from unittest.mock import call, mock_open, patch

import cv2
import numpy as np

from pjinoise import io
from pjinoise import model as m
from pjinoise import operations as op
from pjinoise import sources as s
from pjinoise.__version__ import __version__


class IOTestCase(ut.TestCase):
    def test_load_config_from_json_file(self):
        """Given the path of a configuration serialized as JSON, 
        deserialize and return that configuration.
        """
        # Set up for expected data.
        format = 'JPEG'
        filename = 'spam.json'
        framerate = None
        imagefile = 'spam.jpeg'
        location = [0, 0, 0]
        mode = 'RGB'
        size = [1, 1280, 720]
        
        # Expected data.
        exp_conf = m.Image(**{
            'source': m.Layer(**{
                'source': s.Spot(**{
                    'radius': 128,
                    'ease': 'l',
                }),
                'location': location,
                'filters': [],
                'mask': None,
                'mask_filters': [],
                'blend': op.difference,
                'blend_amount': 1.0,
            }),
            'size': size,
            'filename': imagefile,
            'format': format,
            'mode': mode,
            'framerate': None
        })
        exp_args = (filename, 'r')
        
        # Set up test data and state.
        conf = {
            'Version': '0.2.0',
            'Image': exp_conf.asdict(),
        }
        text = json.dumps(conf)
        open_mock = mock_open()
        with patch('pjinoise.io.open', open_mock, create=True):
            open_mock.return_value.read.return_value = text
        
            # Run test.
            act_conf = io.load_conf(filename)
        
        # Determine if test passed.
        self.assertEqual(exp_conf, act_conf)
        open_mock.assert_called_with(*exp_args)
    
    @patch('PIL.Image.Image.save')
    def test_save_grayscale_image(self, mock_save):
        """Given image configuration and image data, save the image 
        data as a TIFF file.
        """
        # Set up data for expected values.
        filename = 'spam.tiff'
        format = 'TIFF'

        # Expected value.
        exp = call(filename, format)
        
        # Set up test data and state.
        a = np.array([[
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ],]).astype(float)
        a = a / 0xff
        mode = 'L'
        
        # Run test.
        io.save_image(a, filename, format, mode)
        
        # Extract actual result.
        act = mock_save.call_args
        
        # Determine if test passed.
        self.assertTupleEqual(exp, act)
    
    
    @patch('PIL.Image.Image.save')
    def test_save_color_image(self, mock_save):
        """Given image configuration and image data, save the image 
        data as a TIFF file.
        """
        # Set up data for expected values.
        filename = 'spam.tiff'
        format = 'TIFF'

        # Expected value.
        exp = call(filename, format)
        
        # Set up test data and state.
        a = np.array([[
            [[0xa1,0xa1,0xa1,],[0x81,0x81,0x81,],[0x61,0x61,0x61,],],
            [[0xa1,0xa1,0xa1,],[0x81,0x81,0x81,],[0x61,0x61,0x61,],],
            [[0xa1,0xa1,0xa1,],[0x81,0x81,0x81,],[0x61,0x61,0x61,],],
        ],]).astype(np.float32)
        mode = 'RGB'
        
        # Run test.
        io.save_image(a, filename, format, mode)
        
        # Extract actual result.
        act = mock_save.call_args
        
        # Determine if test passed.
        self.assertTupleEqual(exp, act)
    
    
    @patch('cv2.VideoWriter')
    def test_save_video(self, mock_write):
        """Given an image and save configuration, save the image as a 
        MP4 file.
        """
        # Set up data for actual values.
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
        a = a.astype(float) / 0xff
        format = 'MP4'
        mode = 'L'
        
        # Run test.
        io.save_image(a, filename, format, mode, framerate)
        
        # Extract actual result.
        act = mock_write.mock_calls
        act[1] = [act[1][0], act[1][1][0].tolist()]
        act[2] = [act[2][0], act[2][1][0].tolist()]
        
        # Determine if test passed.
        self.assertListEqual(exp, act)    


    def test_serialize_config_to_json_file(self):
        """Given a configuration object, serialize that object to 
        file as JSON.
        """
        # Set up for expected data.
        format = 'JPEG'
        filename = 'spam.json'
        framerate = None
        imagefile = 'spam.jpeg'
        location = [0, 0, 0]
        mode = 'RGB'
        size = [1, 1280, 720]
        conf = m.Image(**{
            'source': m.Layer(**{
                'source': s.Spot(**{
                    'radius': 128,
                    'ease': 'l',
                }),
                'location': location,
                'filters': [],
                'mask': None,
                'mask_filters': [],
                'blend': op.difference,
                'blend_amount': 1.0,
            }),
            'size': size,
            'filename': imagefile,
            'format': format,
            'mode': mode,
            'framerate': None
        })
        serialized_conf = {
            'Version': __version__,
            'Image': conf.asdict()
        }
        
        # Expected values.
        exp_json = json.dumps(serialized_conf, indent=4)
        exp_args = (filename, 'w')
        
        # Set up test data and state.
        open_mock = mock_open()
        with patch('pjinoise.io.open', open_mock, create=True):
        
            # Run test.
            io.save_conf(conf)
        
        # Extract actual values.
        act_json = open_mock.return_value.write.call_args[0][0] 
        
        # Determine if test passed.
        self.assertEqual(exp_json, act_json)
        open_mock.assert_called_with(*exp_args)
