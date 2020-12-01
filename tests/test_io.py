"""
test_io
~~~~~~~

Unit tests for the pjinoise.io module.
"""
import argparse
import json
import unittest as ut
from unittest.mock import call, MagicMock, mock_open, patch, PropertyMock
from typing import Sequence

import cv2
import numpy as np

from pjinoise import io
from pjinoise import model as m
from pjinoise import operations as op
from pjinoise import sources as s
from pjinoise.__version__ import __version__


# Utility functions.
def _get_cli_args_mock() -> MagicMock:
    """Get a mock object that represents command line arguments."""
    args = MagicMock()
    type(args).filename = PropertyMock(return_value=None)
    type(args).load_config = PropertyMock(return_value=None)
    type(args).size = PropertyMock(return_value=None)
    type(args).location = PropertyMock(return_value=None)
    return args


# Test cases.
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

    def test_load_config_cli_override_filename(self):
        """If a filename was passed to the CLI, override the filename
        in the loaded config with that filename.
        """
        # Expected value.
        exp = 'eggs.tiff'
        exp_format = 'TIFF'

        # Build test data and state.
        filename = 'spam.conf'
        args = _get_cli_args_mock()
        type(args).filename = PropertyMock(return_value=exp)
        type(args).load_config = PropertyMock(return_value=filename)
        image = m.Image(**{
            'source': m.Layer(**{
                'source': s.Spot(**{
                    'radius': 128,
                    'ease': 'l',
                }),
                'location': [0, 0, 0],
                'filters': [],
                'mask': None,
                'mask_filters': [],
                'blend': op.difference,
                'blend_amount': 1.0,
            }),
            'size': [1, 1280, 720],
            'filename': 'spam.jpeg',
            'format': 'JPEG',
            'mode': 'RGB',
            'framerate': None
        })
        conf = json.dumps({
            'Version': __version__,
            'Image': image.asdict()
        })
        open_mock = mock_open()
        with patch('pjinoise.io.open', open_mock, create=True):
            open_mock.return_value.read.return_value = conf

            # Run test.
            result = io.load_conf(filename, args)

        # Extract actual values from result.
        act = result.filename
        act_format = result.format

        # Determine if test passed.
        self.assertEqual(exp, act)
        self.assertEqual(exp_format, act_format)

    def test_load_config_cli_override_location(self):
        """If an image location was passed to the CLI, offset the
        locations in the loaded config with that location.
        """
        # Expected value.
        exp = [10, 10, 10]

        # Build test data and state.
        filename = 'spam.conf'
        location = [4, 5, 6]
        offset = [6, 5, 4]
        args = _get_cli_args_mock()
        type(args).location = PropertyMock(return_value=offset[::-1])
        type(args).load_config = PropertyMock(return_value=filename)
        image = m.Image(**{
            'source': m.Layer(**{
                'source': [
                    m.Layer(**{
                        'source': s.Spot(**{
                            'radius': 128,
                            'ease': 'l',
                        }),
                        'location': location,
                        'filters': [],
                        'mask': None,
                        'mask_filters': [],
                        'blend': op.replace,
                        'blend_amount': 1.0,
                    }),
                    m.Layer(**{
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
                ],
                'location': location,
                'filters': [],
                'mask': None,
                'mask_filters': [],
                'blend': op.replace,
                'blend_amount': 1.0,
            }),
            'size': [1, 1280, 720],
            'filename': 'spam.jpeg',
            'format': 'JPEG',
            'mode': 'RGB',
            'framerate': None
        })
        conf = json.dumps({
            'Version': __version__,
            'Image': image.asdict()
        })
        open_mock = mock_open()
        with patch('pjinoise.io.open', open_mock, create=True):
            open_mock.return_value.read.return_value = conf

            # Run test.
            result = io.load_conf(filename, args)

        # Extract actual values from result.
        def find_location(item):
            result = []
            if 'location' in vars(item):
                result.append(item.location)
            if '_source' in vars(item):
                if isinstance(item._source, Sequence):
                    for obj in item._source:
                        result.extend(find_location(obj))
                else:
                    result.extend(find_location(item._source))
            return result
        acts = find_location(result)
        for act in acts:

            # Determine if test passed.
            self.assertListEqual(exp, act)

    def test_load_config_cli_override_size(self):
        """If an image size was passed to the CLI, override the size
        in the loaded config with that size.
        """
        # Expected value.
        exp = [2, 8, 8]

        # Build test data and state.
        filename = 'spam.conf'
        args = _get_cli_args_mock()
        type(args).size = PropertyMock(return_value=exp[::-1])
        type(args).load_config = PropertyMock(return_value=filename)
        image = m.Image(**{
            'source': m.Layer(**{
                'source': s.Spot(**{
                    'radius': 128,
                    'ease': 'l',
                }),
                'location': [0, 0, 0],
                'filters': [],
                'mask': None,
                'mask_filters': [],
                'blend': op.difference,
                'blend_amount': 1.0,
            }),
            'size': [1, 1280, 720],
            'filename': 'spam.jpeg',
            'format': 'JPEG',
            'mode': 'RGB',
            'framerate': None
        })
        conf = json.dumps({
            'Version': __version__,
            'Image': image.asdict()
        })
        open_mock = mock_open()
        with patch('pjinoise.io.open', open_mock, create=True):
            open_mock.return_value.read.return_value = conf

            # Run test.
            result = io.load_conf(filename, args)

        # Extract actual values from result.
        act = result.size

        # Determine if test passed.
        self.assertListEqual(exp, act)

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
