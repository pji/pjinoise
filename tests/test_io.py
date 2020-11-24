"""
test_io
~~~~~~~

Unit tests for the pjinoise.io module.
"""
import json
import unittest as ut
from unittest.mock import mock_open, patch

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
        mode = 'RGB'
        size = [1, 1280, 720]
        
        # Expected data.
        exp_conf = m.Image(**{
            'source': m.Layer(**{
                'source': s.Spot(**{
                    'radius': 128,
                    'ease': 'l',
                }),
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
    
    def test_serialize_config_to_json_file(self):
        """Given a configuration object, serialize that object to 
        file as JSON.
        """
        # Set up for expected data.
        format = 'JPEG'
        filename = 'spam.json'
        framerate = None
        imagefile = 'spam.jpeg'
        mode = 'RGB'
        size = [1, 1280, 720]
        conf = m.Image(**{
            'source': m.Layer(**{
                'source': s.Spot(**{
                    'radius': 128,
                    'ease': 'l',
                }),
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
