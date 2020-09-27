"""
test_pjinoise
~~~~~~~~~~~~~

Unit tests for pjinoise.pjinoise2.
"""
import unittest as ut
from unittest.mock import call, patch
import sys

import numpy as np

from pjinoise import pjinoise2 as pn


class CLITestCase(ut.TestCase):
    def test_configure_from_command_line(self):
        """When the script is invoked with command line arguments, 
        pjinoise.configure should update the script configuration 
        based on those arguments.
        """
        exp = {
            'filename': 'spam.tiff',
            'format': 'TIFF',
            'size': [64, 64],
        }
        
        sys.argv = [
            'python3.8 -m pjinoise.pjinoise', 
            '-s',
            str(exp['size'][0]),
            str(exp['size'][1]),
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


if __name__ == '__main__':
    raise NotImplementedError