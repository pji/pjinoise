"""
test_pjinoise
~~~~~~~~~~~~~

Unit tests for pjinoise.pjinoise2.
"""
import unittest as ut
from unittest.mock import call, patch

import numpy as np

from pjinoise import pjinoise2 as pn


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