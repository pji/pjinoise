"""
test_filters
~~~~~~~~~~~~

Unit tests for the pjinoise.filters module.
"""
import unittest as ut

import numpy as np

from pjinoise import filters


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
        act = filters.cut_shadow(test)
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
        act = filters.pixelate(matrix, size)
        self.assertListEqual(exp, act)
    
    def test_skew(self):
        """Given an array of pixel values and a slope, filter.skew 
        should return an array of those pixel values skewed on the Y 
        axis by the given slope.
        """
        exp = [
            [2, 0, 1], 
            [0, 1, 2],
            [1, 2, 0],
        ]
        
        values = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ])
        slope = 1
        act = filters.skew(values, slope).tolist()
        
        self.assertListEqual(exp, act)
    
    def test_skew_with_three_dimensions(self):
        """Given an array of pixel values and a slope, filter.skew 
        should return an array of those pixel values skewed on the Y 
        axis by the given slope.
        """
        exp = [
            [
                [2, 0, 1], 
                [0, 1, 2],
                [1, 2, 0],
            ],
            [
                [2, 0, 1], 
                [0, 1, 2],
                [1, 2, 0],
            ],
        ]
        
        values = np.array([
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ],
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ],
        ])
        slope = 1
        act = filters.skew(values, slope).tolist()
        
        self.assertListEqual(exp, act)


if __name__ == '__main__':
    raise NotImplemented