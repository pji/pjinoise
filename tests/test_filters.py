"""
test_filters
~~~~~~~~~~~~

Unit tests for the pjinoise.filters module.
"""
import unittest as ut

import numpy as np
from PIL import Image

from pjinoise import filters
from pjinoise.constants import X, Y, Z


# Layer filter tests..
class ClassTestCase(ut.TestCase):
    def test_create_from_command_string(self):
        """Given the name of a filter and its parameters as a 
        comma delimited string, filters.make_filter should return 
        an initialized Filter object of the correct type.
        """
        exp_super = filters.ForLayer
        exp_cls = filters.Rotate90
        exp_param = 'r'
        
        direction = 'r'
        act = filters.make_filter('Rotate90', ('r',))
        act_param = act.direction
        
        self.assertIsInstance(act, exp_super)
        self.assertIsInstance(act, exp_cls)
        self.assertEqual(exp_param, act_param)


class CutLightTestCase(ut.TestCase):
    def test_process_cut_light(self):
        """Given an image, remove every gray below 50% and then 
        adjust the remaining grays to fill the whole gamut.
        """
        exp = [
            [0x00, 0x7f, 0xff, 0xff, 0xff,],
            [0x00, 0x7f, 0xff, 0xff, 0xff,],
            [0x00, 0x7f, 0xff, 0xff, 0xff,],
            [0x00, 0x7f, 0xff, 0xff, 0xff,],
            [0x00, 0x7f, 0xff, 0xff, 0xff,],
        ]
        
        img = np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ])
        threshold = 0x80
        f = filters.CutLight(threshold)
        act = f.process(img).tolist()
        
        self.assertListEqual(exp, act)


class CutShadowTestCase(ut.TestCase):
    def test_process_cut_shadow(self):
        """Given an image, remove every gray below 50% and then 
        adjust the remaining grays to fill the whole gamut.
        """
        exp = [
            [0x00, 0x00, 0x00, 0x80, 0xff,],
            [0x00, 0x00, 0x00, 0x80, 0xff,],
            [0x00, 0x00, 0x00, 0x80, 0xff,],
            [0x00, 0x00, 0x00, 0x80, 0xff,],
            [0x00, 0x00, 0x00, 0x80, 0xff,],
        ]
        
        img = np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ])
        threshold = 0x80
        f = filters.CutShadow(threshold)
        act = f.process(img).tolist()
        
        self.assertListEqual(exp, act)

    def test_process_cut_shadow_in_out_cubic(self):
        """Given an image, remove every gray below 50% and then 
        adjust the remaining grays to fill the whole gamut using 
        an in-out cubic easing function.
        """
        exp = [
            [0x00, 0x00, 0x10, 0x82, 0xf0, 0xff],
            [0x00, 0x00, 0x10, 0x82, 0xf0, 0xff],
            [0x00, 0x00, 0x10, 0x82, 0xf0, 0xff],
            [0x00, 0x00, 0x10, 0x82, 0xf0, 0xff],
            [0x00, 0x00, 0x10, 0x82, 0xf0, 0xff],
        ]
        
        img = np.array([
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        ])
        threshold = 0x80
        easing = 'io3'
        f = filters.CutShadow(threshold, easing)
        act = f.process(img).tolist()
        
        self.assertListEqual(exp, act)
    
    def test_process_cut_shadow_in_quint(self):
        """Given an image, remove every gray below 50% and then 
        adjust the remaining grays to fill the whole gamut using 
        an in-out cubic easing function.
        """
        exp = [
            [0x00, 0x00, 0x00, 0x08, 0x3e, 0xff],
            [0x00, 0x00, 0x00, 0x08, 0x3e, 0xff],
            [0x00, 0x00, 0x00, 0x08, 0x3e, 0xff],
            [0x00, 0x00, 0x00, 0x08, 0x3e, 0xff],
            [0x00, 0x00, 0x00, 0x08, 0x3e, 0xff],
        ]
        
        img = np.array([
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        ])
        threshold = 0x80
        easing = 'i5'
        f = filters.CutShadow(threshold, easing)
        act = f.process(img).tolist()
        
        self.assertListEqual(exp, act)


class Rotate90TestCase(ut.TestCase):
    def test_rotate90_preprocessing(self):
        """The rotation filter only works if the image is a square 
        during the rotation. So, when given X and Y axes that are 
        not equal in size, return a new size where the X and Y axes 
        are equal and store how much padding the filter added, and 
        on which axis.
        """
        exp_size = (4, 128, 1280, 1280)
        exp_stored = (0, 0, 560, 0)
        
        size = (4, 128, 720, 1280)
        f = filters.Rotate90('r')
        act_size = f.preprocess(size)
        act_stored = f.padding
        
        self.assertTupleEqual(exp_size, act_size)
        self.assertEqual(exp_stored, act_stored)
    
    def test_rotate90_postprocessing(self):
        """When given an image size, reverse any padding that had 
        been previous done, returning the new size.
        """
        exp = (4, 128, 720, 1280)
        
        size = (4, 128, 1280, 1280)
        f = filters.Rotate90('r')
        f.padding = (0, 0, 560, 0)
        act = f.postprocess(size)
        
        self.assertTupleEqual(exp, act)
    
    def test_rotate90_rotate_right(self):
        """Given the colors of an image, rotate that image 90° in 
        the direction set on the filter."""
        exp = [
            [
                [3, 1],
                [4, 2],
            ],
            [
                [7, 5],
                [8, 6],
            ],
        ]
        
        values = np.array([
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ])
        f = filters.Rotate90('r')
        new = f.process(values)
        act = new.tolist()
        
        self.assertListEqual(exp, act)


class SkewTestCase(ut.TestCase):
    def test_skew_preprocess(self):
        """Skewing the image leaves artifacts on the edge of the image. 
        Given the current size of the image and the original size of 
        the image, return a size that has been padded to prevent the 
        artifacts from appearing in the final image, and store the 
        amount of padding added.
        """
        exp_size = (6, 128, 5, 9)
        exp_padding = (0, 0, 0, 2)
        
        original_size = (6, 128, 5, 5)
        size = (6, 128, 5, 7)
        slope = 1
        f = filters.Skew(slope)
        act_size = f.preprocess(size, original_size)
        act_padding = f.padding
        
        self.assertEqual(exp_padding, act_padding)
        self.assertTupleEqual(exp_size, act_size)
    
    def test_skew_slope_1(self):
        """Given the colors in the image, skew the Y axis with a slope 
        of 1, returning the new image.
        """
        exp = [
            [
                [1, 2],
                [4, 3],
            ],
            [
                [5, 6],
                [8, 7],
            ],
        ]
        
        values = np.array([
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ])
        f = filters.Skew(1)
        new = f.process(values)
        act = new.tolist()
        
        self.assertListEqual(exp, act)


# Processing tests.
class ProcessTestCase(ut.TestCase):
    def test_preprocess_layer_filters(self):
        """Given the list of filters and an image size, filters.process 
        should run the preprocessing for each image and return the new 
        image size.
        """
        exp = (6, 128, 9, 9)
        
        size = (6, 128, 5, 5)
        fs = [
            [],
            [
                filters.Skew(1),
                filters.Rotate90('r'),
            ],
            [
                filters.Skew(-1),
                filters.Rotate90('l'),
            ],
        ]
        act = filters.preprocess(size, fs)
        
        self.assertTupleEqual(exp, act)
    
    def test_postprocess_layer_filters(self):
        """Given a list of filters and an image, remove any padding 
        the filters added to the image and return the new image.
        """
        exp = [
            [
                [0x5, 0x6],
                [0x9, 0xA],
            ],
            [
                [0x5, 0x6],
                [0x9, 0xA],
            ],
        ]
        
        img = np.array([
            [
                [0x0, 0x1, 0x2, 0x3],
                [0x4, 0x5, 0x6, 0x7],
                [0x8, 0x9, 0xA, 0xB],
                [0xC, 0xD, 0xE, 0xF],
            ],
            [
                [0x0, 0x1, 0x2, 0x3],
                [0x4, 0x5, 0x6, 0x7],
                [0x8, 0x9, 0xA, 0xB],
                [0xC, 0xD, 0xE, 0xF],
            ],
        ])
        fs = [
            [],
            [
                filters.Skew(1),
                filters.Rotate90('r'),
            ],
        ]
        fs[1][0].padding = (0, 0, 2)
        fs[1][1].padding = (0, 2, 0)
        act = filters.postprocess(img, fs).tolist()
        
        self.assertListEqual(exp, act)
    
    def test_process_layer_filters(self):
        """Given a list of filters and an image, filters.process 
        should perform the filters on the image and return the 
        image.
        """
        exp = [[
            [
                [4, 1],
                [3, 2],
            ],
            [
                [8, 5],
                [7, 6],
            ],
        ],]
        
        values = np.array([[
            [
                [1, 2],
                [3, 4],
            ],
            [
                [5, 6],
                [7, 8],
            ],
        ],])
        fs = [
            [
                filters.Skew(1),
                filters.Rotate90('r'),
            ],
        ]
        act = filters.process(values, fs).tolist()
        
        self.assertListEqual(exp, act)
    
    def test_process_image_filters(self):
        """Given an image and a list of image filters, 
        filters.process_image should return a PIL.Image 
        object that has had the given image filters 
        applied.
        """
        # Expected result.
        exp = [
            [0x48, 0x63, 0x89, 0xaf, 0xc9],
            [0x48, 0x63, 0x89, 0xaf, 0xc9],
            [0x48, 0x63, 0x89, 0xaf, 0xc9],
            [0x48, 0x63, 0x89, 0xaf, 0xc9],
            [0x48, 0x63, 0x89, 0xaf, 0xc9],
        ]
        
        # Set up data and state for running the test.
        blur_amount = 2.0
        ifilters = [
            filters.Overlay(),
            filters.Blur(blur_amount)
        ]
        img = Image.fromarray(np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ], dtype=np.uint8))
        
        # Run the test.
        result_img = filters.process_image(img, ifilters)
        
        # Extract the actual result for comparison.
        act = np.array(result_img).tolist()
                
        # Determine whether the test passed.
        self.assertListEqual(exp, act)


# Image filter tests.
class BlurTestCase(ut.TestCase):
    def test_blur_process(self):
        """Given an image, perform a gaussian blur on the 
        image and return the result.
        """
        exp = [
            [0x42, 0x5c, 0x80, 0xa4, 0xbd],
            [0x42, 0x5c, 0x80, 0xa4, 0xbd],
            [0x42, 0x5c, 0x80, 0xa4, 0xbd],
            [0x42, 0x5c, 0x80, 0xa4, 0xbd],
            [0x42, 0x5c, 0x80, 0xa4, 0xbd],
        ]
        
        img = Image.fromarray(np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ], dtype=np.uint8))
        amount = 2.0
        f = filters.Blur(amount)
        result = f.process(img)
        act = np.array(result).tolist()
        
        self.assertEqual(exp, act)


class CurveTestCase(ut.TestCase):
    def test_curve_process(self):
        """Given an image, perform the easing function on the values 
        in the image.
        """
        exp = [
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
            [0x10, 0x81, 0xca, 0xf0, 0xfd, 0xff],
        ]
        
        img = Image.fromarray(np.array([
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        ], dtype=np.uint8))
        f = filters.Curve('io3')
        
        result = f.process(img)
        
        act = np.array(result).tolist()
        
        self.assertListEqual(exp, act)


class OverlayTestCase(ut.TestCase):
    def test_overlay_process(self):
        """Given an image, perform a 20% overlay operation on the 
        image and return the result.
        """
        exp = [
            [0x00, 0x46, 0x80, 0xed, 0xff],
            [0x00, 0x46, 0x80, 0xed, 0xff],
            [0x00, 0x46, 0x80, 0xed, 0xff],
            [0x00, 0x46, 0x80, 0xed, 0xff],
            [0x00, 0x46, 0x80, 0xed, 0xff],
        ]
        
        img = Image.fromarray(np.array([
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
            [0x00, 0x40, 0x80, 0xc0, 0xff,],
        ], dtype=np.uint8))
        f = filters.Overlay()
        result = f.process(img)
        act = np.array(result).tolist()
        
        self.assertListEqual(exp, act)


if __name__ == '__main__':
    raise NotImplemented