"""
test_pjinoise
~~~~~~~~~~~~~

Unit tests for the core image generation of the pjinoise module.
"""
import unittest as ut

import numpy as np

from pjinoise import filters as f
from pjinoise import model as m
from pjinoise import operations as op
from pjinoise import pjinoise as pn
from pjinoise import sources as s
from pjinoise.common import grayscale_to_ints_list
from pjinoise.constants import X, Y, Z


# Utility classes and functions.
def slice_array(a, new_shape):
    begin = [(o - n) // 2 for o, n in zip(a.shape, new_shape)]
    end = [b + n for b, n in zip(begin, new_shape)]
    slices = tuple(slice(b, e) for b, e in zip(begin, end))
    return a[slices]


class Filter(f.ForLayer):
    amount = .5

    def preprocess(self, size, original_size):
        return [size[Z], size[Y] + 2, size[X] + 2,]

    def process(self, a):
        return a * self.amount

    def postprocess(self, size):
        new_size = [size[Z], size[Y] - 2, size[X] - 2]
        return new_size


class Source(s.ValueSource):
    def __init__(self, flip=False, *args, **kwargs):
        self.image = np.array([
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
        ]).astype(int)
        self.scale = 0xff
        self.size = self.image.shape
        if flip:
            self.image = np.flip(self.image, -1)

    def fill(self, size, loc, *args, **kwargs):
        a = self.image.astype(float) / self.scale
        if loc:
            for axis in X, Y, Z:
                a = np.roll(a, loc[axis], axis=axis)
        if size != self.size:
            a = slice_array(a, size)
        return a


class SourceMask(Source):
    def __init__(self, *args, **kwargs):
        self.image = np.array([
            [
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x80, 0x80, 0x80, 0x80, 0x80,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
            ],
            [
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x00, 0x00, 0x00, 0x00, 0x00,],
                [0x80, 0x80, 0x80, 0x80, 0x80,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
                [0xff, 0xff, 0xff, 0xff, 0xff,],
            ],
        ], dtype=int)
        self.scale = 0xff
        self.size = self.image.shape


# Test cases.
class LayerTestCase(ut.TestCase):
    def test_blend_data_from_multiple_layers(self):
        """Given multiple Layers and a size, produce the image data
        from those layers and return the blended data.
        """
        # Set up data for expected values.
        src = Source()

        # Expected values.
        exp = abs(src.image - src.image).tolist()

        # Set up test data and state.
        size = src.size
        layers = [
            m.Layer(**{
                'source': Source(),
                'blend': op.replace,
                'blend_amount': 1,
                'location': [0, 0, 0],
                'filters': [],
                'mask': None,
                'mask_filters': []
            }),
            m.Layer(**{
                'source': Source(),
                'blend': op.difference,
                'blend_amount': 1,
                'location': [0, 0, 0],
                'filters': [],
                'mask': None,
                'mask_filters': []
            }),
        ]

        # Run test.
        result = pn.process_layers(size, layers)

        # Extract actual values.
        a = np.around(result * src.scale).astype(int)
        act = a.tolist()

        # Determine whether test passed.
        self.assertListEqual(exp, act)

    def test_blend_data_from_nested_layers(self):
        """Given multiple Layers with nested Layer objects and a size,
        produce the image data from those layers and return the blended
        data.
        """
        # Set up data for expected values.
        src = Source()

        # Expected values.
        exp = abs(src.image - abs(src.image - src.image)).tolist()

        # Set up test data and state.
        size = src.size
        layers = [
            m.Layer(**{
                'source': Source(),
                'blend': op.replace,
                'blend_amount': 1,
                'location': [0, 0, 0],
                'filters': [],
                'mask': None,
                'mask_filters': []
            }),
            m.Layer(**{
                'source': [
                    m.Layer(**{
                        'source': Source(),
                        'blend': op.difference,
                        'blend_amount': 1,
                        'location': [0, 0, 0],
                        'filters': [],
                        'mask': None,
                        'mask_filters': []
                    }),
                    m.Layer(**{
                        'source': Source(),
                        'blend': op.difference,
                        'blend_amount': 1,
                        'location': [0, 0, 0],
                        'filters': [],
                        'mask': None,
                        'mask_filters': []
                    }),
                ],
                'blend': op.difference,
                'blend_amount': 1,
                'location': [0, 0, 0],
                'filters': [],
                'mask': None,
                'mask_filters': []
            }),
        ]

        # Run test.
        result = pn.process_layers(size, layers)

        # Extract actual values.
        a = np.around(result * src.scale).astype(int)
        act = a.tolist()

        # Determine whether test passed.
        self.assertListEqual(exp, act)

    def test_blend_data_with_colorization(self):
        """Given nested Layers with the colorization filter, ensure
        the layer output is converted to a color space that conserves
        the colors before blending.
        """
        # Expected values.
        exp = [
            [
                [[0xa1,0xa1,0xa1,],[0x81,0x81,0x81,],[0x61,0x61,0x61,],],
                [[0xa1,0xa1,0xa1,],[0x81,0x81,0x81,],[0x61,0x61,0x61,],],
                [[0xa1,0xa1,0xa1,],[0x81,0x81,0x81,],[0x61,0x61,0x61,],],
            ],
            [
                [[0xa1,0xa1,0xa1,],[0x81,0x81,0x81,],[0x61,0x61,0x61,],],
                [[0xa1,0xa1,0xa1,],[0x81,0x81,0x81,],[0x61,0x61,0x61,],],
                [[0xa1,0xa1,0xa1,],[0x81,0x81,0x81,],[0x61,0x61,0x61,],],
            ],
        ]

        # Set up test data and state.
        src = Source()
        size = (2, 3, 3)
        layers = [
            m.Layer(**{
                'source': Source(),
                'blend': op.replace,
                'blend_amount': 1,
                'location': [0, 0, 0],
                'filters': [],
                'mask': None,
                'mask_filters': []
            }),
            m.Layer(**{
                'source': Source(),
                'blend': op.difference,
                'blend_amount': 1,
                'location': [0, 0, 0],
                'filters': [
                    f.Color('W'),
                ],
                'mask': None,
                'mask_filters': []
            }),
            m.Layer(**{
                'source': Source(),
                'blend': op.difference,
                'blend_amount': 1,
                'location': [0, 0, 0],
                'filters': [],
                'mask': None,
                'mask_filters': []
            }),
        ]

        # Run test.
        result = pn.process_layers(size, layers)

        # Extract actual values.
        a = np.around(result).astype(int)
        act = a.tolist()

        # Determine whether test passed.
        self.assertListEqual(exp, act)

    def test_blend_data_with_mask(self):
        """Given a base layer and a blend layer with a mask, blend
        those two layers using the mask.
        """
        # Expected value.
        exp = [
            [
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x80, 0x80, 0x80, 0x80, 0x7f,],
                [0xff, 0xc0, 0x80, 0x40, 0x00,],
                [0xff, 0xc0, 0x80, 0x40, 0x00,],
            ],
            [
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x00, 0x40, 0x80, 0xc0, 0xff,],
                [0x80, 0x80, 0x80, 0x80, 0x7f,],
                [0xff, 0xc0, 0x80, 0x40, 0x00,],
                [0xff, 0xc0, 0x80, 0x40, 0x00,],
            ],
        ]

        # Set up test data and state.
        layers = [
            m.Layer(**{
                'source': Source(),
                'blend': op.replace,
                'blend_amount': 1,
                'location': [0, 0, 0],
                'filters': [],
                'mask': None,
                'mask_filters': []
            }),
            m.Layer(**{
                'source': Source(flip=True),
                'blend': op.replace,
                'blend_amount': 1,
                'location': [0, 0, 0],
                'filters': [],
                'mask': SourceMask(),
                'mask_filters': []
            }),
        ]
        size = layers[0].source.size

        # Run test.
        result = pn.process_layers(size, layers)

        # Extract actual values.
        act = grayscale_to_ints_list(result)

        # Determine whether test passed.
        self.maxDiff = None
        self.assertListEqual(exp, act)

    def count_sources(self):
        """Given a configuration object, count the number of sources
        used in the configuration. This is used to display progress
        in the UI.
        """
        # Expected values.
        exp = 3

        # Set up test data and state.
        src = Source()
        conf = m.Image(**{
            'source': [
                m.Layer(**{
                    'source': Source(),
                    'blend': op.replace,
                    'blend_amount': 1,
                    'location': [0, 0, 0],
                    'filters': [],
                    'mask': None,
                    'mask_filters': []
                }),
                m.Layer(**{
                    'source': [
                        m.Layer(**{
                            'source': Source(),
                            'blend': op.difference,
                            'blend_amount': 1,
                            'location': [0, 0, 0],
                            'filters': [],
                            'mask': None,
                            'mask_filters': []
                        }),
                        m.Layer(**{
                            'source': Source(),
                            'blend': op.difference,
                            'blend_amount': 1,
                            'location': [0, 0, 0],
                            'filters': [],
                            'mask': None,
                            'mask_filters': []
                        }),
                    ],
                    'blend': op.difference,
                    'blend_amount': 1,
                    'location': [0, 0, 0],
                    'filters': [],
                    'mask': None,
                    'mask_filters': []
                }),
            ],
            'size': src.size,
            'filename': 'spam.jpg',
            'format': 'JPEG',
            'mode': 'RGB',
            'framerate': None,
        })

        # Run test.
        act = conf.count_sources()

        # Determine if test passed.
        self.assertEqual(exp, act)

    def test_create_image_data_from_valuesource(self):
        """Given a ValueSource and a size, produce an amount of image
        data equal to size from the given location within the source.
        """
        # Set up data for expected values.
        src = Source()

        # Expected values.
        exp = src.image.tolist()

        # Set up test data and state.
        size = src.size

        # Run test.
        result = pn.render_source(src, size)

        # Extract actual data from the test.
        a = np.around(result * src.scale).astype(int)
        act = a.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_create_image_data_with_location(self):
        """If a location is given, the image data should be offset
        by that location when generated by the ValueSource.
        """
        # Expected values.
        exp = [
            [
                [0x40, 0x80, 0xc0, 0xff, 0x00],
                [0x40, 0x80, 0xc0, 0xff, 0x00],
                [0x40, 0x80, 0xc0, 0xff, 0x00],
                [0x40, 0x80, 0xc0, 0xff, 0x00],
                [0x40, 0x80, 0xc0, 0xff, 0x00],
            ],
            [
                [0x40, 0x80, 0xc0, 0xff, 0x00],
                [0x40, 0x80, 0xc0, 0xff, 0x00],
                [0x40, 0x80, 0xc0, 0xff, 0x00],
                [0x40, 0x80, 0xc0, 0xff, 0x00],
                [0x40, 0x80, 0xc0, 0xff, 0x00],
            ],
        ]

        # Set up test data and state.
        src = Source()
        size = src.size
        location = [0, 0, -1]

        # Run test.
        result = pn.render_source(src, size, location)

        # Extract actual data from the test.
        a = np.around(result * src.scale).astype(int)
        act = a.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)

    def test_create_image_data_with_filters(self):
        """If a list of filters are given, run those filters on the
        image data generated by the ValueSource.
        """
        # Expected data.
        exp = [
            [
                [0x20, 0x40, 0x60,],
                [0x20, 0x40, 0x60,],
                [0x20, 0x40, 0x60,],
            ],
            [
                [0x20, 0x40, 0x60,],
                [0x20, 0x40, 0x60,],
                [0x20, 0x40, 0x60,],
            ],
        ]

        # Set up test data and state.
        src = Source()
        filters = [Filter(),]
        size = [2, 3, 3]
        location = [0, 1, 1]

        # Run test.
        result = pn.render_source(src, size, location, filters)

        # Extract actual data from the test.
        a = np.around(result * src.scale).astype(int)
        act = a.tolist()

        # Determine if test passed.
        self.assertListEqual(exp, act)
