"""
common
~~~~~~

Utilities and other commonly reused functions for pjinoise.
"""
from functools import wraps
from typing import List, Mapping, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from pjinoise.constants import SUPPORTED_FORMATS


X, Y, Z = 2, 1, 0


# General purpose functions.
def convert_color_space(a: np.ndarray,
                        src_space: str = '',
                        dst_space: str = 'RGB') -> np.ndarray:
    """Convert an array to the given color space."""
    # The shape of the output is based on the space, so we can't
    # build out until we do the first conversion. However, setting
    # it to None here makes the process of detecting whether we've
    # set up the output array a little smoother later.
    out = None

    # Most of pjinoise tries to work with grayscale color values
    # that go from zero to one. However, pillow's grayscale mode
    # is 'L', which represents the color as an unsigned 8 bit
    # integer. The data will need to at least be in mode 'L' for
    # pillow to be able to convert the color space.
    if src_space == '':
        assert np.max(a) <= 1.0
        a = np.around(a * 0xff).astype(np.uint8)
        src_space = 'L'

    # PIL.image.convert can only convert two-dimensional (or three,
    # with color channel being the third) images. So, for animations
    # we have to iterate through the Z axis, coverting one frame at
    # a time. Since pjinoise thinks of still images as single frame
    # animations, this means we're always going to have to handle
    # the Z axis like this.
    for i in range(a.shape[Z]):
        img = Image.fromarray(a[i], mode=src_space)
        img = img.convert(dst_space)
        a_img = np.array(img)
        if out is None:
            out = np.zeros((a.shape[Z], *a_img.shape), dtype=np.uint8)
        out[i] = a_img
    return out


def deserialize_sequence(value: Union[Sequence[float], str]) -> Tuple[float]:
    """Deserialize a set of coordinates that could have come from
    command line input.
    """
    if not value:
        return (0, 0, 0)
    if isinstance(value, str):
        value = value.split(',')[::-1]
    return tuple(float(n) for n in value)


def grayscale_to_ints_list(a: np.ndarray,
                           astype: type = np.uint8) -> List[int]:
    """pjinoise grayscale stores color values as floats between
    zero and one. This is a pain to read on a screen or type
    expected values for. This function converts that to lists
    of integers for easier test comparison and printing.
    """
    a = a.copy()
    a = np.around(a * 0xff)
    a = a.astype(astype)
    return a.tolist()


def get_format(filename: str) -> str:
    """Determine the image type based on the filename."""
    name_part = filename.split('.')[-1]
    extension = name_part.casefold()
    try:
        return SUPPORTED_FORMATS[extension]
    except KeyError:
        print(f'The file type {name_part} is not supported.')
        supported = ', '.join(SUPPORTED_FORMATS)
        print(f'The supported formats are: {supported}.')
        raise SystemExit


def print_array(a: np.ndarray, depth: int = 0) -> None:
    """Write the values of the given array to stdout."""
    if len(a.shape) > 1:
        print(' ' * (4 * depth) + '[')
        for i in range(a.shape[0]):
            print_array(a[i], depth + 1)
        print(' ' * (4 * depth) + '],')

    else:
        if a.dtype != np.uint8:
            a = np.around(a.copy() * 0xff).astype(np.uint8)
        tmp = '0x{:02x}'
        nums = [tmp.format(n) for n in a]
        print(' ' * (4 * depth) + '[' + ', '.join(nums) + '],')


def remove_private_attrs(map: Mapping) -> Mapping:
    """Remove the keys for private attributes from an object that
    has been serialized as an mapping.
    """
    pvt_keys = [key for key in map if key.startswith('_')]
    for key in pvt_keys:
        del map[key]
    return map
