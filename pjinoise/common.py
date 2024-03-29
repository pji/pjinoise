"""
common
~~~~~~

Utilities and other commonly reused functions for pjinoise.
"""
from typing import Any, List, Mapping, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from pjinoise.constants import SUPPORTED_FORMATS, X, Y, Z


# General purpose functions.
def convert_color_space(a: np.ndarray,
                        src_space: str = '',
                        dst_space: str = 'RGB') -> np.ndarray:
    """Convert an array to the given color space.

    :param src_space: (Optional.) This is the identifier for the
        current color space of the image data. These identifiers
        are either an empty string to represent pjinoise grayscale
        or a color mode used by the pillow module (see below).
    :param dst_space: (Optional.) This is the identifier for the
        destination color space of the image data. These identifiers
        are either an empty string to represent pjinoise grayscale
        or a color mode used by the pillow module (see below).
    :return: :class:ndarray object.
    :rtype: numpy.ndarray

    Color Modes
    -----------
    The color modes used by the pillow library can be found here:

        https://pillow.readthedocs.io/en/stable/handbook/
        concepts.html#concept-modes
    """
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

    # The pjinoise grayscale isn't a mode that is recognized by
    # pillow, so we'll need pillow to convert it to its grayscale
    # first (mode 'L').
    dst_is_pjinoise_grayscale = False
    if dst_space == '':
        dst_is_pjinoise_grayscale = True
        dst_space = 'L'

    # PIL.image.convert can only convert two-dimensional (or three,
    # with color channel being the third) images. So, for animations
    # we have to iterate through the Z axis, converting one frame at
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

    # If we are converting to pjinoise grayscale, need to take the
    # eight-bit integers from pillow and turn them into the pjinoise
    # grayscale floats.
    if dst_is_pjinoise_grayscale:
        out = out.astype(float) / 0xff
    return out


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


# Diagnostic output functions.
def print_array(a: np.ndarray, depth: int = 0, color: bool = True) -> None:
    """Write the values of the given array to stdout."""
    if len(a.shape) > 1:
        print(' ' * (4 * depth) + '[')
        for i in range(a.shape[0]):
            print_array(a[i], depth + 1, color)
        print(' ' * (4 * depth) + '],')

    else:
        if a.dtype != np.uint8 and color:
            a = (a.copy() * 0xff).astype(np.uint8)
            tmp = '0x{:02x}'
        else:
            tmp = '{}'
        nums = [tmp.format(n) for n in a]
        print(' ' * (4 * depth) + '[' + ', '.join(nums) + '],')


def print_float_array(a: np.ndarray, depth: int = 0) -> None:
    """Write the values of the given array to stdout."""
    if len(a.shape) > 1:
        print(' ' * (4 * depth) + '[')
        for i in range(a.shape[0]):
            print_float_array(a[i], depth + 1)
        print(' ' * (4 * depth) + '],')

    else:
        tmp = '{:05.2f}'
        nums = [tmp.format(n) for n in a]
        print(' ' * (4 * depth) + '[' + ', '.join(nums) + '],')


def print_seq(seq: Sequence[Any], depth: int = 0) -> None:
    """Write the values of the given sequence to stdout."""
    print_array(np.array(seq), depth, False)


# Serialization/deserialization functions.
def deserialize_sequence(value: Union[Sequence[float], str]) -> Tuple[float]:
    """Deserialize a set of coordinates that could have come from
    command line input.
    """
    if not value:
        return (0, 0, 0)
    if isinstance(value, str):
        value = value.split(',')[::-1]
    return tuple(float(n) for n in value)


def remove_private_attrs(mapping: Mapping) -> Mapping:
    """Remove the keys for private attributes from an object that
    has been serialized as an mapping.
    """
    cls = type(mapping)
    public_keys = [key for key in mapping if not key.startswith('_')]
    dict_ = {key: mapping[key] for key in public_keys}
    return cls(dict_)


def text_to_int(text: Union[bytes, str, int, None]) -> Union[int, None]:
    if isinstance(text, int) or text is None:
        return text
    if isinstance(text, str):
        text = bytes(text, 'utf_8')
    return int.from_bytes(text, 'little')


# Interpolation functions.
def lerp(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Perform a linear interpolation on the values of two arrays

    :param a: The "left" values.
    :param b: The "right" values.
    :param x: An array of how close the location of the final value
        should be to the "left" value.
    :return: A :class:ndarray object
    :rtype: numpy.ndarray

    Usage::

        >>> import numpy as np
        >>>
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([3, 4, 5])
        >>> x = np.array([.5, .5, .5])
        >>> lerp(a, b, x)
        array([2., 3., 4.])
    """
    return a.astype(float) * (1 - x.astype(float)) + b.astype(float) * x


def trilinear_interpolation(a: np.ndarray, factor: float) -> np.ndarray:
    """Resize an three dimensional array using trilinear
    interpolation.

    :param a: The array to resize. The array is expected to have at
        least three dimensions.
    :param factor: The amount to resize the array. Given how the
        interpolation works, you probably don't get great results
        with factor less than or equal to .5. Consider multiple
        passes of interpolation with larger factors in those cases.
    :return: A :class:ndarray object.
    :rtype: numpy.ndarray

    Usage::

        >>> import numpy as np
        >>>
        >>> a = np.array([
        ...     [
        ...             [0, 1],
        ...             [1, 0],
        ...     ],
        ...     [
        ...             [1, 0],
        ...             [0, 1],
        ...     ],
        ... ])
        >>> trilinear_interpolation(a, 2)
        array([[[0. , 0.5, 1. , 1. ],
                [0.5, 0.5, 0.5, 0.5],
                [1. , 0.5, 0. , 0. ],
                [1. , 0.5, 0. , 0. ]],
        <BLANKLINE>
               [[0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5]],
        <BLANKLINE>
               [[1. , 0.5, 0. , 0. ],
                [0.5, 0.5, 0.5, 0.5],
                [0. , 0.5, 1. , 1. ],
                [0. , 0.5, 1. , 1. ]],
        <BLANKLINE>
               [[1. , 0.5, 0. , 0. ],
                [0.5, 0.5, 0.5, 0.5],
                [0. , 0.5, 1. , 1. ],
                [0. , 0.5, 1. , 1. ]]])
    """
    # Return the array unchanged if the array won't be magnified.
    if factor == 1:
        return a

    # Perform a defensive copy of the original array to avoid
    # unexpected side effects.
    a = a.copy()

    # Since we are magnifying the given array, the new array's shape
    # will increase by the magnification factor.
    mag_size = tuple(int(s * factor) for s in a.shape)

    # Map out the relationship between the old space and the
    # new space.
    indices = np.indices(mag_size)
    if factor > 1:
        whole = (indices // factor).astype(int)
        parts = (indices / factor - whole).astype(float)
    else:
        new_ends = [s - 1 for s in mag_size]
        old_ends = [s - 1 for s in a.shape]
        true_factors = [n / o for n, o in zip(new_ends, old_ends)]
        for i in range(len(true_factors)):
            if true_factors[i] == 0:
                true_factors[i] = .5
        whole = indices.copy()
        parts = indices.copy()
        for i in Z, Y, X:
            whole[i] = (indices[i] // true_factors[i]).astype(int)
            parts[i] = (indices[i] / true_factors[i] - whole[i]).astype(float)
    del indices

    # Trilinear interpolation determines the value of a new pixel by
    # comparing the values of the eight old pixels that surround it.
    # The hashes are the keys to the dictionary that contains those
    # old pixel values. The key indicates the position of the pixel
    # on each axis, with one meaning the position is ahead of the
    # new pixel, and zero meaning the position is behind it.
    hashes = [f'{n:>03b}'[::-1] for n in range(2 ** 3)]
    hash_table = {}

    # The original array needs to be made one dimensional for the
    # numpy.take operation that will occur as we build the tables.
    raveled = np.ravel(a)

    # Build the table that contains the old pixel values to
    # interpolate.
    for hash in hashes:
        hash_whole = whole.copy()

        # Use the hash key to adjust the which old pixel we are
        # looking at.
        for axis in Z, Y, X:
            if hash[axis] == '1':
                hash_whole[axis] += 1

                # Handle the pixels that were pushed off the far
                # edge of the original array by giving them the
                # value of the last pixel along that axis in the
                # original array.
                m = np.zeros(hash_whole[axis].shape, dtype=bool)
                m[hash_whole[axis] >= a.shape[axis]] = True
                hash_whole[axis][m] = a.shape[axis] - 1

        # Since numpy.take() only works in one dimension, we need to
        # map the three dimensional indices of the original array to
        # the one dimensional indices used by the raveled version of
        # that array.
        raveled_indices = hash_whole[Z] * a.shape[Y] * a.shape[X]
        raveled_indices += hash_whole[Y] * a.shape[X]
        raveled_indices += hash_whole[X]

        # Get the value of the pixel in the original array.
        hash_table[hash] = np.take(raveled, raveled_indices.astype(int))

    # Once the hash table has been built, clean up the working arrays
    # in case we are running short on memory.
    else:
        del hash_whole, raveled_indices, whole

    # Everything before this was to set up the interpolation. Now that
    # it's set up, we perform the interpolation. Since we are doing
    # this across three dimensions, it's a three stage process. Stage
    # one is along the X axis.
    x1 = lerp(hash_table['000'], hash_table['001'], parts[X])
    x2 = lerp(hash_table['010'], hash_table['011'], parts[X])
    x3 = lerp(hash_table['100'], hash_table['101'], parts[X])
    x4 = lerp(hash_table['110'], hash_table['111'], parts[X])

    # Stage two is along the Y axis.
    y1 = lerp(x1, x2, parts[Y])
    y2 = lerp(x3, x4, parts[Y])
    del x1, x2, x3, x4

    # And stage three is along the Z axis. Since this is the last step
    # we can just return the result.
    return lerp(y1, y2, parts[Z])


if __name__ == '__main__':
    a = [
        [
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0],
            [0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0],
            [0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0],
            [0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80],
            [0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60],
            [0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40],
            [0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40, 0x20],
            [0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40, 0x20, 0x00],
        ],
        [
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0],
            [0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0],
            [0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0],
            [0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80],
            [0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60],
            [0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40],
            [0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40, 0x20],
            [0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40, 0x20, 0x00],
        ],
    ]
    a = np.array(a, dtype=float)
    a = a / 0xff
    factor = .5
    result = trilinear_interpolation(a, factor)
    print_array(result)
