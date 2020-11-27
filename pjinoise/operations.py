"""
operations
~~~~~~~~~~

Array operations for use when combining two layers of an image or
animation.
"""
from functools import wraps
from typing import Callable, Union

import cv2
import numpy as np
from PIL import Image


# Decorators.
def clipped(fn: Callable) -> Callable:
    """Operations that use division or unbounded addition or
    subtraction can overflow the scale of the image. This will
    detect whether the scale is one or 0xff, then clip the
    image by setting everything below zero to zero and everything
    above the scale to the scale maximum before returning the
    image.
    """
    @wraps(fn)
    def wrapper(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
        scale = 1
        if len(a.shape) == 4:
            scale = 0xff
        ab = fn(a, b, amount)
        ab[ab < 0] = 0
        ab[ab > scale] = scale
        return ab
    return wrapper


def masked(fn: Callable) -> Callable:
    """Apply a blending mask to the operation."""
    @wraps(fn)
    def wrapper(a: np.ndarray, 
                b: np.ndarray, 
                amount: float = 1, 
                mask: Union[None, np.ndarray] = None) -> np.ndarray:
        ab = fn(a, b, amount)
        if mask is None:
            return ab
        diff = ab - a
        return a + diff * mask
    return wrapper


def scaled(fn: Callable) -> Callable:
    """Operations with multiplication rely on values being scaled to
    0 ≤ x ≤ 1 to keep the result from overflowing. Operations that add
    or subtract by one rely on that same scaling. Many color spaces
    are scaled to 0x00 ≤ x ≤ 0xff, so this will attempt to detect
    those images, rescale to one for the operation, and then rescale
    back to 0xff after the operation.
    """
    @wraps(fn)
    def wrapper(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
        rescaled = False
        if len(a.shape) == 4:
            try:
                a = a.astype(float) / 0xff
                b = b.astype(float) / 0xff
                rescaled = True
            except TypeError as e:
                msg = f'max {np.max(a)}, {e}'
                raise TypeError(msg)
        ab = fn(a, b, amount)
        if rescaled:
            ab = np.around(ab * 0xff).astype(np.uint8)
        return ab
    return wrapper


# Non-blends.
@masked
def replace(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """Simple replacement filter. Can double as an opacity filter
    if passed an amount, but otherwise this will just replace the
    values in a with the values in b.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param amount: (Optional.) How much of the blending should be
        applied to the values in a as a percentage. This is like
        the opacity setting on the top layer in a photo editing
        tool.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    if amount == 1:
        return b
    return a + (b - a) * float(amount)


# Darker/burn blends.
def darker(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    ab = a.copy()
    ab[b < a] = b[b < a]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@scaled
def multiply(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    ab = a * b
    if amount == 1:
        return ab
    ab = a + (ab - a) * float(amount)
    return ab


@clipped
@scaled
def color_burn(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """Taken from:
    http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
    """
    m = b != 0
    ab = np.zeros_like(a)
    ab[m] = 1 - (1 - a[m]) / b[m]
    ab[~m] = 0
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@clipped
@scaled
def linear_burn(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """Taken from:
    http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
    """
    if amount == 1:
        return a + b - 1
    ab = a + b - 1
    return a + (ab - a) * float(amount)


# Lighter/dodge blends.
def lighter(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    ab = a.copy()
    ab[b > a] = b[b > a]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@scaled
def screen(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    rev_a = 1 - a
    rev_b = 1 - b
    ab = rev_a * rev_b
    if amount == 1:
        return 1 - ab
    return a + (ab - a) * float(amount)


@clipped
@scaled
def color_dodge(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """Taken from:
    http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
    """
    if amount == 1:
        return a / (1 - b)
    ab = a / (1 - b)
    return a + (ab - a) * float(amount)


@clipped
def linear_dodge(a: np.ndarray,
                 b: np.ndarray,
                 amount: float = 1) -> np.ndarray:
    """Taken from:
    http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
    """
    if amount == 1:
        return a + b
    ab = a + b
    return a + (ab - a) * float(amount)


# Mixed blends.
def difference(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    if amount == 1:
        return np.abs(a - b)
    ab = np.abs(a - b)
    ab = a + (ab - a) * float(amount)
    return ab


@scaled
def exclusion(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """This is based on the equations found here:
    http://www.simplefilter.de/en/basics/mixmods.html
    """
    ab = a + b - 2 * a * b
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@scaled
def hard_light(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """This is based on the equations found here:
    http://www.simplefilter.de/en/basics/mixmods.html
    """
    ab = np.zeros(a.shape)
    ab[a < .5] = 2 * a[a < .5] * b[a < .5]
    ab[a >= .5] = 1 - 2 * (1 - a[a >= .5]) * (1 - b[a >= .5])
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@scaled
def hard_mix(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """This is based on the equations found here:
    http://www.simplefilter.de/en/basics/mixmods.html
    """
    ab = np.zeros_like(a)
    ab[a < 1 - b] = 0
    ab[a > 1 - b] = 1
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@clipped
@scaled
def linear_light(a: np.ndarray,
                 b: np.ndarray,
                 amount: float = 1) -> np.ndarray:
    """This is based on the equations found here:
    http://www.simplefilter.de/en/basics/mixmods.html
    """
    ab = b + 2 * a - 1
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@scaled
def overlay(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """This is based on the solution found here:
    https://stackoverflow.com/questions/52141987
    """
    mask = a >= .5
    ab = np.zeros_like(a)
    ab[~mask] = (2 * a[~mask] * b[~mask])
    ab[mask] = (1 - 2 * (1 - a[mask]) * (1 - b[mask]))
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@clipped
@scaled
def pin_light(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """This is based on the equations found here:
    http://www.simplefilter.de/en/basics/mixmods.html
    """
    # Build array masks to handle how the algorithm changes.
    m1 = np.zeros(a.shape, bool)
    m1[b < 2 * a - 1] = True
    m2 = np.zeros(a.shape, bool)
    m2[b > 2 * a] = True
    m3 = np.zeros(a.shape, bool)
    m3[~m1] = True
    m3[m2] = False

    # Blend the arrays using the algorithm.
    ab = np.zeros_like(a)
    ab[m1] = 2 * a[m1] - 1
    ab[m2] = 2 * a[m2]
    ab[m3] = b[m3]

    # Reduce the effect by the given amount and return.
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@scaled
def soft_light(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """This is based on the equations found here:
    http://www.simplefilter.de/en/basics/mixmods.html
    """
    m = np.zeros(a.shape, bool)
    ab = np.zeros(a.shape)
    m[a < .5] = True
    ab[m] = (2 * a[m] - 1) * (b[m] - b[m] ** 2) + b[m]
    ab[~m] = (2 * a[~m] - 1) * (np.sqrt(b[~m]) - b[~m]) + b[~m]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@clipped
@scaled
def vivid_light(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """This is based on the equations found here:
    http://www.simplefilter.de/en/basics/mixmods.html
    """
    # Create masks to handle the algorithm change and avoid division
    # by zero.
    m1 = np.zeros(a.shape, bool)
    m1[a <= .5] = True
    m1[a == 0] = False
    m2 = np.zeros(a.shape, bool)
    m2[a > .5] = True
    m2[a == 1] = False

    # Use the algorithm to blend the arrays.
    ab = np.zeros_like(a)
    ab[m1] = 1 - (1 - b[m1]) / (2 * a[m1])
    ab[m2] = b[m2] / (2 * (1 - a[m2]))
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


# Color blending operations.
def rgb_hue(a: np.ndarray, b: np.ndarray, amount: float = 1) -> np.ndarray:
    """This is based on the equations found here:
    http://www.simplefilter.de/en/basics/mixmods.html
    """
    if len(a.shape) != 4:
        raise ValueError('Given arrays must be RGB images.')
    ab = np.zeros_like(a)
    for i in range(a.shape[0]):
        a_hsv = cv2.cvtColor(a[i].astype(np.float32), cv2.COLOR_RGB2HSV)
        b_hsv = cv2.cvtColor(b[i].astype(np.float32), cv2.COLOR_RGB2HSV)
        ab_hsv = a_hsv.copy()
        ab_hsv[:, :, 0] = b_hsv[:, :, 0] * (b_hsv[:, :, 2] / 0xff)
        ab_hsv[:, :, 0] += a_hsv[:, :, 0] * (1 - (b_hsv[:, :, 2] / 0xff))
        ab_rgb = cv2.cvtColor(ab_hsv, cv2.COLOR_HSV2RGB)
        ab[i] = ab_rgb
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


# Registration.
registered_ops = {
    '': replace,
    'colorburn': color_burn,
    'colordodge': color_dodge,
    'darker': darker,
    'difference': difference,
    'exclusion': exclusion,
    'hardlight': hard_light,
    'hardmix': hard_mix,
    'lighter': lighter,
    'linearburn': linear_burn,
    'lineardodge': linear_dodge,
    'linearlight': linear_light,
    'multiply': multiply,
    'overlay': overlay,
    'pinlight': pin_light,
    'replace': replace,
    'screen': screen,
    'softlight': soft_light,
    'vividlight': vivid_light,

    'rgbhue': rgb_hue,
}
op_names = {registered_ops[k]: k for k in registered_ops}

if __name__ == '__main__':
    a = [
        [
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        ],
        [
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        ],
    ]
    b = [
        [
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20],
            [0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40],
            [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60],
            [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80],
            [0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0],
            [0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0],
            [0xe0, 0xe0, 0xe0, 0xe0, 0xe0, 0xe0, 0xe0, 0xe0, 0xe0],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        ],
        [
            [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            [0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20],
            [0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40],
            [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60],
            [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80],
            [0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0],
            [0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0, 0xc0],
            [0xe0, 0xe0, 0xe0, 0xe0, 0xe0, 0xe0, 0xe0, 0xe0, 0xe0],
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        ],
    ]
    scale = 0xff
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a[a != 0] = a[a != 0] / scale
    b[b != 0] = b[b != 0] / scale
    amount = 1

    op = hard_mix

    ab = op(a, b, amount)
    ab = np.around(ab * scale).astype(int)
    print('[')
    for frame in ab:
        print(' ' * 4 + '[')
        for row in frame:
            cols = [f'0x{n:02x}' for n in row]
            print(' ' * 8 + '[' + ', '.join(cols) + ']',)
        print(' ' * 4 + ']')
    print(']')
