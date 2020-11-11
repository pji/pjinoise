"""
operations
~~~~~~~~~~

Array operations for use when combining two layers of an image or 
animation.
"""
from functools import wraps
from typing import Callable

import numpy as np


# Decorators.
def clipped(fn:Callable) -> Callable:
    """Operations that use division or unbounded addition or 
    subtraction can overflow the scale of the image. This will 
    detect whether the scale is one or 0xff, then clip the 
    image by setting everything below zero to zero and everything 
    above the scale to the scale maximum before returning the 
    image.
    """
    @wraps(fn)
    def wrapper(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
        scale = 1
        if len(a.shape) == 4:
            scale = 0xff
        ab = fn(a, b, amount)
        ab[ab < 0] = 0
        ab[ab > scale] = scale
        return ab
    return wrapper


def scaled(fn:Callable) -> Callable:
    """Operations with multiplication rely on values being scaled to 
    0 ≤ x ≤ 1 to keep the result from overflowing. Operations that add 
    or subtract by one rely on that same scaling. Many color spaces 
    are scaled to 0x00 ≤ x ≤ 0xff, so this will attempt to detect 
    those images, rescale to one for the operation, and then rescale 
    back to 0xff after the operation.
    """
    @wraps(fn)
    def wrapper(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
        rescaled = False
        if len(a.shape) == 4:
            a /= 0xff
            b /= 0xff
            rescaled = True
        ab = fn(a, b, amount)
        if rescaled:
            ab *= 0xff
        return ab
    return wrapper


# Non-blends.
def replace(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
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
def darker(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    ab = a.copy()
    ab[b < a] = b[b < a]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@scaled
def multiply(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    ab = a * b
    if amount == 1:
        return ab
    ab = a + (ab - a) * float(amount)
    return ab


@clipped
@scaled
def color_burn(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
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
def linear_burn(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    """Taken from:
    http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
    """
    if amount == 1:
        return a + b - 1
    ab = a + b - 1
    return a + (ab - a) * float(amount)


# Lighter/dodge blends.
def lighter(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    ab = a.copy()
    ab[b > a] = b[b > a]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@scaled
def screen(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    rev_a = 1 - a
    rev_b = 1 - b
    ab = rev_a * rev_b
    if amount == 1:
        return 1 - ab
    return a + (ab - a) * float(amount)


@clipped
@scaled
def color_dodge(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    """Taken from:
    http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
    """
    if amount == 1:
        return a / (1 - b)
    ab = a / (1 - b)
    return a + (ab - a) * float(amount)


@clipped
def linear_dodge(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    """Taken from:
    http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
    """
    if amount == 1:
        return a + b
    ab = a + b
    return a + (ab - a) * float(amount)


# Mixed blends.
def difference(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    if amount == 1:
        return np.abs(a - b)
    ab = np.abs(a - b)
    ab = a + (ab - a) * float(amount)
    return ab


@scaled
def hard_light(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
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
def overlay(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    """This is based on the solution found here:
    https://stackoverflow.com/questions/52141987
    """
    mask = a >= .5
    ab = np.zeros_like(a)
    ab[~mask] = (2 * a * b)[~mask]
    ab[mask] = (1 - 2 * (1 - a) * (1 - b))[mask]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@scaled
def soft_light(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    """This is based on the equations found here:
    http://www.simplefilter.de/en/basics/mixmods.html
    """
    m = np.zeros(a.shape, bool)
    ab = np.zeros(a.shape)
    m[a < .5] = True
    ab[m] = (2 * a[m] - 1) * (b[m] - b[m] ** 2) + b[m]
    ab[~m] = (2 * a[~m] -1) * (np.sqrt(b[~m]) - b[~m]) + b[~m]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


@clipped
@scaled
def vivid_light(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    m = np.zeros(a.shape, bool)
    m[a <= .5] = True
    ab = np.zeros_like(a)
    ab[m] = 1 - (1 - b[m]) / (2 * a[m])
    ab[~m] = b / (2 * (1 - a))
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
    'hardlight': hard_light,
    'lighter': lighter,
    'linearburn': linear_burn,
    'lineardodge': linear_dodge,
    'multiply': multiply,
    'overlay': overlay,
    'replace': replace,
    'screen': screen,
    'softlight': soft_light,
}

if __name__ == '__main__':
    raise NotImplementedError