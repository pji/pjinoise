"""
ease
~~~~

Basic easing function implementations. Easing functions taken from:

    https://easings.net

An easing function is used to adjust the "curve" of the array. The
curve is the graph of values in the array and the number of times each
value appears. An easing function will change the values in the array
in a predictable way that will change the shape of that curve. This
can be used to create various effects when applied to the color values
in an image.


Basic Usage
===========
Easing functions follow the easing protocol, which is the following:

    :param a: A numpy n-dimensional array object containing float
        values within the range 0 <= x <= 1.
    :return: A numpy n-dimensional array object containing float
        values within the range of 0 <= x <= 1.
    :rtype: numpy.ndarray

Usage::

    >>> import numpy as np
    >>> a = np.array([0x40, 0xa0, 0xc0], dtype=float)
    >>> a = a / 0xff
    >>> out = in_quint(a)


Overflowing Eases
=================
Several easing functions will return values outside of the expected
range. These functions are given the overflows decorator that allows
you to chose how it behaves. Functions with the overflows decorator
have an additional parameter:

    :param action: (Optional.) A function that accepts and returns
        a numpy n-dimensional array that determines how the easing
        function handles values that fall outside the range of
        0 <= x <= 1. Defaults to rescaling the values to fit within
        the range.

The following actions are available within this module:

*   clip: Set all values less than zero to zero and all values greater
    than one to one.
*   nochange: Return the values outside the range unchanged.
*   rescale: Adjust the scale of the values in the array so that the
    minimum value is zero and the maximum value is one.

Usage::

    >>> import numpy as np
    >>> a = np.array([0x40, 0xa0, 0xc0], dtype=float)
    >>> a = a / 0xff
    >>> out = in_out_back(a, clip)


Serialization Helpers
=====================
The ease module has a few capabilities to help with serialization. The
get_regname_for_func() function returns a string that represents an
easing function that has been registered with the ease module.

Usage::

    >>> fn = in_out_circ
    >>> get_regname_for_func(fn)
    'ior'

New easing functions can be registered with the ease module by adding
them to the registered_functions dictionary. The key should be the
short string you want to use as the serialized value of the function.

Usage::

    >>> def spam(a):
    ...     return a / .5
    ...
    >>> registered_functions['sp'] = spam
    >>> get_regname_for_func(spam)
    'sp'

The primary purpose for this feature is to allow classes that hold
easing values as attributes to be serialized with the short name of
function rather than the function itself. This makes serialization
into a format like JSON easier, and it provides a measure of input
validation at deserialization to reduce the risk of remote code
execution vulnerabilities.
"""
from functools import wraps
from typing import Callable

import numpy as np


# Overflow actions.
def clip(a: np.ndarray) -> np.ndarray:
    """Clip an array that exceeds the boundaries of zero and one."""
    a[a < 0] = 0
    a[a > 1] = 1
    return a


def nochange(a: np.ndarray) -> np.ndarray:
    """Return an array without change."""
    return a


def rescale(a: np.ndarray) -> np.ndarray:
    """Rescale an array that exceeds the boundaries of zero and one."""
    scale = np.max(a) - np.min(a)
    if scale != 0 and (np.max(a) > 1.0 or np.min(a) < 0):
        old_scale = np.max(a) - np.min(a)
        a = a - np.min(a)
        a = a / old_scale
    elif scale == 0 and np.min(a) < 0:
        a.fill(0)
    elif scale == 0 and np.min(a) > 1:
        a.fill(1)
    return a


# Common decorators.
def overflows(fn: Callable) -> Callable:
    """A decorator to handle easing functions that overflow the
    boundary values of zero and one.
    """
    @wraps(fn)
    def wrapper(a: np.array, action: Callable = rescale) -> np.ndarray:
        a = fn(a)
        return action(a)
    return wrapper


# Don't ease function.
def linear(a: np.ndarray) -> np.ndarray:
    """Don't perform easing. This exists to avoid having to check if
    easing is needed before sending to an easing function.
    """
    return a


# Ease in and out functions.
@overflows
def in_out_back(a: np.ndarray) -> np.ndarray:
    c1 = 1.70158
    c2 = c1 * 1.525
    m = np.zeros(a.shape, bool)
    m[a < .5] = True
    a[m] = (2 * a[m]) ** 2 * ((c2 + 1) * 2 * a[m] - c2) / 2
    a[~m] = ((2 * a[~m] - 2) ** 2 * ((c2 + 1) * (a[~m] * 2 - 2) + c2) + 2) / 2
    return a


def in_out_circ(a: np.ndarray) -> np.ndarray:
    m = np.zeros(a.shape, bool)
    m[a < .5] = True
    a[m] = (1 - np.sqrt(1 - (2 * a[m]) ** 2)) / 2
    a[~m] = (np.sqrt(1 - (-2 * a[~m] + 2) ** 2) + 1) / 2
    return a


def in_out_cubic(a: np.ndarray) -> np.ndarray:
    """Perform the in out cubic easing function on the array."""
    a[a < .5] = 4 * a[a < .5] ** 3
    a[a >= .5] = 1 - (-2 * a[a >= .5] + 2) ** 3 / 2
    return a


@overflows
def in_out_elastic(a: np.ndarray) -> np.ndarray:
    c5 = (2 * np.pi) / 4.5

    # Create masks for the array.
    m1 = np.zeros(a.shape, bool)
    m1[a < .5] = True
    m1[a <= 0] = False
    m2 = np.zeros(a.shape, bool)
    m2[a >= .5] = True
    m2[a >= 1] = False

    # Run the easing function based on the masks.
    a[m1] = -(2 ** (20 * a[m1] - 10) * np.sin((20 * a[m1] - 11.125) * c5))
    a[m1] = a[m1] / 2
    a[m2] = (2 ** (-20 * a[m2] + 10) * np.sin((20 * a[m2] - 11.125) * c5))
    a[m2] = a[m2] / 2 + 1
    return a


def in_out_quad(a: np.ndarray) -> np.ndarray:
    m = np.zeros(a.shape, bool)
    m[a < .5] = True
    a[m] = 2 * a[m] ** 2
    a[~m] = 1 - (-2 * a[~m] + 2) ** 2 / 2
    return a


def in_out_quint(a: np.ndarray) -> np.ndarray:
    """Perform the in out quint function on the array."""
    a[a < .5] = 16 * a[a < .5] ** 5
    a[a >= .5] = 1 - (-2 * a[a >= .5] + 2) ** 5 / 2
    return a


def in_out_sin(a: np.ndarray) -> np.ndarray:
    return -1 * (np.cos(np.pi * a) - 1) / 2


def in_out_cos(a: np.ndarray) -> np.ndarray:
    return -1 * (np.sin(np.pi * a) - 1) / 2


# Ease in functions.
@overflows
def in_back(a: np.ndarray) -> np.ndarray:
    c1 = 1.70158
    c3 = c1 + 1
    return c3 * a ** 3 - c1 * a ** 2


def in_circ(a: np.ndarray) -> np.ndarray:
    return 1 - np.sqrt(1 - a ** 2)


def in_cubic(a: np.ndarray) -> np.ndarray:
    """Perform the in quint easing function on the array."""
    return a ** 3


def in_elastic(a: np.ndarray) -> np.ndarray:
    c4 = (2 * np.pi) / 3
    m = np.zeros(a.shape, bool)
    m[a == 0] = True
    m[a == 1] = True
    a[~m] = -(2 ** (10 * a[~m] - 10)) * np.sin((a[~m] * 10 - 10.75) * c4)
    return a


def in_quad(a: np.ndarray) -> np.ndarray:
    return a ** 2


def in_quint(a: np.ndarray) -> np.ndarray:
    """Perform the in quint easing function on the array."""
    return a ** 5


def in_sine(a: np.ndarray) -> np.ndarray:
    return 1 - np.cos(a * np.pi / 2)


# Ease out functions.
def out_bounce(a: np.ndarray) -> np.ndarray:
    n1 = 7.5625
    d1 = 2.75

    a[a < 1 / d1] = n1 * a[a < 1 / d1] ** 2
    a[a < 2 / d1] = n1 * (a[a < 2 / d1] - 1.5 / d1) ** 2 + .75
    a[a < 2.5 / d1] = n1 * (a[a < 2.5 / d1] - 2.25 / d1) ** 2 + .9375
    a[a >= 2.5 / d1] = n1 * (a[a >= 2.5 / d1] - 2.625 / d1) ** 2 + .984375

    return a


def out_circ(a: np.ndarray) -> np.ndarray:
    return np.sqrt(1 - (a - 1) ** 2)


def out_cubic(a: np.ndarray) -> np.ndarray:
    return 1 - (1 - a) ** 3


@overflows
def out_elastic(a: np.ndarray) -> np.ndarray:
    c4 = (2 * np.pi) / 3
    m = np.zeros(a.shape, bool)
    m[a == 0] = True
    m[a == 1] = True
    a[~m] = 2 ** (-10 * a[~m]) * np.sin((a[~m] * 10 - .75) * c4) + 1
    return a


def out_quad(a: np.ndarray) -> np.ndarray:
    return 1 - (1 - a) ** 2


def out_quint(a: np.ndarray) -> np.ndarray:
    return 1 - (1 - a) ** 5


def out_sine(a: np.ndarray) -> np.ndarray:
    return np.sin(a * np.pi / 2)


# Other easing functions.
def mid_bump_linear(a: np.ndarray) -> np.ndarray:
    a = np.abs(a - .5)
    m = np.zeros(a.shape, bool)
    m[a < .25] = True
    a[m] = (.25 - a[m]) * 4
    a[~m] = 0
    return a


def mid_bump_sine(a: np.ndarray) -> np.ndarray:
    a = np.abs(a - .5)
    m = np.zeros(a.shape, bool)
    m[a < .25] = True
    a[m] = (.25 - a[m]) * 4
    a[~m] = 0
    return in_out_sin(a)


# Abbreviated function names both for registration and ease of use in
# command line configuration.
registered_functions = {
    '': linear,
    'l': linear,

    'io2': in_out_quad,
    'io3': in_out_cubic,
    'io5': in_out_quint,
    'ioa': in_out_back,
    'ioc': in_out_cos,
    'ioe': in_out_elastic,
    'ior': in_out_circ,
    'ios': in_out_sin,

    'i2': in_quad,
    'i3': in_cubic,
    'i5': in_quint,
    'ia': in_back,
    'ie': in_elastic,
    'is': in_sine,
    'ir': in_circ,

    'o2': out_quad,
    'o3': out_cubic,
    'o5': out_quint,
    'ob': out_bounce,
    'oe': out_elastic,
    'or': out_circ,
    'os': out_sine,

    'mbl': mid_bump_linear,
    'mbs': mid_bump_sine,
}


def get_regname_for_func(func: Callable) -> str:
    regnames = {registered_functions[k]: k for k in registered_functions}
    return regnames[func]


if __name__ == '__main__':
    from pjinoise.common import print_array
    import doctest
    doctest.testmod()

    a = [
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
    ]
    a = np.array(a)
    a = a / 0xff
    res = in_back(a)
    print_array(res)
