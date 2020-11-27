"""
ease
~~~~

Basic easing function implementations. Easing functions taken from:

    https://easings.net
"""
from typing import Callable

import numpy as np


def linear(a:np.ndarray) -> np.ndarray:
    """Don't perform easing. This exists to avoid having to check if 
    easing is needed before sending to an easing function.
    """
    return a


# Ease in and out functions.
def in_out_back(a:np.ndarray) -> np.ndarray:
    c1 = 1.70158
    c2 = c1 * 1.525
    m = np.zeros(a.shape, bool)
    m[a < .5] = True
    a[m] = (2 * a[m]) ** 2 * ((c2 + 1) * 2 * a[m] - c2) / 2
    a[~m] = ((2 * a[~m] - 2) ** 2 * ((c2 + 1) * (a[~m] * 2 - 2) + c2) + 2) / 2
    return a


def in_out_circ(a:np.ndarray) -> np.ndarray:
    m = np.zeros(a.shape, bool)
    m[a < .5] = True
    a[m] = (1 - np.sqrt(1 - (2 * a[m]) ** 2)) / 2
    a[~m] = (np.sqrt(1 - (-2 * a[~m] + 2) ** 2) + 1) / 2
    return a


def in_out_cubic(a:np.ndarray) -> np.ndarray:
    """Perform the in out cubic easing function on the array."""
    a[a < .5] = 4 * a[a < .5] ** 3
    a[a >= .5] = 1 - (-2 * a[a >= .5] + 2) ** 3 / 2
    return a


def in_out_elastic(a:np.ndarray) -> np.ndarray:
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


def in_out_quad(a:np.ndarray) -> np.ndarray:
    m = np.zeros(a.shape, bool)
    m[a < .5] = True
    a[m] = 2 * a[m] ** 2
    a[~m] = 1 - (-2 * a[~m] + 2) ** 2 / 2
    return a


def in_out_quint(a:np.ndarray) -> np.ndarray:
    """Perform the in out quint function on the array."""
    a[a < .5] = 16 * a[a < .5] ** 5
    a[a >= .5] = 1 - (-2 * a[a >= .5] + 2) ** 5 / 2
    return a


def in_out_sin(a:np.ndarray) -> np.ndarray:
    return -1 * (np.cos(np.pi * a) - 1) /2


def in_out_cos(a:np.ndarray) -> np.ndarray:
    return -1 * (np.sin(np.pi * a) - 1) /2


# Ease in functions.
def in_circ(a:np.ndarray) -> np.ndarray:
    return 1 - np.sqrt(1 - a ** 2)


def in_cubic(a:np.ndarray) -> np.ndarray:
    """Perform the in quint easing function on the array."""
    return a ** 3


def in_elastic(a:np.ndarray) -> np.ndarray:
    c4 = (2 * np.pi) / 3
    m = np.zeros(a.shape, bool)
    m[a == 0] = True
    m[a == 1] = True
    a[~m] = -(2 ** (10 * a[~m] - 10)) * np.sin((a[~m] * 10 - 10.75) * c4)
    return a


def in_quad(a:np.ndarray) -> np.ndarray:
    return a ** 2


def in_quint(a:np.ndarray) -> np.ndarray:
    """Perform the in quint easing function on the array."""
    return a ** 5


def in_sine(a:np.ndarray) -> np.ndarray:
    return 1 - np.cos(a * np.pi / 2)


# Ease out functions.
def out_bounce(a:np.ndarray) -> np.ndarray:
    n1 = 7.5625
    d1 = 2.75
    
    a[a < 1 / d1] = n1 * a[a < 1 / d1] ** 2
    a[a < 2 / d1] = n1 * (a[a < 2 / d1] - 1.5 / d1) ** 2 + .75
    a[a < 2.5 / d1] = n1 * (a[a < 2.5 / d1] - 2.25 / d1) ** 2 + .9375
    a[a >= 2.5 / d1] = n1 * (a[a >= 2.5 / d1] - 2.625 / d1) ** 2 + .984375
    
    return a


def out_circ(a:np.ndarray) -> np.ndarray:
    return np.sqrt(1 - (a - 1) ** 2)


def out_cubic(a:np.ndarray) -> np.ndarray:
    return 1 - (1 - a) ** 3


def out_elastic(a:np.ndarray) -> np.ndarray:
    c4 = (2 * np.pi) / 3
    m = np.zeros(a.shape, bool)
    m[a == 0] = True
    m[a == 1] = True
    a[~m] = 2 ** (-10 * a[~m]) * np.sin((a[~m] * 10 - .75) * c4) + 1
    return a


def out_quad(a:np.ndarray) -> np.ndarray:
    return 1 - (1 - a) ** 2


def out_quint(a:np.ndarray) -> np.ndarray:
    return 1 - (1 - a) ** 5


def out_sine(a:np.ndarray) -> np.ndarray:
    return np.sin(a * np.pi / 2)


# Other easing functions.
def mid_bump_linear(a:np.ndarray) -> np.ndarray:
    a = np.abs(a - .5)
    m = np.zeros(a.shape, bool)
    m[a < .25] = True
    a[m] = (.25 - a[m]) * 4
    a[~m] = 0
    return a


def mid_bump_sine(a:np.ndarray) -> np.ndarray:
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


def get_regname_for_func(func:Callable) -> str:
    regnames = {registered_functions[k]: k for k in registered_functions}
    return regnames[func]


if __name__ == '__main__':
    a = [
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
    ]
    a = np.array(a)
    a = a / 0xff
    
    res = mid_bump_sine(a)
    
    res = res * 0xff
    res = np.around(res).astype(int)
    for y in res:
        print(' ' * 8, end='')
        r = [f'0x{x:02x}' for x in y]
        print('[' + ', '.join(r) + '],')
