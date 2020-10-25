"""
ease
~~~~

Basic easing function implementations. Easing functions taken from:

    https://easings.net
"""
import numpy as np


def linear(a:np.ndarray) -> np.ndarray:
    """Don't perform easing. This exists to avoid having to check if 
    easing is needed before sending to an easing function.
    """
    return a


# Ease in and out functions.
def in_out_cubic(a:np.ndarray) -> np.ndarray:
    """Perform the in out cubic easing function on the array."""
    a[a < .5] = 4 * a[a < .5] ** 3
    a[a >= .5] = 1 - (-2 * a[a >= .5] + 2) ** 3 / 2
    return a


# Ease in functions.
def in_cubic(a:np.array) -> np.array:
    """Perform the in quint easing function on the array."""
    return a ** 3


def in_quint(a:np.array) -> np.array:
    """Perform the in quint easing function on the array."""
    return a ** 5


# Ease out functions.
def out_bounce(a:np.array) -> np.array:
    n1 = 7.5625
    d1 = 2.75
    
    a[a < 1 / d1] = n1 * a[a < 1 / d1] ** 2
    a[a < 2 / d1] = n1 * (a[a < 2 / d1] - 1.5 / d1) ** 2 + .75
    a[a < 2.5 / d1] = n1 * (a[a < 2.5 / d1] - 2.25 / d1) ** 2 + .9375
    a[a >= 2.5 / d1] = n1 * (a[a >= 2.5 / d1] - 2.625 / d1) ** 2 + .984375
    
    return a


# Abbreviated function names both for registration and ease of use in 
# command line configuration.
registered_functions = {
    '': linear,
    'ic': in_cubic,
    'ioc': in_out_cubic,
    'iq': in_quint,
    'l': linear,
    'ob': out_bounce,
}

if __name__ == '__main__':
    a = [
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
    ]
    a = np.array(a)
    a = a / 0xff
    
    res = in_cubic(a)
    
    res = res * 0xff
    res = np.around(res).astype(int)
    for y in res:
        print(' ' * 8, end='')
        r = [f'0x{x:02x}' for x in y]
        print('[' + ', '.join(r) + '],')
