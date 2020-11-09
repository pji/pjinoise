"""
operations
~~~~~~~~~~

Array operations for use when combining two layers of an image or 
animation.
"""
import numpy as np


# Non-blends.
def replace(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    return b


# Darker/burn blends.
def darker(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    ab = a.copy()
    ab[b < a] = b[b < a]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


def multiply(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    if amount == 1:
        return a * b
    ab = a * b
    ab = a + (ab - a) * float(amount)
    return ab


def color_burn(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    """Taken from:
    http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
    """
    if amount == 1:
        return 1 - (1 - a) / b
    ab = 1 - (1 - a) / b
    return a + (ab - a) * float(amount)


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


def screen(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    rev_a = 1 - a
    rev_b = 1 - b
    ab = rev_a * rev_b
    if amount == 1:
        return 1 - ab
    return a + (ab - a) * float(amount)


def color_dodge(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    """Taken from:
    http://www.deepskycolors.com/archive/2010/04/21/formulas-for-Photoshop-blending-modes.html
    """
    if amount == 1:
        return a / (1 - b)
    ab = a / (1 - b)
    return a + (ab - a) * float(amount)


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


# Registration.
registered_ops = {
    'colorburn': color_burn,
    'colordodge': color_dodge,
    'darker': darker,
    'difference': difference,
    'lighter': lighter,
    'linearburn': linear_burn,
    'lineardodge': linear_dodge,
    'multiply': multiply,
    'overlay': overlay,
    'replace': replace,
    'screen': screen,
}

if __name__ == '__main__':
    raise NotImplementedError