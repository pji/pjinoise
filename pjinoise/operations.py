"""
operations
~~~~~~~~~~

Array operations for use when combining two layers of an image or 
animation.
"""
import numpy as np


# Non-blends.
def replace(a:np.ndarray, b:np.array) -> np.ndarray:
    return b


# Darker/burn blends.
def darker(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    ab = a.copy()
    ab[b < a] = b[b < a]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


def multiply(a:np.ndarray, b:np.array, amount:float = 1) -> np.ndarray:
    if amount == 1:
        return a * b
    ab = a * b
    ab = a + (ab - a) * float(amount)
    return ab


# Lighter/dodge blends.
def lighter(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    ab = a.copy()
    ab[b > a] = b[b > a]
    if amount == 1:
        return ab
    return a + (ab - a) * float(amount)


def screen(a:np.ndarray, b:np.array, amount:float = 1) -> np.ndarray:
    rev_a = 1 - a
    rev_b = 1 - b
    result = rev_a * rev_b
    if amount == 1:
        return 1 - result
    return a + (ab - a) * float(amount)


# Mixed blends.
def difference(a:np.ndarray, b:np.ndarray, amount:float = 1) -> np.ndarray:
    if amount == 1:
        return np.abs(a - b)
    ab = np.abs(a - b)
    ab = a + (ab - a) * float(amount)
    return ab


def overlay(a:np.ndarray, b:np.array, amount:float = 1) -> np.ndarray:
    """This is based on the solution found here:
    https://stackoverflow.com/questions/52141987/overlay-blending-mode-in-python-efficiently-as-possible-numpy-opencv
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
    'darker': darker,
    'difference': difference,
    'lighter': lighter,
    'multiply': multiply,
    'overlay': overlay,
    'replace': replace,
    'screen': screen,
}

if __name__ == '__main__':
    raise NotImplementedError