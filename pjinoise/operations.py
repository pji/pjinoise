"""
operations
~~~~~~~~~~

Array operations for use when combining two layers of an image or 
animation.
"""
import numpy as np

def difference(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    return np.abs(a - b)


def multiply(a:np.ndarray, b:np.array) -> np.ndarray:
    return a * b


def overlay(a:np.ndarray, b:np.array) -> np.ndarray:
    """This is based on the solution found here:
    https://stackoverflow.com/questions/52141987/overlay-blending-mode-in-python-efficiently-as-possible-numpy-opencv
    """
    mask = a >= .5
    ab = np.zeros_like(a)
    ab[~mask] = (2 * a * b)[~mask]
    ab[mask] = (1 - 2 * (1 - a) * (1 - b))[mask]
    return ab


def replace(a:np.ndarray, b:np.array) -> np.ndarray:
    return b


def screen(a:np.ndarray, b:np.array) -> np.ndarray:
    rev_a = 1 - a
    rev_b = 1 - b
    result = rev_a * rev_b
    return 1 - result


# Registration.
registered_ops = {
    'difference': difference,
    'multiply': multiply,
    'overlay': overlay,
    'replace': replace,
    'screen': screen,
}