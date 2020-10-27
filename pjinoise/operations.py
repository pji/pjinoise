"""
operations
~~~~~~~~~~

Array operations for use when combining two layers of an image or 
animation.
"""
import numpy as np

def difference(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    return np.abs(a - b)