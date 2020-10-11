"""
filters
~~~~~~~

Postprocessing filters to use on noise images.
"""
from itertools import chain
from typing import Sequence

import numpy as np

from pjinoise.constants import X, Y, Z


# Vectorized filters.
def rotate90(values:np.array, direction:str='r') -> np.array:
    if direction == 'r':
        return np.rot90(values, -1, (-2, -1))
    if direction == 'l':
        return np.rot90(values, 1, (-2, -1))
    raise ValueError('Direction must be either r or l')


def rotate90_size_adjustment(size:Sequence[int]) -> Sequence[int]:
    def get_diff(size:Sequence[int], larger:int, smaller:int) -> Sequence[int]:
        diff = size[larger] - size[smaller]
        size[smaller] = size[smaller] + diff
        return size
    
    size = list(size[:])
    if size[Y] > size[X]:
        size = get_diff(size, Y, X)
    elif size[X] > size[Y]:
        size = get_diff(size, X, Y)
    return size


def skew(values:np.array, slope:float) -> np.array:
    def _perform_skew(y:int, x:int, amount:int, dim_x:int, z:int=None) -> float:
        amount = int(amount)
        new_x = (x + amount) % dim_x
        if z is None:
            return values[y][new_x]
        else:
            return values[z][y][new_x]
    
    shift = values.shape[Y] // (slope * 2)
    indices = np.indices(values.shape)
    amount = indices[Y] / slope
    perform_skew = np.vectorize(_perform_skew)
    
    if len(values.shape) == 2:
        values = np.roll(values, (0, shift))
        return perform_skew(indices[Y], indices[X], amount, values.shape[X])
    else:
        values = np.roll(values, (0, 0, shift))
        new_values = perform_skew(indices[Y], 
                                  indices[X], 
                                  amount, 
                                  values.shape[X], 
                                  indices[Z])
        return new_values

def skew_size_adjustment(size:Sequence[int], slope:float) -> Sequence[int]:
    size = list(size[:])
    padding = size[Y] // slope + 1
    size[X] = size[X] + padding * 2
    return size


# Not vectorized filters.
def cut_shadow(values:Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
    """Remove the lower half of colors from the image."""
    values = values[:]
    brightest = max(chain(col for col in values))
    darkest = 127
    for x in range(len(values)):
        for y in range(len(values[x])):
            if values[x][y] < 128:
                values[x][y] = 0
            else:
                new_color = 255 * (values[x][y] - 127) / 128
                values[x][y] = round(new_color)
    return values


def pixelate(matrix:list, size:int = 32) -> list:
    """Create squares of color from the image."""
    matrix = matrix[:]
    x_start, y_start = 0, 0
    while x_start < len(matrix):
        x_end = x_start + size
        if x_end > len(matrix):
            x_end = len(matrix) % size
        
        while y_start < len(matrix[x_start]):
            y_end = y_start + size
            if y_end > len(matrix[x_start]):
                y_end = len(matrix[x_start]) % size
            
            square = [[n for n in col[y_start:y_end]] 
                         for col in matrix[x_start:x_end]]
            average = sum(chain(*square)) / size ** 2
            color = round(average)
            
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    matrix[x][y] = color
            
            y_start +=size
        x_start += size
        y_start = 0
    return matrix


if __name__ == '__main__':
    raise NotImplemented