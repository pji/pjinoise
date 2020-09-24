"""
filters
~~~~~~~

Postprocessing filters to use on noise images.
"""
from itertools import chain
from typing import Sequence


# Filters.
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


