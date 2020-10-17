"""
filters
~~~~~~~

Postprocessing filters to use on noise images.
"""
from abc import ABC, abstractmethod
from itertools import chain
from typing import Sequence, Tuple, Union

import numpy as np

from pjinoise.constants import X, Y, Z


# Filter classes.
class Filter(ABC):
    """The base class for filter objects."""
    padding = None
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.asdict() == other.asdict()
    
    # Public methods.
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = self.__dict__.copy()
        attrs['type'] = self.__class__.__name__
        return attrs
    
    def preprocess(self, size:Sequence[int], *args) -> Sequence[int]:
        """Determine the size the filter needs the image to be during 
        processing.
        """
        return size
    
    @abstractmethod
    def process(self, values:np.array) -> np.array:
        """Run the filter over the image."""
    
    def postprocess(self, size:Sequence[int], *args) -> Sequence[int]:
        """Return the original size of the image."""
        if self.padding is None:
            return size
        return tuple(n - pad for n, pad in zip (size, self.padding))


class CutLight(Filter):
    def __init__(self, threshold:float, scale:float = 256) -> None:
        self.threshold = threshold
        self.scale = scale
    
    # Public methods.
    def process(self, values:np.array, *args) -> np.array:
        values = values.copy()
        values = self.scale - (values + 1)
        values = values - self.threshold
        values[values < 0] = 0
        threshold_scale = self.scale - self.threshold
        values = self.threshold - (values + 1)
        return (values / threshold_scale) * self.scale


class CutShadow(Filter):
    def __init__(self, threshold:float, scale:float = 256) -> None:
        self.threshold = threshold
        self.scale = scale
    
    # Public methods.
    def process(self, values:np.array, *args) -> np.array:
        values = values.copy()
        values = values - self.threshold
        values[values < 0] = 0
        threshold_scale = self.scale - self.threshold
        return (values / threshold_scale) * self.scale


class Rotate90(Filter):
    """Rotate the image ninety degrees in the given direction."""
    def __init__(self, direction:str) -> None:
        self.direction = direction
    
    # Filter protocol.
    def preprocess(self, size:Sequence[int], *args) -> Sequence[int]:
        """Determine the size the filter needs the image to be during 
        processing.
        """
        new_size = list(size[:])
        if size[Y] > size[X]:
            new_size[X] = size[Y]
        elif size[Y] < size[X]:
            new_size[Y] = size[X]
        self.padding = tuple([new - old for new, old in zip(new_size, size)])
        return tuple(new_size)         
    
    def process(self, values:np.array) -> np.array:
        """Run the filter over the image."""
        direction = 1
        if self.direction == 'r':
            direction = -1
        return np.rot90(values, direction, (Y, X))


class Skew(Filter):
    """Skew the image."""
    def __init__(self, slope:Union[float, str]) -> None:
        self.slope = float(slope)
    
    # Public methods.
    def preprocess(self, size:Sequence[int], 
                   original_size:Sequence[int], *args) -> Sequence[int]:
        """Determine the size the filter needs the image to be during 
        processing.
        """
        # Determine the amount of skew needed for this skew's slope.
        padding = (original_size[Y] - 1) // self.slope
        if original_size[Y] % self.slope:
            padding += 2
        other_padding = size[X] - original_size[X]
        
        # If the image already has then much padding, no new padding 
        # is needed.
        if other_padding >= padding:
            return size
        
        # Otherwise add the new padding to the image.
        new_size = list(original_size[:])
        new_size[X] = new_size[X] + padding
        self.padding = tuple([new - old for new, old in zip(new_size, size)])
        self.padding = tuple(int(n) for n in self.padding)
        return tuple(int(n) for n in new_size)
    
    def process(self, values:np.array) -> np.array:
        """Run the filter over the image."""
        def _lookup(*index) -> float:
            return values[index]

        # Determine how much skew for each row.
        skew_by_row = np.indices((values.shape[Y],)) - (values.shape[Y] // 2)
        skew_by_row = np.round(skew_by_row * self.slope)
        skew_by_row = np.reshape(skew_by_row, (values.shape[Y], 1))
        
        # Determine the amount of skew for each position.
        skew_by_row = np.tile(skew_by_row, (1, values.shape[X]))
        indices = np.indices(values.shape)
        indices[X] = (indices[X] + skew_by_row) % values.shape[X]
        del skew_by_row
        
        # Return the skewed image.
        lookup = np.vectorize(_lookup)
        return lookup(*indices)


# Factories.
def make_filter(name:str, args:Sequence = ()) -> Filter:
    name = name.casefold()
    cls = REGISTERED_FILTERS[name]
    return cls(*args)


# Processing functions.
def preprocess(size:Sequence[int], 
               f_layers:Sequence[Sequence[Filter]]) -> Tuple[int]:
    new_size = size[:]
    for filters in f_layers:
        for filter in filters:
            new_size = filter.preprocess(new_size, size)
    return new_size


def process(values:np.array, 
            f_layers:Sequence[Sequence[Filter]],
            status:'ui.Status' = None) -> np.array:
    for index, filters in enumerate(f_layers):
        for filter in filters:
            if status:
                status.update('filter', filter.__class__.__name__)
            values[index] = filter.process(values[index])
            if status:
                status.update('filter_end', filter.__class__.__name__)
    return values


def postprocess(values:np.array, 
                f_layers:Sequence[Sequence[Filter]]) -> np.array:
    # Find original size of the image.
    old_size = values.shape
    new_size = old_size[:]
    for filters in f_layers:
        for filter in filters:
            new_size = filter.postprocess(new_size)
    
    # Crop the image back to the original size.
    pads = [old - new for old, new in zip(old_size, new_size)]
    starts = [pad // 2 for pad in pads]
    ends = [old - (pad - start) for old, pad, start 
            in zip(old_size, pads, starts)]
    slices = [slice(start, end) for start, end in zip(starts, ends)]
    return values[tuple(slices)]


# Not vectorized filters.
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


# Registrations.
REGISTERED_FILTERS = {
    'cutshadow': CutShadow,
    'cutlight': CutLight,
    'rotate90': Rotate90,
    'skew': Skew,
}

if __name__ == '__main__':
    raise NotImplemented