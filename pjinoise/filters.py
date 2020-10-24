"""
filters
~~~~~~~

Postprocessing filters to use on noise images.
"""
from abc import ABC, abstractmethod
from itertools import chain
from typing import Sequence, Tuple, Union

from PIL import Image, ImageChops, ImageFilter, ImageOps
import numpy as np

from pjinoise.constants import X, Y, Z
from pjinoise import ease as e
from pjinoise import noise


# Layer filter classes.
class ForLayer(ABC):
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


class CutLight(ForLayer):
    def __init__(self, threshold:float, 
                 ease:str = '', 
                 scale:float = 0xff) -> None:
        self.scale = scale
        self.threshold = threshold
        self._ease = e.registered_functions[ease]
    
    # Public methods.
    def process(self, values:np.array, *args) -> np.array:
        values[values > self.threshold] = self.threshold
        values = values / self.threshold
        values = self._ease(values) * self.scale
        values = values.astype(int)
        return values


class CutShadow(CutLight):
    # Public methods.
    def process(self, values:np.array, *args) -> np.array:
        values[values < self.threshold] = 0
        values[values >= self.threshold] -= self.threshold
        threshold_scale = self.scale - self.threshold
        values = values / threshold_scale
        values = self._ease(values) * self.scale
        values = values.astype(int)
        return values


class Rotate90(ForLayer):
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


class Skew(ForLayer):
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
        raveled_values = np.ravel(values)
        raveled_index = np.ravel_multi_index(indices, values.shape)
        if len(raveled_index.shape) != 1:
            raveled_index = np.ravel(raveled_index)
        skewed = np.take(values, raveled_index)
        return np.reshape(skewed, values.shape)


# Image filter classes.
class ForImage(ABC):
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.asdict() == other.asdict()
    
    # Public methods.
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = self.__dict__.copy()
        attrs = {k: attrs[k] for k in attrs if not k.startswith('_')}
        attrs['type'] = self.__class__.__name__
        return attrs
    
    @abstractmethod
    def process(self, values:np.ndarray) -> np.ndarray:
        """Run the filter over the image."""


class Autocontrast(ForImage):
    # Public methods.
    def process(self, img:Image.Image) -> Image.Image:
        return ImageOps.autocontrast(img)


class Blur(ForImage):
    def __init__(self, amount:float, *args, **kwargs):
        self.amount = amount
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def process(self, img:Image.Image) -> Image.Image:
        blur = ImageFilter.GaussianBlur(self.amount)
        return img.filter(blur)


class Colorize(ForImage):
    def __init__(self, white:str, black:str, *args, **kwargs):
        self.white = white
        self.black = black
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def process(self, img:Image.Image) -> Image.Image:
        return ImageOps.colorize(img, self.black, self.white)


class Curve(ForImage):
    """Apply an easing curve to the given image."""
    def __init__(self, ease:str, scale:float = 0xff) -> None:
        self.ease = e.registered_functions[ease]
        self.scale = scale
    
    def process(self, img:Image.Image) -> Image.Image:
        if img.mode != 'L':
            raise NotImplementedError('Curve does not support images '
                                      f'with mode {img.mode}.')
        a = np.array(img)
        a = a / self.scale
        a = self.ease(a)
        a = a * self.scale
        a = np.around(a).astype(np.uint8)
        return Image.fromarray(a, mode='L')


class Grain(ForImage):
    def __init__(self, scale:float, *args, **kwargs):
        self.scale = scale
        self._grain = None
        self._noise = noise.RangeNoise(127, self.scale)
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def process(self, img:Image.Image) -> Image.Image:
        if not self._grain:
            size = (img.height, img.width)
            grain = self._noise.fill(size)
            grain = np.around(grain).astype(np.uint8)
            grain_image = Image.fromarray(grain, mode='L')
            if img.mode != 'L':
                grain_img = grain_image.convert(img.mode)
            self._grain = grain_img
        return ImageChops.overlay(img, self._grain)


class Overlay(ForImage):
    def __init__(self, amount:float = .2, *args, **kwargs):
        self.amount = amount
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def process(self, img:Image.Image) -> Image.Image:
        mode = img.mode
        full = ImageChops.overlay(img, img)
        original = np.array(img)
        part = (original - np.array(full)) * self.amount
        result = np.around(part + original).astype(np.uint8)
        return Image.fromarray(result, mode=mode)


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


# Factories.
def make_filter(name:str, args:Sequence = ()) -> ForLayer:
    name = name.casefold()
    cls = REGISTERED_FILTERS[name]
    return cls(*args)


# Layer filter processing functions.
def preprocess(size:Sequence[int], 
               f_layers:Sequence[Sequence[ForLayer]]) -> Tuple[int]:
    if not f_layers:
        return size
    new_size = size[:]
    for filters in f_layers:
        for filter in filters:
            new_size = filter.preprocess(new_size, size)
    return new_size


def process(values:np.ndarray, 
            f_layers:Sequence[Sequence[ForLayer]],
            status:'ui.Status' = None) -> np.array:
    if not f_layers:
        return values
    for index, filters in enumerate(f_layers):
        for filter in filters:
            if status:
                status.update('filter', filter.__class__.__name__)
            values[index] = filter.process(values[index])
            if status:
                status.update('filter_end', filter.__class__.__name__)
    return values


def postprocess(values:np.ndarray, 
                f_layers:Sequence[Sequence[ForLayer]]) -> np.array:
    # If there are no layer filters, bounce out.
    if not f_layers:
        return values
    
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


# Image filter processing functions.
def process_image(img:Image.Image, 
                  filters:Sequence[ForImage],
                  status:'ui.Status' = None) -> Image.Image:
    """Run the given filters on the image."""
    for filter in filters:
        if status:
            status.update('filter', f.__class__.__name__)
        img = filter.process(img)
        if status:
            status.update('filter_end', f.__class__.__name__)
    return img


# Registrations.
REGISTERED_FILTERS = {
    'cutshadow': CutShadow,
    'cutlight': CutLight,
    'rotate90': Rotate90,
    'skew': Skew,
}
REGISTERED_IMAGE_FILTERS = {
    'autocontrast': Autocontrast,
    'blur': Blur,
    'colorize': Colorize,
    'grain': Grain,
    'overlay': Overlay,
}


if __name__ == '__main__':
#     raise NotImplemented
    
    a = [
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
        [0x40, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
    ]
    a = np.array(a, dtype=np.uint8)
    
    f = CutLight(0x80, 'q')
    res = f.process(a)
    
    for y in res:
        print(' ' * 8, end='')
        r = [f'0x{x:02x}' for x in y]
        print('[' + ', '.join(r) + '],')
