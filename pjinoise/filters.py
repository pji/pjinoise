"""
filters
~~~~~~~

Postprocessing filters to use on noise images.
"""
from abc import ABC, abstractmethod
from itertools import chain
from typing import Mapping, Sequence, Tuple, Union

import cv2
from PIL import Image, ImageChops, ImageFilter, ImageOps
import numpy as np
from skimage.transform import swirl

from pjinoise.common import deserialize_sequence
from pjinoise.constants import COLOR, X, Y, Z
from pjinoise import ease as e
from pjinoise import generators as g
from pjinoise import operations as op


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
        attrs['type'] = get_regname_for_class(self)
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


class BoxBlur(ForLayer):
    def __init__(self, box_size: Union[str, int]) -> None:
        self.box_size = int(box_size)
    
    # Public methods.
    def process(self, a: np.ndarray, *args) -> np.ndarray:
        """Taken from:
        https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
        """
        size = self.box_size
        kernel = np.ones((size, size), float)/size ** 2
        for i in range(a.shape[Z]):
            blur = np.zeros(a.shape[Y:], dtype=a.dtype)
            blur = a[i]
            blur = cv2.filter2D(blur, -1, kernel)
            a[i] = blur
        return a


class Color(ForLayer):
    def __init__(self, 
                 colorkey: str = '', 
                 white: str = '', 
                 black: str = '',
                 invert: bool = False) -> None:
        if colorkey:
            white, black = COLOR[colorkey]
        if invert:
            white, black = black, white
        self.white = white
        self.black = black
    
    # Public methods.
    def process(self, a: np.ndarray, *args) -> np.ndarray:
        out = None
        a = (a * 0xff).astype(np.uint8)
        for i in range(a.shape[Z]):
            img = Image.fromarray(a[i], mode='L')
            img = ImageOps.colorize(img, self.black, self.white)
            img = img.convert('RGB')
            a_img = np.array(img, dtype=float)
            if out is None:
                out = np.zeros((a.shape[Z], *a_img.shape), dtype=np.uint8)
            out[i] = a_img
        return out


class Contrast(ForLayer):
    def process(self, a:np.ndarray, *args) -> np.ndarray:
        a_min = np.min(a)
        a_max = np.max(a)
        scale = a_max - a_min
        a -= a_min
        a /= scale
        return a


class CutLight(ForLayer):
    def __init__(self, threshold:float, 
                 ease:str = '', 
                 scale:float = 0xff) -> None:
        self.scale = float(scale)
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


class Inverse(ForLayer):
    def __init__(self, ease:str = '') -> None:
        self._ease = e.registered_functions[ease]
    
    # Public methods.
    def process(self, a:np.ndarray, *args) -> np.ndarray:
        return 1 - a


class LinearToPolar(ForLayer):
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
    
    def process(self, a:np.ndarray) -> np.ndarray:
        """ Based on code taken from:
        https://stackoverflow.com/questions/51675940/converting-an-image-from-cartesian-to-polar-limb-darkening
        """
        # cv2.warpPolar only works on two dimensional arrays. If the 
        # array is three dimensional, call process recursively with 
        # each two-dimensional slice of the three-dimensional array.
        if len(a.shape) == 3:
            for index in range(a.shape[Z]):
                a[index] = self.process(a[index])
            return a
        
        # Roll the image into polar coordinates.
        center = tuple([n / 2 for n in a.shape])
        max_radius = np.sqrt(sum(n ** 2 for n in center))
        flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
        return cv2.warpPolar(a, a.shape, center, max_radius, flags)


class Pinch(ForLayer):
    def __init__(self, amount:Union[float, str], 
                 radius:Union[float, str], 
                 scale:Union[Tuple[float], str], 
                 offset:Union[Tuple[float], str] = (0, 0, 0)):
        self.amount = np.float32(amount)
        self.radius = np.float32(radius)
        if isinstance(scale, str):
            scale = scale.split(',')
        self.scale = tuple(np.float32(n) for n in scale)
        if isinstance(offset, str):
            offset = offset.split(',')
        self.offset = tuple(np.float32(n) for n in offset)
    
    # Public methods.
    def process(self, a:np.ndarray, *args) -> np.ndarray:
        """Based on logic found here:
        https://stackoverflow.com/questions/64067196/pinch-bulge-distortion-using-python-opencv
        """
        scale = self.scale
        amount = self.amount
        radius = self.radius
        center = tuple((n) / 2 + o for n, o in zip(a.shape, self.offset))
        
        # set up the x and y maps as float32
        flex_x = np.zeros(a.shape[Y:], np.float32)
        flex_y = np.zeros(a.shape[Y:], np.float32)
        
        indices = np.indices(a.shape)
        y = indices[Y][0]
        x = indices[X][0]
        delta_y = scale[Y] * (y - center[Y])
        delta_x = scale[X] * (x - center[X])
        distance = delta_x ** 2 + delta_y ** 2
        
        r_mask = np.zeros(x.shape, bool)
        r_mask[distance >= radius ** 2] = True
        flex_x[r_mask] = x[r_mask]
        flex_y[r_mask] = y[r_mask]
        
        pmask = np.zeros(x.shape, bool)
        pmask[distance > 0.0] = True
        pmask[r_mask] = False
        factor = np.sin(np.pi * np.sqrt(distance) / radius / 2)
        factor[factor > 0] = factor[factor > 0] ** -amount
        factor[factor < 0] = -((-factor[factor < 0]) ** -amount)
        flex_x[pmask] = factor[pmask] * delta_x[pmask] /scale[X] + center[X]
        flex_y[pmask] = factor[pmask] * delta_y[pmask] /scale[Y] + center[Y]
        
        flex_x[~pmask] = 1.0 * delta_x[~pmask] /scale[X] + center[X]
        flex_y[~pmask] = 1.0 * delta_y[~pmask] /scale[Y] + center[Y]
        
        for i in range(a.shape[Z]):
            a[i] = cv2.remap(a[i], flex_x, flex_y, cv2.INTER_LINEAR)
        return a


class PolarToLinear(ForLayer):
    # Filter Protocol.
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
    
    def process(self, a:np.ndarray) -> np.ndarray:
        """ Based on code taken from:
        https://stackoverflow.com/questions/51675940/converting-an-image-from-cartesian-to-polar-limb-darkening
        """
        # cv2.linearPolar only works on two dimensional arrays. If the 
        # array is three dimensional, call process recursively with 
        # each two-dimensional slice of the three-dimensional array.
        if len(a.shape) == 3:
            for index in range(a.shape[Z]):
                a[index] = self.process(a[index])
            return a
        
        # Map the array into polar coordinates and then make those 
        # polar coordinates into linear coordinates.
        dim_half = tuple([n / 2 for n in a.shape])
        dim_pyth = np.sqrt(sum(n ** 2 for n in dim_half))
        a = cv2.linearPolar(a, dim_half, dim_pyth, cv2.WARP_FILL_OUTLIERS)
        return a.astype(float)


class Resize(ForLayer):
    """Generate layers at a different size than the final image 
    size. This can be used to reduce the generation time of expensive 
    generators or crop out awkward edges of generated images.
    """
    def __init__(self, new_size:Sequence[int], crop:bool = False) -> None:
        if crop == 'false':
            crop = False
        self.crop = crop
        if isinstance(new_size, str):
            new_size = new_size.split(',')
            new_size = [int(n) for n in new_size[::-1]]
        self.new_size = new_size
    
    # Public methods.
    def preprocess(self, size:Sequence[int], *args) -> Sequence[int]:
        """Determine the size the filter needs the image to be during 
        processing.
        """
        # The cv2.resize function is only two dimensional, so at 
        # this time Resize cannot handle resizing along the Z axis.
        if size[0] != self.new_size[0]:
            raise ValueError('Resize filter cannot change Z axis.')
        
        self.padding = [n - o for n, o in zip(self.new_size, size)]

        # Resize doesn't currently add padding when set to crop the 
        # image, so it can't handle situations where a dimension of 
        # the image is smaller than the original size when cropping.
        if self.crop and [n for n in self.padding if n < 0]:
            raise ValueError('Resize filter can\'t crop smaller images.')
        
        return self.new_size
    
    def process(self, a:np.ndarray) -> np.ndarray:
        """If the filter isn't cropping, resize the image. Otherwise, 
        return the image unchanged.
        """
        if self.crop:
            return a
        X, Y, Z = 2, 1, 0
        current_size = a.shape[:3]
        old_size = [n - p for n, p in zip(self.new_size, self.padding)]
        resized = np.zeros(old_size, dtype=a.dtype)
        for i in range(a.shape[Z]):
            frame = np.zeros(a.shape[Y:3], dtype=a.dtype)
            frame = a[i]
            resized[i] = cv2.resize(frame, (old_size[X], old_size[Y]))
        self.padding = [0 for _ in self.padding]
        return resized


class Ripple(ForLayer):
    def __init__(self, wavelength:Union[Sequence[float], str],
                 amplitude:Union[Sequence[float], str],
                 distort_axis:str = 'cross',
                 offset:Union[Sequence[float], str] = (0, 0, 0)) -> None:
        """Initialize an instance of the Ripple filter.
        
        :param wavelength: The distance between peaks in the distortion. 
            There needs to be one value in the sequence per dimension 
            in the image.
        :param amplitude: The amount of change caused by each ripple. 
            There needs to be one value in the sequence per dimension 
            in the image.
        :param distort_axis: Whether the distortion should be along the 
            same axis being distorted, causing the pattern to bunch up 
            like it is rippling, or along a different axis, causing the 
            pattern to wave like it's the cross-section of a wave. The 
            value "cross" uses different axis. By convention, use "same" 
            or an empty string to use the same axis.
        :param offset: (Optional.) The amount to offset the location 
            of the ripples in the image. There needs to be one value 
            in the sequence per dimension in the image. The default 
            value for all dimensions is zero.
        :return: None.
        :rtype: None.
        
        Note: The cv2.remap function used by this filter only ripples 
        in two dimensions, so right now any values set for the Z axis 
        are ignored. This may change in future versions, so it is 
        probably safest to set Z to zero in all parameters for now.
        """
        self.wave = deserialize_sequence(wavelength)
        self.amp = deserialize_sequence(amplitude)
        self.offset = deserialize_sequence(offset)
        self.distort_axis = (X, Y)
        if distort_axis == 'cross':
            self.distort_axis = (Y, X)
    
    # Public methods.
    def process(self, a:np.ndarray) -> np.ndarray:
        """Adapted from the example by jpmutant here:
        https://stackoverflow.com/questions/42732873
        """
        # Map out the volume of the given image and make sure everything is 
        # in float32 to keep the cv2.remap function happy.
        a = a.astype(np.float32)
        flex = np.indices(a.shape, np.float32)
        flex_x = flex[X].copy()
        flex_y = flex[Y].copy()
        
        # Modify the mapping to apply the ripple to create the flex 
        # maps for cv.remap. The flex map value for each pixel will 
        # indicate how far that pixel moves in the remapped image.
        da_x, da_y = self.distort_axis
        _, off_y, off_x = self.offset
        if self.wave[X]:
            flex_x = np.cos((off_x + flex[da_x]) / self.wave[X] * 2 * np.pi)
            flex_x = flex[X] + flex_x * self.amp[X]
        if self.wave[Y]:
            flex_y = np.cos((off_y + flex[da_y]) / self.wave[Y] * 2 * np.pi)
            flex_y = flex[Y] + flex_y * self.amp[Y]
        
        # Remap the color values in the original image using the 
        # rippled flex map.
        for i in range(a.shape[Z]):
            a[i] = cv2.remap(a[i], flex_x[i], flex_y[i], cv2.INTER_LINEAR)
        return a.astype(float)
        
        import math
        
        # Grab the dimensions of the image and calculate the center
        # of the image  (center not needed at this time)
        (t, h, w) = a.shape
        center = (w // 2, h // 2)

        # set up the x and y maps as float32
        flex_x = np.zeros((h,w),np.float32)
        flex_y = np.zeros((h,w),np.float32)

        # create simple maps with a modified assignment
        # the math modifier creates ripples.  increase the divisor for less waves, 
        # increase the multiplier for greater movement
        # this is where the magic is assembled
        for y in range(h):
            for x in range(w):
                flex_x[y,x] = x + math.cos(x/15) * 15
                flex_y[y,x] = y + math.cos(y/30) * 25


        # do the remap  this is where the magic happens      
        dst = cv2.remap(a[0],flex_x,flex_y,cv2.INTER_LINEAR)
        return dst


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
    
    def process(self, values:np.ndarray) -> np.ndarray:
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
    
    def process(self, values:np.ndarray) -> np.ndarray:
        """Run the filter over the image."""
        # The affine transform is only two dimensional, so if we're 
        # given three dimensions, call this recursively for every Z.
        if len(values.shape) > 2:
            for z in range(values.shape[Z]):
                values[z] = self.process(values[z])
            return values
        
        original_type = values.dtype
        values = values.astype(np.float32)
        
        # Create the transform matrix by defining three points in the 
        # original image, and then showing where they move to in the 
        # new, transformed image. The order of the axes is reversed 
        # for this in comparison to how it's generally used in pjinoise. 
        # This is due to the implementation of OpenCV.
        original = np.array([
            [0, 0],
            [values.shape[X] - 1, 0],
            [0, values.shape[Y] - 1],
        ]).astype(np.float32)
        new = np.array([
            [0, 0],
            [values.shape[X] - 1, 0],
            [(values.shape[Y] - 1) * self.slope, values.shape[Y] - 1],
        ]).astype(np.float32)
        
        # Perform the transform on the image by first creating a warp 
        # matrix from the example points. Then apply that matrix to 
        # the image, telling OpenCV to wrap pixels that are pushed off 
        # the edge of the image.
        matrix = cv2.getAffineTransform(original, new)
        values = cv2.warpAffine(values, 
                                matrix, 
                                (values.shape[X], values.shape[Y]), 
                                borderMode=cv2.BORDER_WRAP)
        return values.astype(original_type)


class Twirl(ForLayer):
    def __init__(self, radius: Union[str, float], 
                 strength: Union[str, float], 
                 offset: Union[str, Sequence[int]] = (0, 0, 0)) -> None:
        self.radius = float(radius)
        self.strength = float(strength)
        self.offset = deserialize_sequence(offset)
    
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
    
    def process(self, a:np.ndarray) -> np.ndarray:
        """ Based on code taken from:
        https://stackoverflow.com/questions/30448045
        """
        # Determine the location of the center of the twirl effect.
        center = [n / 2 + o for n, o in zip(a.shape[1:3], self.offset[1:3])]
        
        # Run the swirl filter.
        for i in range(a.shape[Z]):
            a[i] = swirl(a[i], center, self.strength, self.radius)
        return a


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


# Mixed filter classes:
class Autocontrast(ForImage, ForLayer):
    # Public methods.
    def process(self, img:Image.Image) -> Image.Image:
        if isinstance(img, Image.Image):
            return ImageOps.autocontrast(img)
        else:
            for i in range(img.shape[0]):
                a = np.zeros(img.shape[1:], dtype=img.dtype)
                a = img[i]
                a_img = Image.fromarray(a, mode='L')
                a_img = ImageOps.autocontrast(a_img)
                img[i] = np.array(a_img)
            return img


class Curve(ForImage, ForLayer):
    """Apply an easing curve to the given image."""
    def __init__(self, ease:str, scale:float = 0xff) -> None:
        self.ease = e.registered_functions[ease]
        self.scale = scale
    
    def process(self, img:Image.Image) -> Image.Image:
        if isinstance(img, Image.Image):
            if img.mode != 'L':
                raise NotImplementedError('Curve does not support images '
                                          f'with mode {img.mode}.')
            a = np.array(img)
            a = a / self.scale
            a = self.ease(a)
            a = a * self.scale
            a = np.around(a).astype(np.uint8)
            return Image.fromarray(a, mode='L')
        else:
            return self.ease(img)


class Grain(ForImage, ForLayer):
    def __init__(self, scale:float, *args, **kwargs):
        self.scale = float(scale)
        self._grain = None
        self._noise = noise.RangeNoise(127, self.scale)
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def process(self, img:Union[np.ndarray, 
                                Image.Image]) -> Union[np.ndarray, Image.Image]:
        """Perform the filter action on the image."""
        # Handle as a pillow Image for old ForImage interface.
        if isinstance(img, Image.Image):
            if not self._grain:
                size = (img.height, img.width)
                grain = self._noise.fill(size)
                grain = np.around(grain).astype(np.uint8)
                grain_image = Image.fromarray(grain, mode='L')
                if img.mode != 'L':
                    grain_img = grain_image.convert(img.mode)
                self._grain = grain_img
            return ImageChops.overlay(img, self._grain)
        
        # Handle as an ndarray for the ForLayer interface.
        if not self._grain:
            size = img.shape
            self._noise = g.Random(.5, self.scale)
            grain = self._noise.fill(size)
            self._grain = grain
        return op.overlay(img, self._grain)


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


# Layer filter processing functions.
def preprocess(size:Sequence[int], 
               filters:Sequence[Sequence[ForLayer]]) -> Tuple[int]:
    if not filters:
        return size
    new_size = size[:]
    for filter in filters:
        new_size = filter.preprocess(new_size, size)
#     try:
#         for layer in filters:
#             new_size = preprocess(new_size, layer)
#     except TypeError:
#         for filter in filters:
#             new_size = filter.preprocess(new_size, size)
    return new_size


def process(values:np.ndarray, 
            f_layers:Sequence[Sequence[ForLayer]],
            status:'ui.Status' = None) -> np.array:
    if not f_layers:
        return values
    
    try:
        for index, filters in enumerate(f_layers):
            for filter in filters:
                if status:
                    status.update('filter', filter.__class__.__name__)
                values[index] = filter.process(values[index])
                if status:
                    status.update('filter_end', filter.__class__.__name__)
    
    except TypeError:
        for filter in f_layers:
            values = filter.process(values)
    
    return values


def postprocess(values:np.ndarray, 
                f_layers:Sequence[Sequence[ForLayer]]) -> np.ndarray:
    # If there are no layer filters, bounce out.
    if not f_layers:
        return values
    
    # Find original size of the image.
    old_size = values.shape
    new_size = old_size[:]
    
    try:
        for filters in f_layers:
            for filter in filters:
                new_size = filter.postprocess(new_size)
    except TypeError:
        for filter in f_layers:
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
registered_filters = {
    'autocontrast': Autocontrast,
    'boxblur': BoxBlur,
    'color': Color,
    'contrast': Contrast,
    'cutshadow': CutShadow,
    'cutlight': CutLight,
    'inverse': Inverse,
    'lineartopolar': LinearToPolar,
    'pinch': Pinch,
    'polartolinear': PolarToLinear,
    'resize': Resize,
    'ripple': Ripple,
    'rotate90': Rotate90,
    'skew': Skew,
    'twirl': Twirl,
    'curve': Curve,
    'grain': Grain,
}
REGISTERED_IMAGE_FILTERS = {
    'autocontrast': Autocontrast,
    'blur': Blur,
    'colorize': Colorize,
    'curve': Curve,
    'grain': Grain,
    'overlay': Overlay,
}


# Registration and deserialization utility functions.
def make_filter(name:str, args:Sequence = ()) -> ForLayer:
    name = name.casefold()
    cls = registered_filters[name]
    return cls(*args)


def deserialize_filter(attrs: Mapping) -> ForLayer:
    cls = registered_filters[attrs['type']]
    del attrs['type']
    return cls(**attrs)


def get_regname_for_class(obj:object) -> str:
    regnames = {registered_filters[k]: k for k in registered_filters}
    clsname = obj.__class__
    return regnames[clsname]


if __name__ == '__main__':
#     raise NotImplemented
    
#     a = [
#         [
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#         ],
#         [
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#             [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
#         ],
#     ]
    a = [
        [
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0],
            [0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0],
            [0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0],
            [0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80],
            [0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60],
            [0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40],
            [0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40, 0x20],
            [0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40, 0x20, 0x00],
        ],
        [
            [0x00, 0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff],
            [0x20, 0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0],
            [0x40, 0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0],
            [0x60, 0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0],
            [0x80, 0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80],
            [0xa0, 0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60],
            [0xc0, 0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40],
            [0xe0, 0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40, 0x20],
            [0xff, 0xe0, 0xc0, 0xa0, 0x80, 0x60, 0x40, 0x20, 0x00],
        ],
    ]
    a = np.array(a, dtype=float)
    a = a / 0xff
    
#     srctri = np.array([[0, 0], [a.shape[X] - 1, 0], [0, a.shape[Y] - 1]])
#     srctri = srctri.astype(np.float32)
#     dsttri = np.array([
#         [0, a.shape[X] * .33],
#         [a.shape[X] * .85, a.shape[Y] * .25],
#         [a.shape[X] * .15, a.shape[Y] * 0.7]
#     ]).astype(np.float32)

#     srcpts = np.array([
#         [0, 0],
#         [4, 0],
#         [0, 4],
#     ]).astype(np.float32)
#     dstpts = np.array([
#         [0, 0],
#         [4, 0],
#         [4, 4],
#     ]).astype(np.float32)
#     warp_mat = cv2.getAffineTransform(srcpts, dstpts)
#     res = cv2.warpAffine(a, warp_mat, (a.shape[X], a.shape[Y]), borderMode=cv2.BORDER_WRAP)
    
#     skew = Skew(1)
#     res = skew.process(a)
    
#     obj = PolarToLinear()
#     obj = Inverse()
#     obj = Pinch('.75', '16', '5,5,5')
#     obj = Ripple('0,8,8','0,3.0,3.0', 'cross', '0,1,1')
#     size = (2, 13, 10)
#     obj = Resize((2, 9, 9))
    
#     obj = LinearToPolar()
#     obj = Twirl(3, 10)
    obj = BoxBlur(5)
    size = preprocess(a.shape, [obj,])
    res = obj.process(a)
    res = postprocess(res, [obj,])
    
    print(res.shape)
    res = np.around(res * 0xff).astype(np.uint)
    if len(res.shape) == 2:
        for y in res:
            print(' ' * 8, end='')
            r = [f'0x{x:02x}' for x in y]
            print('[' + ', '.join(r) + '],')
    
    if len(res.shape) == 3:
        for plane in res:
            print('    [')
            for row in plane:
                print(' ' * 8, end='')
                r = [f'0x{col:02x}' for col in row]
                print('[' + ', '.join(r) + '],')
            print('    ],')
