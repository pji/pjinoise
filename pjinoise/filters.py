"""
filters
~~~~~~~

Postprocessing filters to use on image data.


Basic Usage: Filter Objects
===========================
A Filter object is used to alter image data. Creating an instance of
a subclass of Filter (a "filter") works like instantiating any other
Python class. The specific parameters vary based on what the specific
filter does.

Usage::

    >>> fltr = BoxBlur(5)
    >>> fltr
    BoxBlur(box_size=5)


process()
---------
The process() method is used to alter image data. It always adheres
to the following protocol:

    :param values: The numpy n-dimensional array of image data to
        alter. This should be able to two different types of arrays:
        a three-dimensional array of floats within the range of
        0 <= x <= 1 and a four-dimensional array of unsigned eight-
        bit integers.
    :return: In most cases this will return a numpy n-dimensional
        array of the same shape and data type it was given. However,
        certain filters, such as the Color filter will return array
        with a different shape or data type.
    :rtype: numpy.ndarray

Usage::

    >>> import numpy as np
    >>>
    >>> a = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    >>> fltr = Flip('v')
    >>> fltr.process(a)
    array([[[1., 1., 1.],
            [0., 0., 0.]]])


preprocess() and postprocess()
------------------------------
The preprocess() and postprocess() methods are used to handle image
data generators (sources) in cases where the filter has specific
requirements about the data it will alter. The primary use case is
to change the size of the generated data to avoid undesired artifacts
of the alteration in the altered image data.

The preprocess() method sets the amount of padding that needs to be
added to the size of the image in order to avoid the artifacts. It
also saves the amount of padding internally, so it can be removed by
the postprocess() method. The preprocess() function always adheres
to the following protocol:

    :param size: The current size of the image to be created.
    :param original_size: The original size of the image to be
        created. This is used in situations where multiple
        filters are being applied to an array of image data.
        If a previous filter had added padding to the image,
        that padding wouldn't need to be added by this filter.
    :return: The new size of the image as a :class:tuple object.
    :rtype: tuple

The postprocess() method returns the size the image should be with
the padding added by the preprocess() method removed. The postprocess()
method always adheres to the following protocol:

    :param size: The current size of the array of image data.
    :return: The new size of the image with any padding removed
        as a :class:tuple object.
    :rtype: tuple

Usage::

    >>> fltr = Skew(.5)
    >>> size = (1, 720, 1280)
    >>> new_size = fltr.preprocess(size, size)
    >>> new_size
    (1, 720, 2718)
    >>> old_size = fltr.postprocess(new_size)
    >>> old_size
    (1, 720, 1280)


asdict()
--------
The asdict() method is primarily used to make serializing source
objects to JSON easier. It also can be used in testing to compare
two source objects. The following is true of the output of asdict():

*   The "type" key contains the serialization name for the object's
    class.
*   The other keys are the parameters for creating a new instance
    of the objects class that is a copy of the object.
*   The values of those parameters are data types that can be
    serialized by the json module.

Note: When creating subclasses of Filter, care should be taken to
ensure the subclass's asdict() method only returns data that can
be serialized by the json module. There are a couple of ways to
accomplish this:

*   Filter.asdict() won't return private attributes in the output
    dictionary, so the serialized version can be stored in the
    public attribute (or at least returned by calls to the public
    attribute) and the deserialized version stored in a private
    attribute.
*   The asdict() method can be subclassed to serialize the value of
    the attribute before the output dictionary is returned.

Usage:

    >>> glow = Glow(50)
    >>> glow.asdict()
    {'start_sigma': 50, 'type': 'glow'}


Processing Helpers
==================
The filters module has a few capabilities to help with processing
filters. The preprocess(), process(), and postprocess() functions
are used to handle a sequence of filters that will be applied to
an array of image data.

Usage::

    >>> import numpy as np
    >>>
    >>> size = (1, 4, 4)
    >>> fltrs = [Skew(.5), Inverse(),]
    >>> new_size = preprocess(size, fltrs)
    >>> new_size
    (1, 4, 10)
    >>>
    >>> a = np.zeros(new_size, dtype=float)
    >>> a
    array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    >>>
    >>> a = process(a, fltrs)
    >>> a
    array([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
    >>>
    >>> a = postprocess(a, fltrs)
    >>> a
    array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]])


Serialization Helpers
=====================
The filters module has a few capabilities to help with serialization.
The get_regname_for_class() function returns a string that represents
a filter class that has been registered with the filters module.

Usage::

    >>> obj = Color(colorkey='b')
    >>> get_regname_for_class(obj)
    'color'

New source classes can be registered with the source module by adding
them to the registered_filters dictionary. The key should be the
short string you want to use as the serialized value of the class.

Usage::

    >>> class Spam(Filter):
    ...     def process(*args):
    ...         pass
    ...
    >>> registered_filters['spam'] = Spam
    >>> spam = Spam()
    >>> get_regname_for_class(spam)
    'spam'

The primary purpose for this feature is to allow classes to be easily
deserialized from JSON objects. It also should provide a measure of
input validation at deserialization to reduce the risk of remote code
execution vulnerabilities though deserialization.
"""
from abc import ABC, abstractmethod
from functools import wraps
from itertools import chain
from typing import Callable, Mapping, Sequence, Tuple, Union

import cv2
from PIL import Image, ImageChops, ImageFilter, ImageOps
import numpy as np
from skimage.transform import swirl

from pjinoise.constants import COLOR, X, Y, Z
from pjinoise import common as c
from pjinoise import ease as e
from pjinoise import operations as op
from pjinoise import sources as s


# Decorators
def channeled(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(obj, a: np.ndarray, *args, **kwargs) -> np.ndarray:
        channeled = False
        if len(a.shape) == 3:
            return fn(obj, a, *args, **kwargs)
        out = np.zeros_like(a)
        for i in range(a.shape[-1]):
            channel = np.zeros(a.shape[:-1], dtype=a.dtype)
            channel[:, :] = a[:, :, :, i]
            channel = fn(obj, channel, *args, **kwargs)
            out[:, :, :, i] = channel[:, :]
        return out
    return wrapper


def scaled(fn: Callable) -> Callable:
    """Operations with multiplication rely on values being scaled to
    0 ≤ x ≤ 1 to keep the result from overflowing. Operations that add
    or subtract by one rely on that same scaling. Many color spaces
    are scaled to 0x00 ≤ x ≤ 0xff, so this will attempt to detect
    those images, rescale to one for the operation, and then rescale
    back to 0xff after the operation.
    """
    @wraps(fn)
    def wrapper(obj, a: np.ndarray, *args, **kwargs) -> np.ndarray:
        rescaled = False
        if len(a.shape) == 4:
            a = a.astype(float) / 0xff
            assert np.max(a) <= 1.0
            rescaled = True
        a = fn(obj, a, *args, **kwargs)
        if rescaled:
            assert np.max(a) <= 1.0
            a = np.around(a * 0xff).astype(np.uint8)
        if not rescaled and np.max(a) > 1.0:
            a = a / np.max(a)
        return a
    return wrapper


# Layer filter classes.
class Filter(ABC):
    """The base class for filter objects."""
    padding = None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.asdict() == other.asdict()

    def __repr__(self):
        cls = self.__class__.__name__
        attrs = self.asdict()
        del attrs['type']
        params = []
        for key in attrs:
            val = attrs[key]
            if isinstance(val, str):
                val = f"'{val}'"
            if isinstance(val, bytes):
                val = f"b'{val}'"
            params.append(f'{key}={val}')
        params_str = ', '.join(params)
        return f'{cls}({params_str})'

    # Public methods.
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = self.__dict__.copy()
        attrs['type'] = get_regname_for_class(self)
        if 'ease' in attrs:
            attrs['ease'] = e.get_regname_for_func(attrs['ease'])

        # This is to address a problem with the Pinch filter. It
        # likely be fixed in the Pinch filter instead.
        for key in attrs:
            if isinstance(attrs[key], np.float32):
                attrs[key] = float(attrs[key])
            if (isinstance(attrs[key], Sequence)
                    and not isinstance(attrs[key], str)):
                try:
                    if isinstance(attrs[key][0], np.float32):
                        attrs[key] = [float(n) for n in attrs[key]]
                except IndexError:
                    raise TypeError(type(attrs[key]))
        if 'padding' in attrs:
            del attrs['padding']

        pvt_keys = [key for key in attrs if key.startswith('_')]
        for key in pvt_keys:
            del attrs[key]
        return attrs

    def preprocess(self, size: Sequence[int], *args) -> Sequence[int]:
        """Determine the size the filter needs the image to be during
        processing.
        """
        return size

    @abstractmethod
    def process(self, values: np.array) -> np.array:
        """Run the filter over the image."""

    def postprocess(self, size: Sequence[int], *args) -> Sequence[int]:
        """Return the original size of the image."""
        if self.padding is None:
            return size
        return tuple(n - pad for n, pad in zip(size, self.padding))


class BoxBlur(Filter):
    def __init__(self, box_size: Union[str, int]) -> None:
        self.box_size = int(box_size)

    # Public methods.
    @scaled
    def process(self, a: np.ndarray, *args) -> np.ndarray:
        """Taken from:
        https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
        """
        size = self.box_size
        kernel = np.ones((size, size), float) / size ** 2
        for i in range(a.shape[Z]):
            blur = np.zeros(a.shape[Y:], dtype=a.dtype)
            blur = a[i]
            blur = cv2.filter2D(blur, -1, kernel)
            a[i] = blur

        return a


class Color(Filter):
    def __init__(self,
                 colorkey: str = '',
                 white: str = '',
                 black: str = '',
                 invert: bool = False,
                 src_space: str = '',
                 dst_space: str = 'RGB') -> None:
        if colorkey:
            white, black = COLOR[colorkey]
        if invert:
            white, black = black, white
        self.white = white
        self.black = black
        self.src_space = src_space
        self.dst_space = dst_space

    # Public methods.
    def process(self, a: np.ndarray, *args) -> np.ndarray:
        out = None
        src_space = self.src_space
        if self.src_space == '':
            assert np.max(a) <= 1.0
            a = np.around(a * 0xff).astype(np.uint8)
            src_space = 'L'
        src_space = 'L'
        for i in range(a.shape[Z]):
            img = Image.fromarray(a[i], mode=src_space)
            img = ImageOps.colorize(**{'image': img,
                                       'black': self.black,
                                       'white': self.white,
                                       'blackpoint': 0x00,
                                       'midpoint': 0x79,
                                       'whitepoint': 0xff,})
            img = img.convert(self.dst_space)
            a_img = np.array(img, dtype=np.uint8)
            if out is None:
                out = np.zeros((a.shape[Z], *a_img.shape), dtype=np.uint8)
            out[i] = a_img
        return out


class Contrast(Filter):
    """Adjust the contrast of the image, setting the darkest value
    to 0% and the brightest value to 100%.
    """
    @scaled
    def process(self, a: np.ndarray, *args) -> np.ndarray:
        a_min = np.min(a)
        a_max = np.max(a)
        scale = a_max - a_min
        if scale != 0:
            a = a - a_min
            a = a / scale
        return a


class Curve(Filter):
    """Apply an easing curve to the given image."""
    def __init__(self, ease: str) -> None:
        self.ease = e.registered_functions[ease]

    def process(self, a: np.ndarray) -> np.ndarray:
        return self.ease(a)


class CutLight(Filter):
    def __init__(self, threshold: float,
                 ease: str = '',
                 scale: float = 1.0) -> None:
        self.scale = float(scale)
        self.threshold = threshold
        self.ease = e.registered_functions[ease]

    # Public methods.
    def process(self, a: np.array, *args) -> np.array:
        a[a > self.threshold] = self.threshold
        a = a / self.threshold
        return self.ease(a)


class CutShadow(CutLight):
    # Public methods.
    @scaled
    def process(self, a: np.array, *args) -> np.array:
        a = 1.0 - a
        threshold = 1.0 - self.threshold
        a[a > threshold] = threshold
        a = a / threshold
        a = 1.0 - a
        return self.ease(a)


class Flip(Filter):
    def __init__(self, direction: str, *args, **kwargs) -> None:
        self.direction = direction

    # Public methods.
    def process(self, a: np.ndarray) -> np.ndarray:
        if self.direction == 'h':
            return np.flip(a, X)
        if self.direction == 'v':
            return np.flip(a, Y)
        if self.direction == 't':
            return np.flip(a, Z)
        msg = f'Direction {self.direction} not recognized.'
        raise ValueError(msg)


class GaussianBlur(Filter):
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    # Public method.
    def process(self, a: np.ndarray, *args) -> np.ndarray:
        for i in range(a.shape[0]):
            frame = np.zeros(a.shape[1:], dtype=a.dtype)
            frame = a[i]
            frame = cv2.GaussianBlur(frame, (0, 0), self.sigma, self.sigma, 0)
            a[i] = frame
        return a


class Grain(Filter):
    def __init__(self, scale: float, *args, **kwargs):
        self.scale = float(scale)
        self._grain = None
        self._noise = s.Random(.5, self.scale)
        super().__init__(*args, **kwargs)

    # Public methods.
    def process(self, a: np.ndarray) -> np.ndarray:
        """Perform the filter action on the image."""
        if not self._grain:
            size = a.shape
            grain = self._noise.fill(size)
            self._grain = grain
        return op.overlay(a, self._grain)


class Glow(Filter):
    """Use gaussian blurs to create a halo around brighter objects
    in the image.
    """
    def __init__(self, start_sigma: float) -> None:
        self.start_sigma = start_sigma

    # Public methods.
    def process(self, a: np.ndarray, *args) -> np.ndarray:
        b = a.copy()
        sigma = self.start_sigma
        while sigma > 0:
            if sigma % 2 != 1:
                sigma -= 1
            blur = GaussianBlur(sigma)
            b = blur.process(b)
            b = op.screen(a, b)
            sigma = sigma // 2
        return b


class Grow(Filter):
    """Expand the generated image and then crop back to the original
    image size.
    """
    def __init__(self, factor: float,
                 center: Union[None, Sequence[int]] = None) -> None:
        self.factor = factor
        self.center = center

    # Public methods.
    def process(self, a: np.ndarray) -> np.ndarray:
        X, Y, Z = 2, 1, 0

        resized = np.zeros_like(a)
        slices = None
        dim = list(a.shape[1:])
        dim[0] = int(dim[0] * self.factor)
        dim[1] = int(dim[1] * self.factor)
        for i in range(a.shape[Z]):
            frame = cv2.resize(a[i], tuple(dim[::-1]))
            if not slices:
                osize = a.shape
                nsize = list(osize[:])
                nsize[Y], nsize[X] = frame.shape[:2]
                start = [(n - o) // 2 for n, o in zip(nsize[1:], osize[1:])]
                end = [s + o for s, o in zip(start, osize[1:])]
                slices = tuple(slice(s, e) for s, e in zip(start, end))
            resized[i] = frame[slices]
        return resized


class Inverse(Filter):
    def __init__(self, ease: str = '') -> None:
        self.ease = e.registered_functions[ease]

    # Public methods.
    @scaled
    def process(self, a: np.ndarray, *args) -> np.ndarray:
        return self.ease(1 - a)


class LinearToPolar(Filter):
    # Filter protocol.
    def preprocess(self, size: Sequence[int], *args) -> Sequence[int]:
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

    def process(self, a: np.ndarray) -> np.ndarray:
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


class MotionBlur(Filter):
    """Apply a blur in a given direction to give the appearance of
    motion.
    """
    def __init__(self, size: Union[str, int],
                 direction: str) -> None:
        self.size = int(size)
        self.direction = direction

    # Public methods.
    def process(self, a: np.ndarray, *args) -> np.ndarray:
        """Taken from:
        https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
        """
        size = self.size
        kernel = np.zeros((size, size), float)
        if self.direction == 'h':
            y = int(size // 2)
            for x in range(size):
                kernel[y][x] = 1 / size
        if self.direction == 'v':
            x = int(size // 2)
            for y in range(size):
                kernel[y][x] = 1 / size

        for i in range(a.shape[Z]):
            blur = np.zeros(a.shape[Y:], dtype=a.dtype)
            blur = a[i]
            blur = cv2.filter2D(blur, -1, kernel)
            a[i] = blur
        return a


class Pinch(Filter):
    def __init__(self, amount: Union[float, str],
                 radius: Union[float, str],
                 scale: Union[Tuple[float], str],
                 offset: Union[Tuple[float], str] = (0, 0, 0)):
        self.amount = np.float32(amount)
        self.radius = np.float32(radius)
        if isinstance(scale, str):
            scale = scale.split(',')
        self.scale = tuple(np.float32(n) for n in scale)
        offset = c.deserialize_sequence(offset)
        self.offset = tuple(np.float32(n) for n in offset)

    # Public methods.
    def preprocess(self, size: Sequence[int],
                   original_size: Sequence[int],
                   *args) -> Sequence[int]:
        factor = 1 + self.amount
        side = self.radius * 2 * factor
        needed_size = [side for n in original_size[1:]]
        needed_size = [original_size[0], *needed_size]
        needed_size = [int(n) for n in needed_size]
        self.padding = []
        new_size = []
        for axis in range(len(size)):
            if size[axis] >= needed_size[axis]:
                new_size.append(size[axis])
                self.padding.append(0)
            else:
                new_size.append(needed_size[axis])
                pad = needed_size[axis] - size[axis]
                self.padding.append(pad)
        return new_size

    @channeled
    def process(self, a: np.ndarray, *args) -> np.ndarray:
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
        flex_x[pmask] = factor[pmask] * delta_x[pmask] / scale[X] + center[X]
        flex_y[pmask] = factor[pmask] * delta_y[pmask] / scale[Y] + center[Y]

        flex_x[~pmask] = 1.0 * delta_x[~pmask] / scale[X] + center[X]
        flex_y[~pmask] = 1.0 * delta_y[~pmask] / scale[Y] + center[Y]

        for i in range(a.shape[Z]):
            a[i] = cv2.remap(a[i], flex_x, flex_y, cv2.INTER_LINEAR)
        return a


class PolarToLinear(Filter):
    # Filter Protocol.
    def preprocess(self, size: Sequence[int], *args) -> Sequence[int]:
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

    def process(self, a: np.ndarray) -> np.ndarray:
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


class Resize(Filter):
    """Generate layers at a different size than the final image
    size. This can be used to reduce the generation time of expensive
    generators or crop out awkward edges of generated images.
    """
    def __init__(self, new_size: Sequence[int], crop: bool = False) -> None:
        if crop == 'false':
            crop = False
        self.crop = crop
        if isinstance(new_size, str):
            new_size = new_size.split(',')
            new_size = [int(n) for n in new_size[::-1]]
        self.new_size = new_size

    # Public methods.
    def preprocess(self, size: Sequence[int],
                   *args) -> Sequence[int]:
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

    def process(self, a: np.ndarray) -> np.ndarray:
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


class Ripple(Filter):
    def __init__(self, wavelength: Union[Sequence[float], str],
                 amplitude: Union[Sequence[float], str],
                 distort_axis: str = 'cross',
                 offset: Union[Sequence[float], str] = (0, 0, 0)) -> None:
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
        self.wave = c.deserialize_sequence(wavelength)
        self.amp = c.deserialize_sequence(amplitude)
        self.offset = c.deserialize_sequence(offset)
        self.distort_axis = (X, Y)
        if distort_axis == 'cross':
            self.distort_axis = (Y, X)

    # Public methods.
    def process(self, a: np.ndarray) -> np.ndarray:
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

        # Create the ripples.
        for y in range(h):
            for x in range(w):
                flex_x[y, x] = x + math.cos(x / 15) * 15
                flex_y[y, x] = y + math.cos(y / 30) * 25

        # Remap the image.
        dst = cv2.remap(a[0], flex_x, flex_y, cv2.INTER_LINEAR)
        return dst


class Rotate90(Filter):
    """Rotate the image ninety degrees in the given direction."""
    def __init__(self, direction: str) -> None:
        self.direction = direction

    # Filter protocol.
    def preprocess(self, size: Sequence[int], *args) -> Sequence[int]:
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

    def process(self, values: np.ndarray) -> np.ndarray:
        """Run the filter over the image."""
        direction = 1
        if self.direction == 'r':
            direction = -1
        return np.rot90(values, direction, (Y, X))


class Skew(Filter):
    """Skew the image."""
    def __init__(self, slope: Union[float, str]) -> None:
        self.slope = float(slope)

    # Public methods.
    def preprocess(self, size: Sequence[int],
                   original_size: Sequence[int], *args) -> Sequence[int]:
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

    def process(self, values: np.ndarray) -> np.ndarray:
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


class Twirl(Filter):
    def __init__(self, radius: Union[str, float],
                 strength: Union[str, float],
                 offset: Union[str, Sequence[int]] = (0, 0, 0)) -> None:
        self.radius = float(radius)
        self.strength = float(strength)
        self.offset = c.deserialize_sequence(offset)

    # Filter protocol.
    def preprocess(self, size: Sequence[int], *args) -> Sequence[int]:
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

    @scaled
    @channeled
    def process(self, a: np.ndarray) -> np.ndarray:
        """ Based on code taken from:
        https://stackoverflow.com/questions/30448045
        """
        # Determine the location of the center of the twirl effect.
        center = [n / 2 + o for n, o in zip(a.shape[1:3], self.offset[1:3])]

        # Run the swirl filter.
        for i in range(a.shape[Z]):
            a[i] = swirl(a[i], center[::-1], self.strength, self.radius)
        return a


# Layer filter processing functions.
def preprocess(size: Sequence[int],
               filters: Sequence[Sequence[Filter]]) -> Tuple[int]:
    if not filters:
        return size
    new_size = size[:]
    for filter in filters:
        new_size = filter.preprocess(new_size, size)
    return new_size


def process(a: np.ndarray,
            filters: Sequence[Filter]) -> np.ndarray:
    if not filters:
        return a
    for filter in filters:
        a = filter.process(a)
    return a


def postprocess(a: np.ndarray,
                filters: Sequence[Filter]) -> np.ndarray:
    # If there are no layer filters, bounce out.
    if not filters:
        return a

    # Find original size of the image.
    old_size = a.shape
    new_size = old_size[:]

    for filter in filters:
        new_size = filter.postprocess(new_size)

    # Crop the image back to the original size.
    pads = [o - n for o, n in zip(old_size, new_size)]
    starts = [pad // 2 for pad in pads]
    ends = [b + n for b, n in zip(starts, new_size)]
    slices = [slice(start, end) for start, end in zip(starts, ends)]
    return a[tuple(slices)]


# Registrations.
registered_filters = {
    'boxblur': BoxBlur,
    'color': Color,
    'contrast': Contrast,
    'curve': Curve,
    'cutshadow': CutShadow,
    'cutlight': CutLight,
    'flip': Flip,
    'gaussianblur': GaussianBlur,
    'grain': Grain,
    'glow': Glow,
    'grow': Grow,
    'inverse': Inverse,
    'lineartopolar': LinearToPolar,
    'motionblur': MotionBlur,
    'pinch': Pinch,
    'polartolinear': PolarToLinear,
    'resize': Resize,
    'ripple': Ripple,
    'rotate90': Rotate90,
    'skew': Skew,
    'twirl': Twirl,
}


# Registration and deserialization utility functions.
def make_filter(name: str, args: Sequence = ()) -> Filter:
    name = name.casefold()
    cls = registered_filters[name]
    return cls(*args)


def deserialize_filter(attrs: Mapping) -> Filter:
    cls = registered_filters[attrs['type']]
    del attrs['type']
    return cls(**attrs)


def get_regname_for_class(obj: object) -> str:
    regnames = {registered_filters[k]: k for k in registered_filters}
    clsname = obj.__class__
    return regnames[clsname]


if __name__ == '__main__':
    import doctest
    doctest.testmod()

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
    obj = GaussianBlur(5)
    size = preprocess(a.shape, [obj,])
    res = obj.process(a)
    res = postprocess(res, [obj,])
    c.print_array(res)
