"""
sources
~~~~~~~

Objects that generate values, both patterned and noise.


Basic Usage
===========
A ValueSource object is used to create an array of data that can be
used for various things, but the main use case is the creation of
still images and video. Creating an instance of a subclass of
ValueSource (a "source") works like instantiating any other Python
class. The specific parameters vary based on what the specifc source
will create.

Usage::

    >>> src = Lines('h', 22.2, 'ios')
    >>> src
    Lines(direction='h', length=22.2, ease='ios')


fill()
------
The fill() method is used to generate data from a source. The fill()
function always adheres to the following protocol:

    :param size: A sequence containing three integers that give the
        Z, Y, and X dimensions of the data array to produce.
    :param loc: (Optional.) A sequence containing three integers that
        offset the data generation within the total space of data the
        source could generate.

        If you think of each position in the output array as a pixel
        in an image, most sources use the location of that pixel within
        the image to determine what the value of that pixel should be.
        The loc parameter allows you to offset the location of all the
        pixels in the image by the given amount, which will change the
        values of the output.
    :return: A numpy n-dimensional array containing floats within the
        range 0 <= x <= 1.
    :rtype: numpy.ndarray

Usage::

    >>> lines = Lines('h', 22.2, 'ios')
    >>> size = (1, 2, 3)
    >>> location = (2, 3, 4)
    >>> lines.fill(size, location)
    array([[[0.45560205, 0.45560205, 0.45560205],
            [0.60298931, 0.60298931, 0.60298931]]])


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

Note: When creating subclasses of ValuesSource, care should be taken
to ensure the subclass's asdict() method only returns data that can
be serialized by the json module. There are a couple of ways to
accomplish this:

*   ValueSource.asdict() won't return private attributes in the
    output dictionary, so the serialized version can be stored in
    the public attribute (or at least returned by calls to the
    public attribute) and the deserialized version stored in a
    private attribute.
*   The asdict() method can be subclassed to serialize the value of
    the attribute before the output dictionary is returned.

Usage:

    >>> lines = Lines('h', 22.2, 'ios')
    >>> lines.asdict()
    {'direction': 'h', 'length': 22.2, 'type': 'lines', 'ease': 'ios'}


Serialization Helpers
=====================
The sources module has a few capabilities to help with serialization.
The get_regname_for_class() function returns a string that represents
a source class that has been registered with the sources module.

Usage::

    >>> obj = Lines('h', 22.2, 'ios')
    >>> get_regname_for_class(obj)
    'lines'

New source classes can be registered with the source module by adding
them to the registered_source dictionary. The key should be the
short string you want to use as the serialized value of the class.

Usage::

    >>> class Spam(ValueSource):
    ...     def fill(*args):
    ...         pass
    ...
    >>> registered_sources['spam'] = Spam
    >>> obj = Spam()
    >>> get_regname_for_class(obj)
    'spam'

The primary purpose for this feature is to allow classes to be easily
deserialized from JSON objects. It also should provide a measure of
input validation at deserialization to reduce the risk of remote code
execution vulnerabilities though deserialization.
"""
from abc import ABC, abstractmethod
from functools import wraps
import inspect
from operator import itemgetter
import random
from typing import Any, Callable, List, Mapping, Sequence, Tuple, Union

import cv2
import numpy as np
from numpy.random import default_rng

from pjinoise.constants import P, TEXT, X, Y, Z
from pjinoise import common as c
from pjinoise import ease as e
from pjinoise import operations as op


# Common decorators.
def eased(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(obj, *args, **kwargs) -> np.ndarray:
        return obj._ease(fn(obj, *args, **kwargs))
    return wrapper


# Base classes.
class ValueSource(ABC):
    """Base class to define common features of noise classes.

    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: ABCs cannot be instantiated.
    :rtype: ABCs cannot be instantiated.
    """
    def __init__(self, ease: str = 'l', *args, **kwargs) -> None:
        self.ease = ease

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

    @property
    def ease(self) -> str:
        return e.get_regname_for_func(self._ease)

    @ease.setter
    def ease(self, value: str) -> None:
        self._ease = e.registered_functions[value]

    # Public methods.
    def asargs(self) -> List[Any]:
        sig = inspect.signature(self.__init__)
        kwargs = self.asdict()
        args = [kwargs[key] for key in sig.parameters if key in kwargs]
        return args

    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = self.__dict__.copy()
        cls = self.__class__.__name__
        attrs['type'] = cls.casefold()
        attrs['ease'] = self.ease
        attrs = c.remove_private_attrs(attrs)
        return attrs

    @abstractmethod
    def fill(self, size: Sequence[int],
             location: Sequence[int] = None) -> np.ndarray:
        """Return a space filled with noise."""

    def noise(self, coords: Sequence[float]) -> int:
        """Generate the noise value for the given coordinates."""
        size = [1 for n in range(len(coords))]
        value = self.fill(size, coords)
        index = [0 for n in range(len(coords))]
        return value[index]


# Pattern generators.
class Gradient(ValueSource):
    """Generate a simple gradient.

    :param direction: (Optional.) This should be 'h' for a horizontal
        gradient or 'v' for a vertical gradient.
    :param stops: (Optional.) A gradient stop sets the color at a
        position in the gradient. This is a one-dimensional sequence
        of numbers. It's parsed in pairs, with the first number being
        the position of the stop and the second being the color value
        of the stop.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Gradient object.
    :rtype: pjinoise.sources.Gradient
    """
    def __init__(self,
                 direction: str = 'h',
                 stops: Union[Sequence[float], str] = (0, 0, 1, 1),
                 *args, **kwargs) -> None:
        self.direction = direction

        # Parse the stops for the gradient.
        if isinstance(stops, str):
            stops = stops.split(',')
        self.stops = stops
        self.stops = []
        for index in range(len(stops))[::2]:
            try:
                stop = [float(stops[index]), float(stops[index + 1])]
            except IndexError:
                raise ValueError(TEXT['gradient_error'])
            self.stops.append(stop)

        # If the stops don't start at index zero, add a stop for
        # index zero to make the color between zero and the first
        # stop match the color at the first stop.
        if self.stops[0][0] != 0:
            self.stops.insert(0, [0, self.stops[0][1]])

        # If the stops don't end at index one, add a stop for index
        # one to make the color between the last stop and one match
        # the color of the last stop.
        if self.stops[-1][0] != 1:
            self.stops.append([1, self.stops[-1][1]])

        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        # Map out the locations of the stops within the gradient.
        if self.direction == 'h':
            a_size = size[X]
        elif self.direction == 'v':
            a_size = size[Y]
        elif self.direction == 't':
            a_size = size[Z]
        a = np.indices((a_size,))[0] / (a_size - 1)
        a_rev = 1 - a.copy()
        a_index = a.copy()

        # Interpolate the color values between the stops.
        # To do this I need to know the percentage of distance each
        # pixel represents. So, I need to do this in pairs.
        left_stop = self.stops[0]
        for right_stop in self.stops[1:]:

            # Create an array mask that isolates the area between the
            # two stops.
            mask = np.zeros(a.shape, bool)
            mask[a_index >= left_stop[0]] = True
            mask[a_index > right_stop[0]] = False

            # Determine where each pixel is within the area between
            # the two stops.
            distance = right_stop[0] - left_stop[0]
            a[mask] = a_index[mask] - left_stop[0]
            a[mask] = a[mask] / distance
            a_rev[mask] = 1 - a[mask]

            # Interpolate the color of the pixel based on its distance
            # from each of those stops and the color of those stops.
            a[mask] = a[mask] * right_stop[1]
            a_rev[mask] = a_rev[mask] * left_stop[1]
            a[mask] = a[mask] + a_rev[mask]

            # The right stop for this part of the gradient is the left
            # stop for the next part of the gradient.
            left_stop = right_stop

        # Run the easing function on the values and return the result.
        if self.direction == 'h':
            a = a.reshape(1, 1, a_size)
            a = np.tile(a, (size[Z], size[Y], 1))
        elif self.direction == 'v':
            a = a.reshape(1, a_size, 1)
            a = np.tile(a, (size[Z], 1, size[X]))
        elif self.direction == 't':
            a = a.reshape(a_size, 1, 1)
            a = np.tile(a, (1, size[Y], size[X]))
        return a


class Lines(ValueSource):
    """Generate simple lines.

    :param direction: (Optional.) This should be 'h' for a horizontal
        gradient or 'v' for a vertical gradient.
    :param length: (Optional.) The distance between each line.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Lines object.
    :rtype: pjinoise.sources.Lines
    """
    def __init__(self,
                 direction: str = 'h',
                 length: Union[float, str] = 64,
                 *args, **kwargs) -> None:
        self.direction = direction
        self.length = float(length)
        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        values = np.indices(size)
        for axis in X, Y, Z:
            values[axis] += loc[axis]
        if self.direction == 'v':
            values = values[X] + values[Z]
        elif self.direction == 'h':
            values = values[Y] + values[Z]
        else:
            values = values[X] + values[Y]
        period = (self.length - 1)
        values = values % period
        values[values > period / 2] = period - values[values > period / 2]
        values = (values / (period / 2))
        return values


class Rays(ValueSource):
    """Create rayes that generate from a central point.

    :param count: The number of rays to generate.
    :param offset: (Optional.) Rotate the rays around the generation
        point. This is measured in radians.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Rays object.
    :rtype: pjinoise.sources.Rays
    """
    def __init__(self, count: Union[str, int],
                 offset: Union[str, float] = 0,
                 *args, **kwargs) -> None:
        self.count = int(count)
        self.offset = float(offset)
        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        # Determine the center of the effect.
        center = [(n - 1) / 2 + o for n, o in zip(size[Y:], loc[Y:])]

        # Determine the angle from center for every point
        # in the array.
        indices = np.indices(size[Y:], dtype=float)
        for axis in X, Y:
            indices[axis] -= center[axis]
        x, y = indices[X], indices[Y]
        angle = np.zeros_like(x)
        angle[x != 0] = np.arctan(y[x != 0] / x[x != 0])

        # Correct for inaccuracy of the arctan function when one or
        # both of the coordinates is less than zero.
        m = np.zeros_like(x)
        m[x < 0] += 1
        m[y < 0] += 3
        angle[m == 1] += np.pi
        angle[m == 4] += np.pi
        angle[m == 3] += 2 * np.pi

        # Create the rays.
        ray_angle = 2 * np.pi / self.count
        offset = (self.offset * np.pi) % (2 * np.pi)
        rays = (angle + offset) % ray_angle
        rays /= ray_angle
        rays = abs(rays - .5) * 2
        if center[X] % 1 == 0 and center[Y] % 1 == 0:
            center = [int(n) for n in center]
            rays[(center[Y], center[X])] = 1
        rays = np.tile(rays, (size[Z], 1, 1))
        return rays


class Ring(ValueSource):
    """Create a series of concentric circles.

    :param radius: The radius of the first ring, which is the ring
        closest to the center. It is measured from the origin point
        of the rings to the middle of the band of the first ring.
    :param width: The width of each band of the ring. It's measured
        from each edge of the band.
    :param gap: (Optional.) The distance between each ring. It's
        measured from the middle of the first band to the middle
        of the next band. The default value of zero causes the
        rings to draw on top of each other, making it look like
        there is only one ring.
    :param count: (Optional.) The number of rings to draw. The
        default is one.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Ring object.
    :rtype: pjinoise.sources.Ring
    """
    def __init__(self, radius: float,
                 width: float,
                 gap: float = 0,
                 count: int = 1,
                 *args, **kwargs) -> None:
        """Initialize an instance of Ring."""
        self.radius = float(radius)
        self.width = float(width)
        self.gap = float(gap)
        self.count = int(count)
        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        # Map out the volume of space that will be created.
        a = np.zeros(size)
        c = np.indices(size)
        for axis in X, Y, Z:
            c[axis] += loc[axis]

            # Calculate where every point is relative to the center
            # of the spot.
            c[axis] = abs(c[axis] - size[axis] // 2)

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        c = np.sqrt(c[X] ** 2 + c[Y] ** 2)
        for i in range(self.count):
            radius = self.radius + self.gap * i
            if radius != 0:
                working = c / np.sqrt(radius ** 2)
                working = np.abs(working - 1)
                wr = self.width / 2 / radius
                m = np.zeros(working.shape, bool)
                m[working <= wr] = True
                a[m] = working[m] * (radius / (self.width / 2))
                a[m] = 1 - a[m]
        return a


class Solid(ValueSource):
    """Fill a space with a solid color.

    :param color: The color to use for the fill. Zero is black. One
        is white. The values between are values of gray.
    :return: :class:Solid object.
    :rtype: pjinoise.sources.Solid
    """
    def __init__(self, color: Union[str, float], *args, **kwargs) -> None:
        self.color = float(color)
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        a = np.zeros(size)
        a.fill(self.color)
        return a


class Spheres(ValueSource):
    """Fill a space with a series of spots.

    :param radius: The radius of an individual spot.
    :param offset: (Optional.) Whether alternating rows or columns
        should be offset. Set to 'x' for rows to be offset. Set to
        'y' for columns to be offset. It defaults to None for no
        offset.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Spheres object.
    :rtype: pjinoise.sources.Spheres
    """
    def __init__(self, radius: float,
                 offset: str = None, *args, **kwargs) -> None:
        self.radius = float(radius)
        self.offset = offset
        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        # Map out the volume of space that will be created.
        a = np.indices(size)
        for axis in X, Y, Z:
            a[axis] += loc[axis]

        # If configured, offset every other row, column, or plane by
        # by the radius of the circle.
        if self.offset == 'x':
            mask = np.zeros(a[Y].shape, bool)
            d = self.radius * 2
            dd = d * 2
            mask[a[Y] % dd < d] = True

            # Note: This used to be just subtracting the radius from
            # a[X][~mask], but it stopped working. I'm not sure why.
            # Maybe it never did, and my headache was keeping me from
            # noticing it. Either way, this seems to work.
            a[X][mask] = a[X][mask] + self.radius
            a[Y][mask] = a[Y][mask] + self.radius
            a[Y][~mask] = a[Y][~mask] + self.radius

        if self.offset == 'y':
            mask = np.zeros(a[X].shape, bool)
            d = self.radius * 2
            dd = d * 2
            mask[a[X] % dd < d] = True

            # Note: For some reason, this is not the same as just
            # subtracting the radius from a[Y][mask]. I don't know
            # why, and my headache is making me disinclined to look
            # at the math.
            a[X][mask] = a[X][mask] + self.radius
            a[X][~mask] = a[X][~mask] + self.radius
            a[Y][~mask] = a[Y][~mask] + self.radius

        # Split the volume into unit cubes that are the size of the
        # diameter of the circle. Then adjust the indicies to measure
        # the distance to the nearest unit rather than the distance
        # from the last unit.
        a = a % (self.radius * 2)
        a[a > self.radius] = self.radius * 2 - a[a > self.radius]

        # Interpolate the unit distances through the sphere equation
        # to generate the regularly spaced spheres in the volume.
        # Then run the easing function on those spheres.
        a = np.sqrt(a[X] ** 2 + a[Y] ** 2 + a[Z] ** 2)
        a = 1 - (a / np.sqrt(3 * self.radius ** 2))
        return a


class Spot(ValueSource):
    """Fill a space with a spot.

    :param radius: The radius of the spot.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Spot object.
    :rtype: pjinoise.sources.Spot
    """
    def __init__(self, radius: float, *args, **kwargs) -> None:
        self.radius = float(radius)
        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        # Map out the volume of space that will be created.
        a = np.indices(size)
        for axis in X, Y, Z:
            a[axis] += loc[axis]

            # Calculate where every point is relative to the center
            # of the spot.
            a[axis] = abs(a[axis] - size[axis] // 2)

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        a = np.sqrt(a[X] ** 2 + a[Y] ** 2)
        a = 1 - (a / np.sqrt(2 * self.radius ** 2))
        a[a > 1] = 1
        a[a < 0] = 0
        return a


class Waves(ValueSource):
    """Generates concentric circles.

    :param length: The radius of the innermost circle.
    :param growth: (Optional.) Either the string 'linear' or the
        string 'geometric'. Determines whether the distance between
        each circle remains constant (linear) or increases
        (geometric). Defaults to linear.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :returns: :class:Waves object.
    :rtype: pjinoise.sources.Waves
    """
    def __init__(self, length: Union[str, float],
                 growth: str = 'l',
                 *args, **kwargs):
        """Initialize an instance of Waves."""
        self.length = float(length)
        self.growth = growth
        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        # Map out the volume of space that will be created.
        a = np.zeros(size)
        c = np.indices(size, dtype=float)
        center = [(n - 1) / 2 + o for n, o in zip(size, loc)]
        for axis in X, Y, Z:
            c[axis] -= center[axis]

        # Perform a spherical interpolation on the points in the
        # volume and run the easing function on the results.
        c = np.sqrt(c[X] ** 2 + c[Y] ** 2)
        if self.growth == 'l' or growth == 'linear':
            a = c % self.length
            a /= self.length
            a = abs(a - .5) * 2

        elif self.growth == 'g' or growth == 'geometric':
            in_length = 0
            out_length = self.length
            while in_length < np.max(c):
                m = np.ones(a.shape, bool)
                m[c < in_length] = False
                m[c > out_length] = False
                a[m] = c[m]
                a[m] -= in_length
                a[m] /= out_length - in_length
                a[m] = abs(a[m] - .5) * 2
                in_length = out_length
                out_length *= 2

        return a


# Random noise generators.
class SeededRandom(ValueSource):
    """Create continuous-uniformly distributed random noise with a
    seed value to allow the noise to be regenerated in a predictable
    way.

    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:SeededRandom object.
    :rtype: pjinoise.sources.SeededRandom
    """
    def __init__(self,
                 seed: Union[None, int, str, bytes] = None,
                 *args, **kwargs) -> None:
        """Initialize an instance of SeededRandom."""
        self.seed = seed
        self._seed = _text_to_int(self.seed)
        self._rng = default_rng(self._seed)
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        # Random number generation is linear and unidirectional. In
        # order to give the illusion of their being a space to move
        # in, we define the location of the first number generated
        # as the origin of the space (so: [0, 0, 0]). We then will
        # make the negative locations in the space the reflection of
        # the positive spaces.
        new_loc = [abs(n) for n in loc]

        # To simulate positioning within a space, we need to burn
        # random numbers from the generator. This would be easy if
        # we were just generating single dimensional noise. Then
        # we'd only need to burn the first numbers from the generator.
        # Instead, we need to burn until with get to the first row,
        # then accept. Then we need to burn again until we get to
        # the second row, and so on. This implementation isn't very
        # memory efficient, but it should do the trick.
        new_size = [s + l for s, l in zip(size, loc)]
        a = self._rng.random(new_size)
        slices = tuple(slice(n, None) for n in loc)
        a = a[slices]
        return a


class Random(SeededRandom):
    """Create random noise with a continuous uniform distribution.

    :param mid: (Optional.) The midpoint level of the noise. This
        is, basically, the base value of each point in the space.
        The random numbers will then be used to increase or
        decrease the value from this point.
    :param scale: (Optional.) The maximum amount the randomness
        should increase or decrease the value of a point in the
        noise.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Random object.
    :rtype: pjinoise.sources.Random
    """
    def __init__(self, mid: float = .5, scale: float = .02,
                 *args, **kwargs) -> None:
        """Initialize an instance of Random."""
        self.mid = mid
        self.scale = scale
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        a = super().fill(size, loc)
        a = a * self.scale * 2 - self.scale
        return a + self.mid


class Embers(SeededRandom):
    """Fill a space with bright points or dots that resemble embers
    or stars.

    :param depth: (Optional.) The number of different sizes of dots
        to create.
    :param threshold: (Optional.) Embers starts by generating random
        values for each point. This sets the minimum value to keep in
        the output. It's a percentage, and the lower the value the
        more points are kept.
    :param blend: (Optional.) A string reference to the operation to
        use when blending different sizes of dots together.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Embers object.
    :rtype: pjinoise.sources.Embers
    """
    def __init__(self, depth: int = 1,
                 threshold: float = .9995,
                 blend: str = 'lighter',
                 *args, **kwargs) -> None:
        self.depth = depth
        self.threshold = threshold
        self.blend = blend
        self._blend = op.registered_ops[blend]
        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        mag = 1
        out = np.zeros(size, dtype=float)
        for layer in range(self.depth):
            # Use the magnification to determine the size of the noise
            # to get.
            fill_size = [size[0], * (n // mag for n in size[1:])]
            fill_size = [int(n) for n in fill_size]

            # Get the noise to work with.
            a = super().fill(fill_size, loc)

            # Use the threshold to turn it into a sparse collection
            # of points. Then scale to increase the apparent difference
            # in brightness.
            a = a - self.threshold
            a[a < 0] = 0
            a[a > 0] = a[a > 0] * .25
            a[a > 0] = a[a > 0] + .75

            # Resize to increase the size of the points.
            resized = np.zeros(size, dtype=a.dtype)
            for i in range(resized.shape[Z]):
                frame = np.zeros(a.shape[Y:3], dtype=a.dtype)
                frame = a[i]
                resized[i] = cv2.resize(frame, (size[X], size[Y]))

            # Blend the layer with previous layers.
            out = self._blend(out, resized)

            mag = mag * 1.5

        return out


# Random noise using unit cubes.
class UnitNoise(ValueSource):
    """An base class for visual noise generated with a unit grid.

    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where colors for
        the gradient are set.
    :param table: (Optional.) The colors to set for the vertices.
        They will repeat if there are more units along the axis
        in an image then there are colors defined for that axis.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: This ABC cannot be instantiated.
    :rtype: The ABC cannot be instantiated.
    """
    hashes = [f'{n:>03b}'[::-1] for n in range(2 ** 3)]

    def __init__(self,
                 unit: Union[Sequence[int], str],
                 table: Union[Sequence[float], str, None] = None,
                 seed: Union[str, int, None] = None,
                 *args, **kwargs) -> None:
        """Initialize an instance of UnitNoise."""
        self._scale = 0xff

        if isinstance(unit, str):
            unit = self._norm_coordinates(unit)
        self.unit = unit

        self.seed = seed
        self._seed = _text_to_int(self.seed)

        if table == 'P':
            table = P
        if table is None:
            table = self._make_table()
        self.table = np.array(table)
        self._shape = self.table.shape
        self.table = np.ravel(self.table)
        super().__init__(*args, **kwargs)

    # Public methods.
    def asdict(self) -> dict:
        attrs = super().asdict()
        if attrs['seed']:
            del attrs['table']
        if 'table' in attrs:
            attrs['table'] = self.table.tolist()
            if attrs['table'] == P:
                attrs['table'] = 'P'
        return attrs

    # Private methods.
    def _build_vertices_table(self,
                              size: Sequence[int],
                              whole: np.ndarray,
                              axes: Sequence[int]) -> dict:
        """Build a hash table of the color values for the vertices of
        the unit cubes/squares within the noise volume.

        Unit noise splits the noise volume into distinct units, then
        sets the color level at the vertices of the cubes or squares
        defined by those units. This method builds a hash table of
        the arrays that represent each of those vertices, filling the
        space within the unit with the value of the vertex. This
        allows you to then interpolate between the values of the
        vertices to determine the color value at each pixel in the
        noise volume.

        :param whole: An array that maps out the number of whole units
            to the left, above, or beneath each pixel, depending on
            the axis in question.
        :param axes: The dimensional axes of the unit cubes/squares
            within the noise.
        :return: A dict containing the arrays representing the values
            for the vertices. The keys are named for the vertex it
            represents. Each axis is represented by a character. Zero
            means the axis is after the pixel, and one means the axis
            is before the pixel.
        :rtype: dict
        """
        hash_table = {}
        for hash in self.hashes:
            hash_whole = whole.copy()
            a_hash = np.zeros(size)
            for axis in axes:
                if hash[axis] == '1':
                    hash_whole[axis] += 1
                a_hash = (a_hash + hash_whole[axis]).astype(int)
                a_hash = np.take(self.table, a_hash)
            hash_table[hash] = a_hash
        return hash_table

    def _measure_units(self, indices: np.ndarray,
                       axes: Sequence[int]) -> Tuple[np.ndarray]:
        """Split the noise volume into unit cubes/squares.

        :param indices: An array that indexes each pixel in the
            noise volume.
        :param axes: The dimensional axes of the unit cube/square.
        :return: This returns a tuple of arrays. The first array
            (whole) represents the number of units before the pixel
            on the axis. The second array (parts) is how far into
            the unit each pixel is as a percentage.
        :rtype: tuple
        """
        whole = np.zeros(indices.shape, int)
        parts = np.zeros(indices.shape, float)
        for axis in axes:
            indices[axis] = indices[axis] / self.unit[axis]
            whole[axis] = indices[axis] // 1
            parts[axis] = indices[axis] - whole[axis]
        return whole, parts

    def _make_table(self) -> List:
        table = [n for n in range(self._scale)]
        table.extend(table)
        rng = default_rng(self._seed)
        rng.shuffle(table)
        return table

    def _norm_coordinates(self, s: Union[Sequence[int], str]) -> Sequence[int]:
        if isinstance(s, str):
            result = s.split(',')
            result = [int(n) for n in result[::-1]]
            return result
        return s

    def _lerp(self, a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Performs a linear interpolation."""
        return a * (1 - x) + b * x


class Curtains(UnitNoise):
    """A class to generate vertical bands of unit noise. Reference
    algorithms taken from:

        https://www.scratchapixel.com/lessons/procedural-generation-
        virtual-worlds/procedural-patterns-noise-part-1/creating-simple
        -1D-noise

    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where colors for
        the gradient are set.
    :param table: (Optional.) The colors to set for the vertices.
        They will repeat if there are more units along the axis
        in an image then there are colors defined for that axis.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Curtains object.
    :rtype: pjinoise.sources.Curtains
    """
    hashes = [f'{n:>02b}'[::-1] for n in range(2 ** 2)]

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = None) -> np.ndarray:
        """Return a space filled with noise."""
        # Start by creating the two-dimensional matrix with the X
        # and Z axis in order to save time by not running repetitive
        # actions along the Y axis. Each position within the matrix
        # represents a pixel in the final image.
        while len(size) < 3:
            size = list(size)
            size.insert(0, 1)
        xz_size = (size[Z], size[X])
        indices = np.indices(xz_size).astype(float)

        # Since we are changing the indices of the axes, everything
        # based on the two-dimensional XZ axes will lowercase the
        # axis names. Everything based on the three-dimensional XYZ
        # axes will continue to capitalize the axis names.
        x, z = -1, -2

        # Adjust the value of the indices by loc. This will offset the
        # generated noise within the space of potential noise.
        if loc is None:
            loc = []
        while len(loc) < 3:
            loc.append(0)
        indices[x] = indices[x] + loc[X]
        indices[z] = indices[z] + loc[Z]

        # Translate the pixel measurements of the indices into unit
        # measurements based on the unit size of noise given to the
        # noise object.
        whole, parts = self._measure_units(indices, (x, z))
        del indices

        # Curtains is, essentially, two dimensional, so the units here
        # only have four vertices around each point. Here we calculate
        # those.
        hash_table = self._build_vertices_table(xz_size, whole, (x, z))

        # And now begins the interpolation.
        x1 = self._lerp(hash_table['00'], hash_table['01'], parts[x])
        x2 = self._lerp(hash_table['10'], hash_table['11'], parts[x])
        result = self._lerp(x1, x2, parts[z])
        result = result / self._scale
        result = np.tile(result[:, np.newaxis, ...], (1, size[Y], 1))
        return result


class CosineCurtains(Curtains):
    """A class to generate vertical bands of unit noise eased with
    a cosine-based function. Reference algorithms taken from:

        https://www.scratchapixel.com/lessons/procedural-generation-
        virtual-worlds/procedural-patterns-noise-part-1/creating-simple
        -1D-noise

    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where colors for
        the gradient are set.
    :param table: (Optional.) The colors to set for the vertices.
        They will repeat if there are more units along the axis
        in an image then there are colors defined for that axis.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:CosineCurtains object.
    :rtype: pjinoise.sources.CosineCurtains
    """

    # Private methods.
    def _lerp(self, a: float, b: float, x: float) -> float:
        """Eased linear interpolation function to smooth the noise."""
        x = (1 - np.cos(x * np.pi)) / 2
        return super()._lerp(a, b, x)


class Path(UnitNoise):
    """Create a maze-like path through a grid.

    :param width: (Optional.) The width of the path. This is the
        percentage of the width of the X axis length of the size
        of the fill. Values over one will probably be weird, but
        not in a great way.
    :param inset: (Optional.) Sets how many units from the end of
        the image to draw the path. Units here refers to the unit
        parameter from the UnitNoise parent class.
    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where colors for
        the gradient are set.
    :param table: (Optional.) The colors to set for the vertices.
        They will repeat if there are more units along the axis
        in an image then there are colors defined for that axis.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Path object.
    :rtype: pjinoise.sources.Path
    """
    def __init__(self, width: float = .2,
                 inset: Sequence[int] = (0, 1, 1),
                 *args, **kwargs) -> None:
        """Initialize an instance of Path."""
        self.width = width
        self.inset = inset
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Fill a space with image data."""
        # Create a grid of values. This uses the same technique Perlin
        # noise uses to add randomness to the noise. A table of values
        # was shuffled, and we use the coordinate of each vertex within
        # the grid as part of the process to lookup the table value
        # for that vertex. This grid will be used to determine the
        # route the path follows through the space.
        unit_dim = [int(s / u) for s, u in zip(size, self.unit)]
        unit_dim = tuple(np.array(unit_dim) + np.array((0, 1, 1)))
        unit_dim = tuple(np.array(unit_dim) - np.array(self.inset) * 2)
        unit_indices = np.indices(unit_dim)
        values = np.take(self.table, unit_indices[X])
        values += unit_indices[Y]
        values = np.take(self.table, values)
        values += unit_indices[Z]
        values = np.take(self.table, values)
        unit_dim = np.array(unit_dim)

        # The cursor will be used to determine our current position
        # on the grid as we create the path.
        cursor = (0, 0, 0)

        # This will be used to track the grid vertices we've already
        # been to as we create the path. It allows us to keep the
        # path from looping back into itself.
        been_there = np.zeros(unit_dim, bool)
        been_there[tuple(cursor)] = True

        # These are the positions of the vertices the cursor could
        # move to next as it creates the path.
        vertices = np.array([
            (0, 0, -1),
            (0, 0, 1),
            (0, -1, 0),
            (0, 1, 0),
        ])

        # The index tracks where we are along the path. This is used
        # to allow us to go back up the path an create a new branch
        # if we run into a dead end while creating the path. It also
        # is how we know we're done when creating the path.
        index = 0

        # Create the path.
        path = []
        while True:

            # Look at the options available for the direction the path
            # can take. Some of them won't be viable because they are
            # outside the bounds of the image or have already been
            # hit.
            cursor = np.array(cursor)
            options = [vertex + cursor for vertex in vertices]
            viable = [(o, values[tuple(o)]) for o in options
                      if self._is_viable_option(o, unit_dim, been_there)]

            # If there is a viable next step, take that step.
            if viable:
                cursor = tuple(cursor)
                viable = sorted(viable, key=itemgetter(1))
                newloc = tuple(viable[0][0])
                path.append((cursor, newloc))
                been_there[newloc] = True
                cursor = newloc
                index = len(path)

            # If there is not a viable next step, go back to the last
            # place you were, so to see if there are any viable steps
            # there. If this goes all the way back to the beginning
            # of the path and there are no viable paths, then the
            # path is complete.
            else:
                index -= 1
                if index < 0:
                    break
                cursor = path[index][0]

        # Fill the requested space with the path.
        a = np.zeros(size, dtype=float)
        width = int(self.unit[-1] * self.width)
        for step in path:
            start = self._unit_to_pixel(step[0])
            end = self._unit_to_pixel(step[1])
            slice_y = self._get_slice(start[Y], end[Y], width)
            slice_x = self._get_slice(start[X], end[X], width)
            a[:, slice_y, slice_x] = 1.0
        return a

    # Private methods.
    def _get_slice(self, start, end, width):
        if start > end:
            start, end = end, start
        start -= width
        end += width
        return slice(start, end)

    def _is_viable_option(self, option, unit_dim, been_there):
        loc = tuple(option)
        if (np.min(option) >= 0
                and all(unit_dim > option)
                and not been_there[loc]):
            return True
        return False

    def _unit_to_pixel(self, unit_loc: Sequence[int]) -> Sequence[int]:
        """Convert an index of the unit grid array into an index
        of the image data.
        """
        unit = np.array(self.unit)
        pixel_loc = np.array(unit_loc) * unit
        pixel_loc += np.array(self.inset) * unit
        return tuple(pixel_loc)


class Perlin(UnitNoise):
    """A class to generate Perlin noise.

    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where colors for
        the gradient are set.
    :param table: (Optional.) The colors to set for the vertices.
        They will repeat if there are more units along the axis
        in an image then there are colors defined for that axis.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Perlin object.
    :rtype: pjinoise.sources.Perlin
    """
    # Public classes.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = None) -> np.array:
        """Return a space filled with Perlin noise."""
        # Perlin noise requires a three-dimensional space.
        while len(size) < 3:
            size = list(size[:])
            size.insert(0, 1)
        while loc and len(loc) < 3:
            loc = list(loc[:])
            loc.append(0)

        # Map out the locations of the pixels that need to be
        # generated within the space. If a starting location was
        # given, adjust the positions of the pixels to be relative
        # to that starting position.
        indices = np.indices(size)
        if loc:
            indices[X] = indices[X] + loc[X]
            indices[Y] = indices[Y] + loc[Y]
            indices[Z] = indices[Z] + loc[Z]

        # The Perlin noise algorithm sets the value of noise at the
        # unit vertices. The size in pixels of these units was set
        # when this noise object was initialized. Here we translate
        # the pixel positions into unit measurements.
        unit_size = np.array([
            np.full(indices[Z].shape, self.unit[Z]),
            np.full(indices[Y].shape, self.unit[Y]),
            np.full(indices[X].shape, self.unit[X]),
        ])
        units = indices / np.array(unit_size)
        units = units % 255
        del indices
        del unit_size

        # The noise value at a pixel, then, is determined through
        # interpolating the noise value at the nearest unit vertices
        # for each pixel. In order to do that, we're going to need
        # the unit each pixel is in (whole) and how far into the unit
        # each pixel is (part). In order to smooth out the noise, we
        # also need a value from an easing function (fades).
        whole = (units // 1).astype(int)
        parts = units - whole
        fades = 6 * parts ** 5 - 15 * parts ** 4 + 10 * parts ** 3
        del units

        # A unit is a rectangle. That means there are eight unit
        # vertices that surround each pixel. Those vertices are
        # named with a binary mask representing whether the vertex
        # is before or after the pixel on a particular axis. So
        # the vertex "011" is ahead of the pixel on the Y and X
        # axis, but behind the pixel on the Z axis. The hash_table
        # then contains the noise value for the named vertex.
        hash_table = {}
        for hash in self.hashes:
            hash_whole = whole.copy()
            if hash[2] == '1':
                hash_whole[X] += 1
            if hash[1] == '1':
                hash_whole[Y] += 1
            if hash[0] == '1':
                hash_whole[Z] += 1
            result = hash_whole[Z]
            for axis in (Y, X):
                result = np.take(self.table, result) + hash_whole[axis]
            hash_table[hash] = result
        del whole
        del hash_whole

        # To be honest, I don't fully understand what this part of
        # the Perlin noise algorithm is doing. It's called the
        # gradient, so it must have something to do with how the
        # level of noise changes between unit vertices. Beyond that
        # I'm not sure.
        def _grad(loc_mask, hash, x, y, z):
            z = z.copy()
            y = y.copy()
            x = x.copy()
            if loc_mask[0] == '1':
                z -= 1
            if loc_mask[1] == '1':
                y -= 1
            if loc_mask[2] == '1':
                x -= 1

            m = hash & 0xf
            out = np.zeros_like(x)
            out[m == 0x0] = x[m == 0x0] + y[m == 0x0]
            out[m == 0x1] = -x[m == 0x1] + y[m == 0x1]
            out[m == 0x2] = x[m == 0x2] - y[m == 0x2]
            out[m == 0x3] = -x[m == 0x3] - y[m == 0x3]
            out[m == 0x4] = x[m == 0x4] + z[m == 0x4]
            out[m == 0x5] = -x[m == 0x5] + z[m == 0x5]
            out[m == 0x6] = x[m == 0x6] - z[m == 0x6]
            out[m == 0x7] = -x[m == 0x7] - z[m == 0x7]
            out[m == 0x8] = y[m == 0x8] + z[m == 0x8]
            out[m == 0x9] = -y[m == 0x9] + z[m == 0x9]
            out[m == 0xa] = y[m == 0xa] - z[m == 0xa]
            out[m == 0xb] = -y[m == 0xb] - z[m == 0xb]
            out[m == 0xc] = y[m == 0xc] + x[m == 0xc]
            out[m == 0xd] = -y[m == 0xd] + z[m == 0xd]
            out[m == 0xe] = y[m == 0xe] - x[m == 0xe]
            out[m == 0xf] = -y[m == 0xf] - z[m == 0xf]
            return out

        # Perform the linear interpolation of the results of the
        # gradient function for each of the surrounding vertices to
        # determine the level of noise at each pixel. This is done
        # by axis, interpolating the X values to get the Y values,
        # and interpolating those to get the Z value.
        grad = _grad
        x1a = grad('000', hash_table['000'], parts[X], parts[Y], parts[Z])
        x1b = grad('001', hash_table['001'], parts[X], parts[Y], parts[Z])
        x1 = self._lerp(x1a, x1b, fades[X])
        del x1a, x1b

        x2a = grad('010', hash_table['010'], parts[X], parts[Y], parts[Z])
        x2b = grad('011', hash_table['011'], parts[X], parts[Y], parts[Z])
        x2 = self._lerp(x2a, x2b, fades[X])
        del x2a, x2b

        x3a = grad('100', hash_table['100'], parts[X], parts[Y], parts[Z])
        x3b = grad('101', hash_table['101'], parts[X], parts[Y], parts[Z])
        x3 = self._lerp(x3a, x3b, fades[X])
        del x3a, x3b

        x4a = grad('110', hash_table['110'], parts[X], parts[Y], parts[Z])
        x4b = grad('111', hash_table['111'], parts[X], parts[Y], parts[Z])
        x4 = self._lerp(x4a, x4b, fades[X])
        del x4a, x4b

        y1 = self._lerp(x1, x2, fades[Y])
        y2 = self._lerp(x3, x4, fades[Y])
        del x1, x2, x3, x4
        values = (self._lerp(y1, y2, fades[Z]) + 1) / 2

        # The result from the Perlin noise function is a percentage
        # of how much of the maximum noise each pixel contains.
        # Run the noise through the easing function and return.
        return values


class Values(UnitNoise):
    """Produce a gradient over a multidimensional space.

    :param size: (Optional.) The expected size of the noise that
        will be generated. This is only used if no table is passed.
    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where colors for
        the gradient are set.
    :param table: (Optional.) The colors to set for the vertices.
        They will repeat if there are more units along the axis
        in an image then there are colors defined for that axis.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Values object.
    :rtype: pjinoise.sources.Values
    """
    def __init__(self,
                 size: Union[Sequence[int], None] = (50, 720, 1280),
                 *args, **kwargs) -> None:
        """Initialize an instance of GradientNoise."""
        self.size = size
        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             location: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        # Map out the space.
        a = np.indices(size, float)
        for axis in X, Y, Z:
            a[axis] += location[axis]

            # Split the space up into units.
            a[axis] = a[axis] / self.unit[axis]
            a[axis] %= 255

        # The unit distances are split. The unit values are needed
        # to set the color value of each vertex within the volume.
        # The parts value is needed to interpolate the noise value
        # at each pixel.
        whole = (a // 1).astype(int)
        parts = a - whole

        # Get the color for the eight vertices that surround each of
        # the pixels.
        hash_table = {}
        for hash in self.hashes:
            hash_whole = whole.copy()
            a_hash = np.zeros(a.shape)
            if hash[2] == '1':
                hash_whole[X] += 1
            if hash[1] == '1':
                hash_whole[Y] += 1
            if hash[0] == '1':
                hash_whole[Z] += 1
            a_hash = (hash_whole[Z] * self._shape[Y] * self._shape[X]
                      + hash_whole[Y] * self._shape[X]
                      + hash_whole[X])
            a_hash = np.take(self.table, a_hash)
            hash_table[hash] = a_hash
        else:
            del a_hash, hash_whole, whole

        # And now we interpolate.
        x1 = self._lerp(hash_table['000'], hash_table['001'], parts[X])
        x2 = self._lerp(hash_table['010'], hash_table['011'], parts[X])
        x3 = self._lerp(hash_table['100'], hash_table['101'], parts[X])
        x4 = self._lerp(hash_table['110'], hash_table['111'], parts[X])

        y1 = self._lerp(x1, x2, parts[Y])
        y2 = self._lerp(x3, x4, parts[Y])
        del x1, x2, x3, x4

        z = self._lerp(y1, y2, parts[Z])
        z = z / self._scale
        del y1, y2

        # Apply the easing function and return.
        return z

    # Private methods.
    def _make_table(self, size: Sequence[int] = None) -> List:
        """Create a color table for vertices."""
        def fill_table(dim: Sequence[int]) -> List:
            """Recursively fill a table of the given dimensions."""
            if len(dim) > 1:
                result = [fill_table(dim[1:]) for _ in range(dim[0])]
            else:
                result = [random.random() for _ in range(dim[0])]
            return result

        # Determine the dimensions of the table.
        # The "upside down" floor division here is the equivalent to
        # ceiling division without the messy float conversion of the
        # math.ceil() function.
        dim = [-(-n // u) + 1 for n, u in zip(self.size, self.unit)]

        # Create the table.
        table = fill_table(dim)
        return table


# Random octave noise using unit cubes.
class OctaveMixin():
    """A mixin for the generation of octave noise.

    :param octave: (Optional.) Sets the number of octaves to generate.
        Essentially, this is how many different scales of noise to
        include in the output.
    :param pesistence: (Optional.) Sets the impact that each octave
        has on the final output.
    :param amplitude: (Optional.) Sets the amount the persistence
        changes for each octave.
    :param frequency: (Optional.) Sets how much the scale changes
        for each octave.
    :return: Mixins aren't intended to be instantiated.
    :rtype: Mixins aren't intended to be instantiated.
    """
    genclass = None

    def __init__(self, octaves: int = 4,
                 persistence: float = 8,
                 amplitude: float = 8,
                 frequency: float = 2,
                 *args, **kwargs) -> None:
        self.octaves = int(octaves)
        self.persistence = float(persistence)
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        super().__init__(*args, **kwargs)

    # Public methods.
    def asargs(self) -> List[Any]:
        sig = inspect.signature(self.__init__)
        sigkeys = [k for k in sig.parameters]
        gcsig = inspect.signature(self.genclass.__init__)
        genclasskeys = [k for k in gcsig.parameters]
        keys = [*sigkeys, *genclasskeys]
        kwargs = self.asdict()
        args = [kwargs[key] for key in keys if key in kwargs]
        return args

    def fill(self, size: Sequence[int],
             loc: Sequence[int] = None) -> np.ndarray:
        total = 0
        max_value = 0
        for i in range(self.octaves):
            amp = self.amplitude + (self.persistence * i)
            freq = self.frequency * 2 ** i
            kwargs = self.asdict()
            kwargs['unit'] = [n / freq for n in self.unit]
            delkeys = [
                'octaves',
                'persistence',
                'amplitude',
                'frequency',
                'shape',
                'type'
            ]
            for key in delkeys:
                if key in kwargs:
                    del kwargs[key]
            octave = self.genclass(**kwargs)
            total += octave.fill(size, loc) * amp
            max_value += amp
        a = total / max_value
        return a


class OldOctaveCosineCurtains(OctaveMixin, CosineCurtains):
    genclass = CosineCurtains

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = None) -> np.ndarray:
        total = 0
        max_value = 0
        for i in range(self.octaves):
            amp = self.amplitude + (self.persistence * i)
            freq = self.frequency * 2 ** i
            kwargs = self.asdict()
            kwargs['unit'] = [n * freq for n in self.unit]
            delkeys = [
                'octaves',
                'persistence',
                'amplitude',
                'frequency',
                'shape',
                'type'
            ]
            for key in delkeys:
                if key in kwargs:
                    del kwargs[key]
            octave = self.genclass(**kwargs)
            total += octave.fill(size, loc) * amp
            max_value += amp
        a = total / max_value
        return a


class OctaveCosineCurtains(OctaveMixin, CosineCurtains):
    """Generate octaves of CosineCurtains noise.

    :param octave: (Optional.) Sets the number of octaves to generate.
        Essentially, this is how many different scales of noise to
        include in the output.
    :param pesistence: (Optional.) Sets the impact that each octave
        has on the final output.
    :param amplitude: (Optional.) Sets the amount the persistence
        changes for each octave.
    :param frequency: (Optional.) Sets how much the scale changes
        for each octave.
    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where colors for
        the gradient are set.
    :param table: (Optional.) The colors to set for the vertices.
        They will repeat if there are more units along the axis
        in an image then there are colors defined for that axis.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:OctaveCosineCurtains object.
    :rtype: pjinoise.sources.OctaveCosineCurtains
    """
    genclass = CosineCurtains


class OctavePerlin(OctaveMixin, Perlin):
    """Create octave Perlin noise.

    :param octave: (Optional.) Sets the number of octaves to generate.
        Essentially, this is how many different scales of noise to
        include in the output.
    :param pesistence: (Optional.) Sets the impact that each octave
        has on the final output.
    :param amplitude: (Optional.) Sets the amount the persistence
        changes for each octave.
    :param frequency: (Optional.) Sets how much the scale changes
        for each octave.
    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where colors for
        the gradient are set.
    :param table: (Optional.) The colors to set for the vertices.
        They will repeat if there are more units along the axis
        in an image then there are colors defined for that axis.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:OctavePerlin object.
    :rtype: pjinoise.sources.OctavePerlin

    Note: Arguments that provide good results:
    octaves: 6
    persistence: -4
    amplitude: 24
    frequency: 4
    unit: (1024, 1024, 1024)
    """
    genclass = Perlin


# Sources that cache fill data.
class CachingMixin():
    """Cache fill results."""
    # This sets up a cache in the CachingMixin class, which, if used
    # would cause all instances with the same key to retur the same
    # cached value regardless of subtype. To avoid this, classes
    # using the CachingMixin should override this with their own
    # dictionary.
    _cache = {}

    def __init__(self, key: str = '_default', *args, **kwargs) -> None:
        self.key = key
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill(self, size: Sequence[int],
             location: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """On first call, generate and return image data, caching
        the data. On subsequent calls, return the cached data
        rather than generating the same data again.
        """
        if self.key not in self._cache:
            self._cache[self.key] = super().fill(size, location)
        return self._cache[self.key].copy()


class CachingOctavePerlin(CachingMixin, OctavePerlin):
    """A caching source for octave Perlin noise.

    :param octave: (Optional.) Sets the number of octaves to generate.
        Essentially, this is how many different scales of noise to
        include in the output.
    :param pesistence: (Optional.) Sets the impact that each octave
        has on the final output.
    :param amplitude: (Optional.) Sets the amount the persistence
        changes for each octave.
    :param frequency: (Optional.) Sets how much the scale changes
        for each octave.
    :param unit: The number of pixels between vertices along an
        axis. The vertices are the locations where colors for
        the gradient are set.
    :param table: (Optional.) The colors to set for the vertices.
        They will repeat if there are more units along the axis
        in an image then there are colors defined for that axis.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:CachingOctavePerlin object.
    :rtype: pjinoise.sources.CachingOctavePerlin
    """
    _cache = {}


# Compound sources.
class TilePaths(ValueSource):
    """Tile the output of a series of Path objects.

    :param tile_size: The size of the fills from the sources.Path
        used in the image.
    :param seeds: An iterator of seed values to use for the
        sources.Path objects used in the image.
    :param unit: The unit size for the sources.Path objects used
        in the image.
    :param line_width: The width of the line used by the sources.Path
        used in the image.
    :param inset: The inset value to use in the sources.Path objects
        used in the image.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:TilePaths object.
    :rtype: pjinoise.sources.TilePaths
    """
    def __init__(self, tile_size: Sequence[float],
                 seeds: Sequence[Any],
                 unit: Sequence[int],
                 line_width: float = .2,
                 inset: Sequence[int] = (0, 1, 1),
                 *args, **kwargs) -> None:
        """Initialize an instance of TilePaths.

        :param tile_size: The output of a Path object is a tile.
            This sets the size of those tiles.
        :param seeds: The seeds to give the individual Path objects
            when creating the tiles. It needs to be a sequence of
            data that can be used to seed a Path object.
        :param unit: The unit size for the Path objects used to
            create the tiles.
        :param line_width: (Optional.) The width of the line used in
            the tiles. This is the value of the 'width' parameter
            given to the Path objects.
        :param inset: (Optional.) The distance, in units, of empty
            space around the edge of each tile. This is the value of
            the 'inset' parameter given to the Path objects.
        :return: None.
        :rtype: NoneType
        """
        self.tile_size = tile_size
        self.seeds = seeds
        self.unit = unit
        self.line_width = line_width
        self.inset = inset
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        a = np.zeros(size, dtype=float)
        cursor = [0, 0, 0]
        seeds = self.seeds[:]
        while cursor[Y] < size[Y]:
            while cursor[X] < size[X]:
                if seeds:
                    seed = seeds.pop(0)
                else:
                    seed = None
                kwargs = {
                    'width': self.line_width,
                    'inset': self.inset,
                    'unit': self.unit,
                    'seed': seed,
                }
                source = Path(**kwargs)
                slices = tuple(self._get_slices(cursor))
                a[slices] = source.fill(self.tile_size)
                cursor[X] += self.tile_size[X]
            cursor[X] = 0
            cursor[Y] += self.tile_size[Y]
        return a

    # Private methods.
    def _get_slices(self, cursor: Sequence[int]) -> Sequence[slice]:
        return [
            slice(0, None),
            slice(cursor[Y], cursor[Y] + self.tile_size[Y]),
            slice(cursor[X], cursor[X] + self.tile_size[X]),
        ]


# Registration.
registered_sources = {
    'gradient': Gradient,
    'lines': Lines,
    'rays': Rays,
    'ring': Ring,
    'solid': Solid,
    'spheres': Spheres,
    'spot': Spot,
    'waves': Waves,

    'curtains': Curtains,
    'cosinecurtains': CosineCurtains,
    'embers': Embers,
    'perlin': Perlin,
    'random': Random,
    'seededrandom': SeededRandom,
    'values': Values,

    'oldoctavecosinecurtains': OldOctaveCosineCurtains,
    'octavecosinecurtains': OctaveCosineCurtains,
    'octaveperlin': OctavePerlin,

    'cachingoctaveperlin': CachingOctavePerlin,

    'tilepaths': TilePaths,
}


# Registration utility functions.
def deserialize_source(attrs: Mapping) -> ValueSource:
    cls = registered_sources[attrs['type']]
    del attrs['type']
    try:
        return cls(**attrs)
    except TypeError as e:
        msg = f'{cls.__name__} raised following: {e}.'
        raise TypeError(msg)


def get_regname_for_class(obj: object) -> str:
    regnames = {registered_sources[k]: k for k in registered_sources}
    clsname = obj.__class__
    return regnames[clsname]


# General utility functions.
def _text_to_int(text: Union[bytes, str, int, None]) -> int:
    if isinstance(text, (int)) or text is None:
        return text
    if isinstance(text, str):
        text = bytes(text, 'utf_8')
    return int.from_bytes(text, 'little')


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    kwargs = {
        'width': .34,
        'inset': (0, 1, 1),
        'unit': (1, 3, 3),
        'seed': 'spam',
        'ease': 'l',
    }
    cls = Path
    size = (2, 10, 10)
    obj = cls(**kwargs)
    val = obj.fill(size)
    c.print_array(val)
