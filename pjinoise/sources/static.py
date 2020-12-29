"""
static
~~~~~~

Sources for the pjinoise module that create non-random patterns.
"""
from typing import Any, Callable, List, Mapping, Sequence, Tuple, Union

import numpy as np

from pjinoise import common as c
from pjinoise.base import ValueSource, eased
from pjinoise.constants import X, Y, Z


# Pattern generators.
class Box(ValueSource):
    """Draw a box.

    :param origin: The location of the upper left corner of the box.
    :param dimensions: The size of the box in three dimensions.
    :param color: The color of the box. This is a float within the
        range 0 <= x <= 1.
    :return: A :class:Box object.
    :rtype: pjinoise.sources.Box
    """
    def __init__(self, origin: Sequence[int],
                 dimensions: Sequence[int],
                 color: float = 1.0,
                 *args, **kwargs) -> None:
        self.origin = origin
        self.dimensions = dimensions
        self.color = color
        super().__init__()

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        a = np.zeros(size)
        start = [n + o for n, o in zip(loc, self.origin)]
        end = [s + d for s, d in zip(start, self.dimensions)]
        slices = [slice(s, e) for s, e in zip(start, end)]
        a[tuple(slices)] = self.color
        return a


class Data(ValueSource):
    """Provide stored image data.

    :param data: The image data for this object.
    :return: A :class:Data object.
    :rtype: pjinoise.sources.Data
    """
    def __init__(self, data: np.ndarray) -> None:
        self.data = np.array(data)
        self._shape = self.data.shape

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        shape = self.data.shape
        if size == shape:
            return self.data
        if any(n < s + l for n, s, l in zip(shape, size, loc)):
            # Find the needed magnification factor along the axis with
            # the largest difference.
            diffs = [(s + l) / n for n, s, l in zip(shape, size, loc)]
            mag_factor = max(diffs)

            # Magnify the seeded data through trilinear interpolation.
            out = c.trilinear_interpolation(self.data, mag_factor)

            # Slice the magnified data to the fill size.
            slices = tuple(slice(0, s) for s in size)
            return out[slices]

        slices = tuple(slice(l, s + l) for s, l in zip(size, loc))
        return self.data[slices]


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
    def __init__(self, direction: str = 'h',
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
        center = [(n - 1) / 2 + o for n, o in zip(size, loc)]

        # Determine the angle from center for every point
        # in the array.
        indices = np.indices(size, dtype=float)
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
