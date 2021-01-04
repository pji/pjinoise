"""
random
~~~~~~

Sources for the pjinoise module that involve random generation.
"""
from operator import itemgetter
from typing import Any, List, Sequence, Tuple, Union

import cv2
import numpy as np
from numpy.random import default_rng

from pjinoise import common as c
from pjinoise import operations as op
from pjinoise.constants import X, Y, Z, P
from .caching import CachingMixin
from .source import Source, eased


# Random noise generators.
class SeededRandom(Source):
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
        self._seed = c.text_to_int(self.seed)
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
        new_size = [s + l for s, l in zip(size, new_loc)]
        a = self._rng.random(new_size)
        slices = tuple(slice(n, None) for n in new_loc)
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


class Worley(SeededRandom):
    """Fill a space with Worley noise.

    Worley noise is a type of cellular noise. The color value of each
    pixel within the space is determined by the distance from the pixel
    to the nearest of a set of randomly located points within the
    image. This creates structures within the noise that look like
    cells or pits.

    This implementation is heavily optimized from code found here:
    https://code.activestate.com/recipes/578459-worley-noise-generator/

    :param points: The number of cells in the image. A cell is a
        randomly placed point and the range of pixels that are
        closer to it than any other point.
    :param volume: (Optional.) The size of the volume that the points
        will be placed in. The default is for them to be evenly spread
        through the space generated during the fill.
    :param origin: (Optional.) The location of the upper-top-left
        corner of the volume that contains the points. This defaults
        to the upper-top-left corner of the space generated during the
        fill.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:Worley object.
    :rtype: pjinoise.sources.Worley
    """
    def __init__(self, points: int,
                 volume: Sequence[int] = None,
                 origin: Sequence[int] = (0, 0, 0),
                 *args, **kwargs) -> None:
        self.points = int(points)
        self.volume = volume
        self.origin = origin
        super().__init__(*args, **kwargs)

    # Public methods.
    @eased
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = None) -> np.ndarray:
        """Return a space filled with noise."""
        m = 0
        a = np.zeros(size, dtype=float)
        volume = self.volume
        if volume is None:
            volume = size
        volume = np.array(volume)

        # Place the seeds in the overall volume of noise.
        seeds = self._rng.random((self.points, 3))
        seeds = np.round(seeds * (volume - 1))
        seeds += np.array(self.origin)

        # Map the distances to the points.
        indices = np.indices(size)
        max_dist = np.sqrt(sum(n ** 2 for n in size))
        dist = np.zeros(size, dtype=float)
        dist.fill(max_dist)
        for i in range(self.points):
            point = seeds[i]
            work = self._hypot(point, indices)
            dist[work < dist] = work[work < dist]

        act_max_dist = np.max(dist)
        a = dist / act_max_dist
        return a

    # Private methods.
    def _hypot(self, point: Sequence[int], indices: np.ndarray) -> np.ndarray:
        axis_dist = [p - i for p, i in zip(point, indices)]
        return np.sqrt(sum(d ** 2 for d in axis_dist))


class WorleyCell(Worley):
    """Fill a space with Worley noise that fills each cell with a
    solid color.

    :param antialias: (Optional.) Soften the edges of the cell
        boundaries. Defaults to true.
    :param points: The number of cells in the image. A cell is a
        randomly placed point and the range of pixels that are
        closer to it than any other point.
    :param volume: (Optional.) The size of the volume that the points
        will be placed in. The default is for them to be evenly spread
        through the space generated during the fill.
    :param origin: (Optional.) The location of the upper-top-left
        corner of the volume that contains the points. This defaults
        to the upper-top-left corner of the space generated during the
        fill.
    :param seed: (Optional.) An int, bytes, or string used to seed
        therandom number generator used to generate the image data.
        If no value is passed, the RNG will not be seeded, so
        serialized versions of this source will not product the
        same values. Note: strings that are passed to seed will
        be converted to UTF-8 bytes before being converted to
        integers for seeding.
    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: :class:WorleyCell object.
    :rtype: pjinoise.sources.WorleyCell
    """
    def __init__(self, antialias: bool = True, *args, **kwargs) -> None:
        self.antialias = antialias
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = None) -> np.ndarray:
        """Return a space filled with noise."""
        m = 0
        a = np.zeros(size, dtype=float)
        volume = self.volume
        if volume is None:
            volume = size
        volume = np.array(volume)

        # Place the seeds in the overall volume of noise.
        seeds = self._rng.random((self.points, 3))
        seeds = np.round(seeds * (volume - 1))
        seeds += np.array(self.origin)

        # Assign a color for each seed.
        colors = [n / (self.points - 1) for n in range(self.points)]

        # Map the distances to the points.
        def _hypot(point, indices, i):
            diffs = (indices - point) ** 2
            diffs = np.sqrt(np.sum(diffs, -1))
            result = np.zeros(diffs.shape, dtype=[('d', float), ('i', float)])
            result['d'] = diffs
            result['i'].fill(i)
            return result

        indices = np.indices(size)
        indices = np.transpose(indices, (1, 2, 3, 0))
        dist = np.zeros((self.points, *size),
                        dtype=[('d', float), ('i', float)])
        for i in range(self.points):
            dist[i] = _hypot(seeds[i], indices, i)
        dist = np.transpose(dist, (1, 2, 3, 0))
        dist.sort(3, order='d')
        map = dist[:, :, :, 0]['i']
        a = np.take(colors, map.astype(int))

        if self.antialias:
            nmap = dist[:, :, :, 1]['i']
            m = dist[:, :, :, 1]['d'] - dist[:, :, :, 0]['d'] < 1
            b = np.take(colors, nmap.astype(int))
            x = dist[:, :, :, 1]['d'] - dist[:, :, :, 0]['d']
            x[m] = 1 - (x[m] / 2 + .5)
            a[m] = c.lerp(a[m], b[m], x[m])

        return a


# Random noise using unit cubes.
class UnitNoise(Source):
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
        self._seed = c.text_to_int(self.seed)

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
                       axes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
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
    :param origin: (Optional.) Where in the grid to start the path.
        This can be either a descriptive string or a three-dimensional
        coordinate. It defaults to the top-left corner of the first
        three-dimensional slice of the data.
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

    Descriptive Origins
    -------------------
    The origin parameter can accept a description of the location
    instead of direct coordinates. This string must either be two
    words delimited by a hyphen or two letters. The first position
    sets the Y axis location can be one of the following options:

    *   top | t
    *   middle | m
    *   bottom | b

    The second position sets the X axis position and can be one of
    the following options:

    *   left | l
    *   middle | m
    *   bottom | b
    """
    def __init__(self, width: float = .2,
                 inset: Sequence[int] = (0, 1, 1),
                 origin: Union[str, Sequence[int]] = (0, 0, 0),
                 *args, **kwargs) -> None:
        """Initialize an instance of Path."""
        super().__init__(*args, **kwargs)
        self.width = width
        self.inset = inset
        self.origin = origin

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Fill a space with image data."""
        values, unit_dim = self._build_grid(size, loc)

        # The cursor will be used to determine our current position
        # on the grid as we create the path.
        cursor = self._calc_origin(self.origin, unit_dim)

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
        # to allow us to go back up the path and create a new branch
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
        return self._draw_path(path, size)

    # Private methods.
    def _build_grid(self, size, loc):
        """Create a grid of values. This uses the same technique
        Perlin noise uses to add randomness to the noise. A table of
        values was shuffled, and we use the coordinate of each vertex
        within the grid as part of the process to lookup the table
        value for that vertex. This grid will be used to determine the
        route the path follows through the space.
        """
        unit_dim = [int(s / u) for s, u in zip(size, self.unit)]
        unit_dim = tuple(np.array(unit_dim) + np.array((0, 1, 1)))
        unit_dim = tuple(np.array(unit_dim) - np.array(self.inset) * 2)
        unit_indices = np.indices(unit_dim)
        for axis in X, Y:
            unit_indices[axis] += loc[axis]
        unit_indices[Z].fill(loc[Z])
        values = np.take(self.table, unit_indices[X])
        values += unit_indices[Y]
        values = np.take(self.table, values % len(self.table))
        values += unit_indices[Z]
        values = np.take(self.table, values & len(self.table))
        unit_dim = np.array(unit_dim)
        return values, unit_dim

    def _calc_origin(self, origin, unit_dim):
        "Determine the starting location of the cursor."
        # If origin isn't a string, no further calculation is needed.
        if not isinstance(origin, str):
            return origin

        # Coordinates serialized as strings should be comma delimited.
        if ',' in origin:
            return c.text_to_int(origin)

        # If it's neither of the above, it's a descriptive string.
        result = [0, 0, 0]
        if '-' in origin:
            origin = origin.split('-')

        # Allow middle to be a shortcut for middle-middle.
        if origin == 'middle' or origin == 'm':
            origin = 'mm'

        # Set the Y axis coordinate.
        if origin[0] in ('top', 't'):
            result[Y] = 0
        if origin[0] in ('middle', 'm'):
            result[Y] = unit_dim[Y] // 2
        if origin[0] in ('bottom', 'b'):
            result[Y] = unit_dim[Y] - 1

        # Set the X axis coordinate.
        if origin[1] in ('left', 'l'):
            result[X] = 0
        if origin[1] in ('middle', 'm'):
            result[X] = unit_dim[X] // 2
        if origin[1] in ('right', 'r'):
            result[X] = unit_dim[X] - 1

        return result

    def _draw_path(self, path, size):
        a = np.zeros(size, dtype=float)
        width = int(self.unit[-1] * self.width)
        for step in path:
            start = self._unit_to_pixel(step[0])
            end = self._unit_to_pixel(step[1])
            slice_y = self._get_slice(start[Y], end[Y], width)
            slice_x = self._get_slice(start[X], end[X], width)
            a[:, slice_y, slice_x] = 1.0
        return a

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


class AnimatedPath(Path):
    """Animate the creation of a path.

    :param delay: (Optional.) The number of frames to wait before
        starting the animation.
    :param linger: (Optional.) The number of frames to hold on the
        last image of the animation.
    :param trace: (Optional.) Whether to show all of the path that
        had been walked to this point (True) or just show this step
        (False).
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
    :return: :class:AnimatedPath object.
    :rtype: pjinoise.sources.AnimatedPath
    """
    def __init__(self, delay=0, linger=0, trace=True, *args, **kwargs) -> None:
        self.delay = delay
        self.linger = linger
        self.trace = trace
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill(self, size: Sequence[int],
             loc: Sequence[int] = (0, 0, 0)) -> np.ndarray:
        a = super().fill(size, loc)
        for _ in range(self.delay):
            a = np.insert(a, 0, np.zeros_like(a[0]), 0)
        for _ in range(self.linger):
            a = np.insert(a, -1, a[-1], 0)
        return a

    # Private methods.
    def _draw_path(self, path, size):
        def _take_step(branch, frame):
            try:
                step = branch[index]
                start = self._unit_to_pixel(step[0])
                end = self._unit_to_pixel(step[1])
                slice_y = self._get_slice(start[Y], end[Y], width)
                slice_x = self._get_slice(start[X], end[X], width)
                frame[slice_y, slice_x] = 1.0
            except IndexError:
                pass
            except TypeError:
                pass
            return frame

        a = np.zeros(size, dtype=float)
        path = self._find_branches(path)
        width = int(self.unit[-1] * self.width)
        index = 0
        frame = a[0].copy()
        while index < size[Z] - 1:
            for branch in path:
                frame = _take_step(branch, frame)
            a[index + 1] = frame.copy()
            index += 1
            if not self.trace:
                frame.fill(0)
        return a

    def _find_branches(self, path):
        """Find the spots where the path starts from the same location
        and split those out into branches, so they can be animated to
        be walked at the same time.
        """
        branches = []
        index = 1
        starts = [step[0] for step in path]
        branch = [path[0],]
        while index < len(path):
            start = path[index][0]
            if start in starts[:index]:
                branches.append(branch)
                for item in branches:
                    bstarts = []
                    for step in item:
                        if step:
                            bstarts.append(step[0])
                        else:
                            bstarts.append(step)
                    if start in bstarts:
                        delay = bstarts.index(start) - 1
                        branch = [None for _ in range(delay)]
                        break
                else:
                    msg = "Couldn't find branch with start."
                    raise ValueError(msg)
            branch.append(path[index])
            index += 1
        branches.append(branch)

        # Make sure all the branches are the same length.
        biggest = max(len(branch) for branch in branches)
        for branch in branches:
            if len(branch) < biggest:
                branch.append(None)
        return branches


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
class TilePaths(Source):
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
    def __init__(self, tile_size: Sequence[int],
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
        seeds = list(self.seeds)
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