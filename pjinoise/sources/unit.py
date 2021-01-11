"""
unit
~~~~

Unit noise classes for the pjinoise module.
"""
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from numpy.random import default_rng

from pjinoise import common as c
from pjinoise.constants import X, Y, Z, P
from .caching import CachingMixin
from .source import Source, eased


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
