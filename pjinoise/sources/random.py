"""
random
~~~~~~

Sources for the pjinoise module that involve random generation.
"""
from typing import Sequence, Union

import cv2
import numpy as np
from numpy.random import default_rng

from pjinoise import common as c
from pjinoise import operations as op
from pjinoise.constants import X, Y, Z
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
