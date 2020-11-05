"""
generators
~~~~~~~~~~

Objects that generate values, both patterned and noise. These are not 
Python "generators," which is, I know, confusing. These are just 
generators in the sense they make things.
"""
from abc import ABC, abstractmethod
import random
from typing import Any, List, Sequence, Union

import numpy as np
from numpy.random import default_rng

from pjinoise.constants import TEXT, X, Y, Z
from pjinoise import ease as e

# Base classes.
class ValueGenerator(ABC):
    """Base class to define common features of noise classes."""
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.asdict() == other.asdict()
    
    # Public methods.
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = self.__dict__.copy()
        attrs['type'] = self.__class__.__name__
        if 'ease' in attrs:
            vals = list(e.registered_functions.values())
            keys = list(e.registered_functions)
            attrs['ease'] = keys[vals.index(attrs['ease'])]
        if 'table' in attrs:
            attrs['table'] = self.table.tolist()
        return attrs
    
    @abstractmethod
    def fill(self, size:Sequence[int], 
             location:Sequence[int] = None) -> np.ndarray:
        """Return a space filled with noise."""

    def noise(self, coords:Sequence[float]) -> int:
        """Generate the noise value for the given coordinates."""
        size = [1 for n in range(len(coords))]
        value = self.fill(size, coords)
        index = [0 for n in range(len(coords))]
        return value[index]
    

# Pattern generators.
class Gradient(ValueGenerator):
    """Generate a simple gradient."""
    def __init__(self, 
                 direction:str = 'h', 
                 ease:str = 'ioq',
                 *args) -> None:
        self.direction = direction
        self.ease = e.registered_functions[ease]
        
        # Parse the stops for the gradient.
        # A gradient stop sets the color at that position in the 
        # gradient. Because this needs to be configured from the 
        # command line, they come in as an ordered list of values, 
        # with the position being first and the color value being 
        # next. 
        self.stops = []
        for index in range(len(args))[::2]:
            try:
                stop = [float(args[index]), float(args[index + 1])]
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
    
    # Public methods.
    def fill(self, size:Sequence[int], 
             loc:Sequence[int] = (0, 0, 0)) -> np.ndarray:
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
        return self.ease(a)


class Lines(ValueGenerator):
    """Generate simple lines."""
    def __init__(self, 
                 direction:str = 'h', 
                 length:Union[int, str] = 64, 
                 ease:str = 'ioq',
                 *args, **kwargs) -> None:
        self.direction = direction
        self.length = int(length)
        self.ease = e.registered_functions[ease]
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def fill(self, size:Sequence[int], 
             loc:Sequence[int] = (0, 0, 0)) -> np.ndarray:
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
        return self.ease(values)


class Solid(ValueGenerator):
    def __init__(self, color:Union[str, float]) -> None:
        self.color = float(color)
    
    # Public methods.
    def fill(self, size:Sequence[int], 
             loc:Sequence[int] = (0, 0, 0)) -> np.ndarray:
        a = np.zeros(size)
        a.fill(self.color)
        return a
 

class Spheres(ValueGenerator):
    def __init__(self, radius:float, ease:str, offset:str = None) -> None:
        self.radius = float(radius)
        self.ease = e.registered_functions[ease]
        self.offset = offset
    
    # Public methods.
    def fill(self, size:Sequence[int], 
             loc:Sequence[int] = (0, 0, 0)) -> np.ndarray:
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
            a[X][~mask] = a[X][~mask] - self.radius
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
        a = self.ease(a)
        return a


class Spot(ValueGenerator):
    def __init__(self, radius:float, ease:str) -> None:
        self.radius = float(radius)
        self.ease = e.registered_functions[ease]
    
    # Public methods.
    def fill(self, size:Sequence[int], 
             loc:Sequence[int] = (0, 0, 0)) -> np.ndarray:
        """Return a space filled with noise."""
        # Map out the volume of space that will be created.
        a = np.indices(size)
        for axis in X, Y, Z:
            a[axis] += loc[axis]
        
            # Calculate where every point is relative to the center 
            # of the spot. 
            a[axis] = abs(a[axis] - size[axis] // 2)
        
        # Then determine what percentage of the way 
        # they are to being over the radius, with everything 
        # over the radius being 100%.
#         a[a > self.radius] = self.radius
        
        # Perform a spherical interpolation on the points in the 
        # volume and run the easing function on the results.
        a = np.sqrt(a[X] ** 2 + a[Y] ** 2)
        a = 1 - (a / np.sqrt(2 * self.radius ** 2))
        a[a > 1] = 1
        a[a < 0] = 0
        a = self.ease(a)
        return a


# Random noise generators.
class Curtains(ValueGenerator):
    """A class to generate value noise. Reference algorithms taken 
    from:
    
    https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/procedural-patterns-noise-part-1/creating-simple-1D-noise
    """
    def __init__(self, 
                 unit:Sequence[int], 
                 ease:str = '',
                 table:Union[Sequence[int], None] = None,
                 scale:int = 255) -> None:
        if isinstance(unit, str):
            unit = unit.split(',')
            unit = [int(n) for n in unit[::-1]]
        while len(unit) < 3:
            unit = list(unit)
            unit.insert(0, scale)
        self.unit = unit
        self.ease = e.registered_functions[ease]
        if not table:
            table = self._make_table()
        self.table = np.array(table)
        self.scale = scale
    
    # Public methods.
    def fill(self, size:Sequence[int], loc:Sequence[int] = None) -> np.ndarray:
        """Return a space filled with noise."""
        # Start by creating the two-dimensional matrix with the X 
        # and Z axis in order to save time by not running repetitive 
        # actions along the Y axis. Each position within the matrix 
        # represents a pixel in the final image.
        while len(size) < 3:
            size = list(size)
            size.insert(0, 1)
        xz_size = (size[Z], size[X])
        indices = np.indices(xz_size)
        
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
        unit = np.array((self.unit[Z], self.unit[X]))
        unit_distance = indices / unit[:, np.newaxis, np.newaxis]
        unit_floor = indices // unit[:, np.newaxis, np.newaxis]
        unit_distance = unit_distance - unit_floor
        
        # Look up and interpolate values.
        lerp = np.vectorize(self._lookup_and_lerp)
        zx_values = lerp(unit_floor[z], unit_distance[z], 
                         unit_floor[x], unit_distance[x])
        
        # Stretch X over the Y axis.
        result = np.zeros(size)
        for i in range(result.shape[Z]):
            try:
                result[i] = np.tile(zx_values[i], (result.shape[Y], 1))
            except TypeError as e:
                raise ValueError(f'i={i}, size={size}')
            except ValueError:
                raise ValueError(f'i={i}, size={size}')
        
        return self.ease(result)
    
    # Private methods.
    def _lerp(self, a:float, b:float, x:float) -> float:
        """Performs a linear interpolation."""
        return a * (1 - x) + b * x

    def _lookup_and_lerp(self, z:int, zd:float, x:int, xd:float) -> float:
        vertex_mask = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ])
        base_location = np.array([z, x])
        vertices = vertex_mask + base_location
        vertices = vertices.astype(int)
        
        try:
            x1a = self.table[sum(vertices[0]) % len(self.table)]
            x1b = self.table[sum(vertices[1]) % len(self.table)]
            x1 = self._lerp(x1a, x1b, xd)
            x2a = self.table[sum(vertices[2]) % len(self.table)]
            x2b = self.table[sum(vertices[3]) % len(self.table)]
            x2 = self._lerp(x2a, x2b, xd)
        except IndexError:
            msg = f'{sum(vertices[0])} % {len(self.table)}'
            raise ValueError(msg)
        
        return self._lerp(x1, x2, zd)
    
    def _make_table(self) -> List:
        table = [n for n in range(256)]
        table.extend(table)
        random.shuffle(table)        
        return table


class Values(ValueGenerator):
    """Produce a gradient over a multidimensional space."""
    def __init__(self, 
                 unit:Union[Sequence[int], str],
                 ease:str,
                 size:Union[Sequence[int], None] = (50, 720, 1280), 
                 table:Union[Sequence, None] = None, 
                 *args, **kwargs) -> None:
        """Initialize an instance of GradientNoise.
        
        :param unit: The number of pixels between vertices along an 
            axis. The vertices are the locations where colors for 
            the gradient are set.
        :param size: (Optional.) The expected size of the noise that 
            will be generated. This is only used if no table is passed. 
        :param table: (Optional.) The colors to set for the vertices. 
            They will repeat if there are more units along the axis 
            in an image then there are colors defined for that axis.
        """
        self.ease = e.registered_functions[ease]
        self.hashes = [f'{n:>03b}'[::-1] for n in range(2 ** 3)]
        if isinstance(unit, str):
            unit = self._normalize_coordinates(unit)
        self.unit = unit
        
        if table is None:
            table = self._make_table(size)
        self.table = np.array(table)
        self.shape = self.table.shape
        self.table = np.ravel(self.table)
    
    # Public methods.
    def fill(self, size:Sequence[int], 
             location:Sequence[int] = (0, 0, 0)) -> np.ndarray:
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
            a_hash = (hash_whole[Z] * self.shape[Y] * self.shape[X] 
                      + hash_whole[Y] * self.shape[X] 
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
        del y1, y2
        
        # Apply the easing function and return.
        return self.ease(z)
        
        
        # Create the space.
        result = np.zeros(size)
        
        # Pad missing dimensions in the size.
        size = size[:]
        if len(self.table.shape) < len(size):
            raise ValueError(TEXT['vol_dim_oob'])
        diff = len(self.table.shape) - len(size)
        padding = [0 for i in range(diff)]
        
        # Fill the space with noise.
        index = [0 for i in range(len(size))]
        while index[0] < size[0]:
            full_index = padding[:]
            full_index.extend(index)
            result[tuple(index)] = self.noise(full_index)
            index[-1] += 1
            for i in range(1, len(size))[::-1]:
                if index[i] == size[i]:
                    index[i] = 0
                    index[i - 1] += 1
                else:
                    break
        
        # Return the noise-filled space.
        return result

    # Private methods.
    def _normalize_coordinates(self, 
                               s:Union[Sequence[int], str]) -> Sequence[int]:
        if isinstance(s, str):
            result = s.split(',')
            result = [int(n) for n in unit[::-1]]
            return result
        return s
    
    def _lerp(self, a:float, b:float, x:float) -> float:
        """Performs a linear interpolation."""
        return a * (1 - x) + b * x

    def _make_table(self, size:Sequence[int]) -> List:
        """Create a color table for vertices."""
        def fill_table(dim:Sequence[int]) -> List:
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
        dim = [-(-n // u) + 1 for n, u in zip(size, self.unit)]
        
        # Create the table.
        table = fill_table(dim)
        return table


class Random(ValueGenerator):
    """Create random noise with a gaussian (normal) distribution."""
    def __init__(self, mid:float = .5, scale:float = .02, 
                 *args, **kwargs) -> None:
        self.mid = mid
        self.rng = default_rng()
        self.scale = scale
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def fill(self, size:Sequence[int], _:Any = None) -> np.array:
        random = self.rng.random(size) * self.scale * 2 - self.scale
        return random + self.mid


# Registration.
registered_generators = {
    'gradient': Gradient,
    'lines': Lines,
    'solid': Solid,
    'spheres': Spheres,
    'spot': Spot,
    
    'curtains': Curtains,
    'random': Random,
}


if __name__ == '__main__':
#     raise NotImplementedError
    
#     ring = Spheres(5, 'l', 'x')
#     spot = Spot(5, 'l')
#     val = spot.fill((1, 15, 15), (0, 0, 0))
#     random = Random(.5, .02)
#     gradient = Gradient('v', 'l', 0., 0., .5, 1., 1., 0.)
#     val = gradient.fill((2, 5, 4), (0, 0, 0))
    
    table = [
        [
            [0, 127, 255, 255],
            [0, 127, 255, 255],
            [0, 127, 255, 255],
            [0, 127, 255, 255],
        ],
        [
            [0, 127, 255, 255],
            [0, 127, 255, 255],
            [0, 127, 255, 255],
            [0, 127, 255, 255],
        ],
        [
            [0, 127, 255, 255],
            [0, 127, 255, 255],
            [0, 127, 255, 255],
            [0, 127, 255, 255],
        ],
    ]
    table = np.array(table) / 255
    table = table.tolist()
    tsize = (3, 4, 4)
    size = (1, 3, 5)
    unit = (2, 2, 2)
    obj = Values(size=tsize, ease='l', unit=unit)
    val = obj.fill(size)
    
    # For hash_tables:
    if isinstance(val, dict):
        for key in val:
            for plane in val[key]:
                for row in plane:
                    for column in row:
                        column = int(column * 0xff)
                        print(f'{column:02x}', end=' ')
                    print()
                print()
    
    if isinstance(val, np.ndarray):
        # For volumetric indicies.
        if len(val.shape) == 4:
            for axis in val:
                for plane in axis:
                    for row in plane:
                        for column in row:
                            column = int(column * 0xff)
                            print(f'{column:02x}', end=' ')
                        print()
                    print()
                print()
    
        # For volumes.
        elif len(val.shape) == 3:
            for plane in val:
                for row in plane:
                    for column in row:
                        column = int(column * 0xff)
                        print(f'{column:02x}', end=' ')
                    print()
                print()
    
        else:
            for column in val:
    #             print(column)
                column = int(column * 0xff)
                print(f'{column:02x}', end=' ')
            print()