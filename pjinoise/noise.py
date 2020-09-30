"""
noise
~~~~~

These are the noise generation objects used by pjinoise.
"""
from abc import ABC, abstractmethod
from concurrent import futures
import math
import numpy as np
import random
from typing import List, Sequence, Tuple, Union

from pjinoise.constants import X, Y, Z, AXES, TEXT, WORKERS

# Base classes.
class BaseNoise(ABC):
    """Base class to define common features of noise classes."""
    def __init__(self, scale:int = 255) -> None:
        self.scale = 255
    
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
    
    def fill(self, size:Sequence[int]) -> np.array:
        """Return a space filled with noise."""
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

    @abstractmethod
    def noise(self, coords:Sequence[float]) -> None:
        """Generate the noise value for the given coordinates."""
    

class Noise():
    def __init__(self, scale:float = 255, *args, **kwargs) -> None:
        self.scale = scale
    
    def __eq__(self, other) -> bool:
        cls = type(self)
        if not isinstance(other, cls):
            return NotImplemented
        return self.asdict() == other.asdict()
    
    @classmethod
    def fromdict(cls, kwargs) -> object:
        """Create an instance of the class from a dictionary."""
        return cls(**kwargs)
    
    def asdict(self) -> dict:
        """Serialize the object as a dictionary."""
        return {
            'type': type(self).__name__,
            'scale': self.scale,
        }
    
    def noise(self, x:int, y:int, z:int) -> float:
        """Generate random noise."""
        return random.random()


# Simple solid color and gradients.
class SolidNoise(BaseNoise):
    """Produce a single color."""
    def __init__(self, color:int = 255, *args, **kwargs) -> None:
        self.color = color
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def noise(self, coords:Sequence[float]) -> int:
        return self.color


class GradientNoise(BaseNoise):
    """Produce a gradient over a multidimensional space."""
    def __init__(self, 
                 unit:Sequence[int], 
                 size:Union[Sequence[int], None] = None, 
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
        super().__init__(*args, **kwargs)
        self.unit = unit
        if size and not table:
            table = self._make_table(size)
        elif not size and not table:
            cls = self.__class__.__name__
            raise ValueError(TEXT['table_or_size'].format(cls))
        self.table = np.array(table)
    
    # Public methods.
    def asdict(self) -> dict:
        attrs = super().asdict()
        attrs['table'] = attrs['table'].tolist()
        return attrs
    
    def noise(self, coords:Sequence[float]) -> float:
        """Create a pixel of noise."""
        units = [self._locate(coords[i], i) for i in range(len(coords))]
        return self._dimensional_lerp(0, units)
    
    # Private methods.
    def _dimensional_lerp(self, index:int, units:Sequence) -> float:
        """Perform a recursive linear interpolation through all 
        dimensions of the noise.
        """
        coords_a = list(units[:])
        n_unit = int(coords_a[index])
        n_partial = coords_a[index] - n_unit
        coords_a[index] = n_unit
        
        coords_b = coords_a[:]
        coords_b[index] += 1
        
        coords_a = tuple(coords_a)
        coords_b = tuple(coords_b)
        
        if index == len(units) - 1:
            a = self.table[coords_a]
            b = self.table[coords_b]
        else:
            a = self._dimensional_lerp(index + 1, coords_a)
            b = self._dimensional_lerp(index + 1, coords_b)
                        
        return self._lerp(a, b, n_partial)
    
    def _lerp(self, a:float, b:float, x:float) -> float:
        """Performs a linear interpolation."""
        return a * (1 - x) + b * x

    def _locate(self, n:int, i:int) -> float:
        "Locate the unit position of the pixel along its axis."
        return n // self.unit[i] + (n % self.unit[i]) / self.unit[i]
    
    def _make_table(self, size:Sequence[int]) -> List:
        """Create a color table for vertices."""
        def fill_table(dim:Sequence[int]) -> List:
            """Recursively fill a table of the given dimensions."""
            if len(dim) > 1:
                result = [fill_table(dim[1:]) for _ in range(dim[0])]
            else:
                result = [random.randrange(self.scale) for _ in range(dim[0])]
            return result
        
        # Determine the dimensions of the table.
        # The "upside down" floor division here is the equivalent to 
        # ceiling division without the messy float conversion of the 
        # math.ceil() function.
        dim = [-(-n // u) + 1 for n, u in zip(size, self.unit)]
        
        # Create the table.
        table = fill_table(dim)
        return table


# Value noise.
# class ValueNoise(Noise):
class ValueNoise(GradientNoise):
    """A class to generate value noise. Reference algorithms taken 
    from:
    
    https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/procedural-patterns-noise-part-1/creating-simple-1D-noise
    """
    def __init__(self, 
                 unit:Sequence[int], 
                 table:Union[Sequence[int], None] = None,
                 *args, **kwargs) -> None:
        self.unit = unit
        if not table:
            table = self._make_table()
        self.table = np.array(table)
    
    # Public methods.
    def fill(self, size:Sequence[int], loc:Sequence[int] = None) -> np.array:
        """Return a space filled with noise."""
        # Create the space.
        result = np.zeros(size)
        
        # Fill the space with noise.
        index = [0 for i in range(len(size))]
        while index[0] < size[0]:
            coords = []
            if loc:
                coords = loc[:]
            coords.extend(index)
            result[tuple(index)] = self.noise(coords)
            index[-1] += 1
            for i in range(1, len(size))[::-1]:
                if index[i] == size[i]:
                    index[i] = 0
                    index[i - 1] += 1
                else:
                    break
        
        # Return the noise-filled space.
        return result

    def noise(self, coords:Sequence[float]) -> int:
        units = [self._locate(coords[i], i) for i in range(len(coords))]
        if len(units) % 2 == 0:
            return self._dimensional_lerp(0, units[1::2])
        return self._dimensional_lerp(0, units[::2])

    # Private methods.
    def _dimensional_lerp(self, index:int, units:Sequence) -> float:
        """Perform a recursive linear interpolation through all 
        dimensions of the noise.
        """
        coords_a = list(units[:])
        n_unit = int(coords_a[index])
        n_partial = coords_a[index] - n_unit
        coords_a[index] = n_unit
        
        coords_b = coords_a[:]
        coords_b[index] += 1
        
        if index == len(units) - 1:
            sum_a = sum(coords_a) % len(self.table)
            sum_b = sum(coords_b) % len(self.table)
        
            a = self.table[sum_a]
            b = self.table[sum_b]
        else:
            a = self._dimensional_lerp(index + 1, coords_a)
            b = self._dimensional_lerp(index + 1, coords_b)
                        
        return self._lerp(a, b, n_partial)
    
    def _make_table(self) -> List:
        table = [n for n in range(256)]
        table.extend(table)
        random.shuffle(table)
        return table


class CosineNoise(ValueNoise):
    """A class to produce cosine smoothed value noise."""
    # Private methods.
    def _lerp(self, a:float, b:float, x:float) -> float:
        """Eased linear interpolation function to smooth the noise."""
        x = (1 - math.cos(x * math.pi)) / 2
        return super()._lerp(a, b, x)


class OctaveCosineNoise(CosineNoise):
    """Generate octaves of alternating dimensional value noise with a 
    cosine-based easing function.
    """
    def __init__(self, 
                 octaves:int = 4,
                 persistence:float = 8,
                 amplitude:float = 8,
                 frequency:float = 2,
                 *args, **kwargs) -> None:
        self.octaves = octaves
        self.persistence = persistence
        self.amplitude = amplitude
        self.frequency = frequency
        super().__init__(*args, **kwargs)

    def asdict(self) -> dict:
        data = super().asdict()
        data['octaves'] = self.octaves
        data['persistence'] = self.persistence
        data['amplitude'] = self.amplitude
        data['frequency'] = self.frequency
        return data
    
    def noise(self, coords:Sequence[float]) -> int:
        total = 0
        max_value = 0
        for i in range(self.octaves):
            amp = self.amplitude + (self.persistence * i)
            freq = self.frequency * 2 ** i
            new_coords = tuple(n * freq for n in coords)
            total += super().noise(new_coords) * amp
            max_value += amp
        value = total / max_value
        return round(value)


# Perlin noise.
class Perlin(Noise):
    """A class to generate Perlin noise."""
    def __init__(self,
                 permutation_table:Union[Sequence[int], None] = None,
                 unit_cube:int = 1024,
                 repeat:int = 0,
                 *args, **kwargs) -> None:
        """Initialize an instance of the object."""
        self.permutation_table = permutation_table
        self.unit_cube = unit_cube
        self.repeat = repeat
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def asdict(self) -> dict:
        data = super().asdict()
        data['permutation_table'] = self.permutation_table
        data['unit_cube'] = self.unit_cube
        data['repeat'] = self.repeat
        return data
    
    def perlin(self, x:float, y:float, z:float, located:bool = False) -> int:
        """Calculate the value of a pixel using a Perlin noise function."""
        if not located:
            coords = [self._locate(n) for n in (x, y, z)]
        else:
            coords = [x, y, z]
        
        if self.repeat > 0:
            coords = [coords[axis] % self.repeat for axis in AXES]
        
        coords_int = [int(coords[axis]) & 255 for axis in AXES]
        coords_float = [coords[axis] - int(coords[axis]) for axis in AXES]
        u, v, w = [self._fade(n) for n in coords_float]
        
        hashes = [f'{n:>03b}' for n in range(8)]
        hash_table = {hash: self._hash(hash, coords_int) for hash in hashes}
    
        x1 = self._lerp(self._grad('000', hash_table, coords_float),
                        self._grad('100', hash_table, coords_float),
                        u)
        x2 = self._lerp(self._grad('010', hash_table, coords_float),
                        self._grad('110', hash_table, coords_float),
                        u)
        y1 = self._lerp(x1, x2, v)
        x1 = self._lerp(self._grad('001', hash_table, coords_float),
                        self._grad('101', hash_table, coords_float),
                        u)
        x2 = self._lerp(self._grad('011', hash_table, coords_float),
                        self._grad('111', hash_table, coords_float),
                        u)
        y2 = self._lerp(x1, x2, v)
        
        value = (self._lerp(y1, y2, w) + 1) / 2
        return round(self.scale * value)
    
    def noise(self, *args, **kwargs) -> int:
        return self.perlin(*args, **kwargs)
    
    # Private methods.
    def _fade(self, t:float) -> float:
        """An easing function for Perlin noise generation."""
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    def _grad(self, mask:str, hash_table:dict, coords:Sequence[float]) -> float:
        """Return the dot product of the gradient and the location 
        vectors.
        """
        coords = coords[:]
        hash = hash_table[mask]
        for axis in AXES:
            if mask[axis] == '1':
                coords[axis] = coords[axis] - 1
        x, y, z = coords
        
        n = hash & 0xF
        out = 0
        if n == 0x0:
            out = x + y
        elif n == 0x1:
            out = -x + y
        elif n == 0x2:
            out = x - y
        elif n == 0x3:
            out = -x - y
        elif n == 0x4:
            out = x + z
        elif n == 0x5:
            out = -x + z
        elif n == 0x6:
            out = x - z
        elif n == 0x7:
            out = -x - z
        elif n == 0x8:
            out = y + z
        elif n == 0x9:
            out = -y + z
        elif n == 0xA:
            out = y - z
        elif n == 0xB:
            out = -y - z
        elif n == 0xC:
            out = y + x
        elif n == 0xD:
            out = -y + z
        elif n == 0xE:
            out = y - x
        elif n == 0xF:
            out = -y - z
        return out

    def _hash(self, pattern:str, coords:Sequence[int]) -> int:
        """Generate a hash for the given coordinates."""
        coords = coords[:]
        for axis in AXES:
            if pattern[axis] == '1':
                coords[axis] = self._inc(coords[axis])
        xy = self.permutation_table[coords[X]] + coords[Y]
        xyz = self.permutation_table[xy] + coords[Z]
        return self.permutation_table[xyz]
    
    def _inc(self, value:int, repeat:int = 0) -> int:
        """Increments the passed value, rolling over if repeat is given."""
        value += 1
        if repeat > 0:
            value %= repeat
        return value

    def _lerp(self, a:float, b:float, x:float) -> float:
        """Performs a linear interpolation."""
        return a + x * (b - a)
    
    def _locate(self, n:float) -> float:
        """Locates the point in the unit cube."""
        return n // self.unit_cube + (n % self.unit_cube) / self.unit_cube


class OctavePerlin(Perlin):
    """A class to generate octave Perlin noise."""
    def __init__(self, 
                 octaves:int = 6,
                 persistence:float = -4,
                 amplitude:float = 24,
                 frequency:float = 4,
                 *args, **kwargs) -> None:
        """Initialize an instance of the class.
        
        :param octaves: The number of different levels of detail in 
            the noise that is generated.
        :param persistence: How much stronger or weaker the effect of 
            each level of detail is on the noise.
        :param amplitude: (Optional.) How strongly the first level of 
            detail affects the generated noise.
        :param frequency: (Optional.) The size of the features in the 
            first level of detail. 
        :returns: None.
        :rtype: None
        """
        self.octaves = octaves
        self.persistence = persistence
        self.amplitude = amplitude
        self.frequency = frequency
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def asdict(self) -> dict:
        data = super().asdict()
        data['octaves'] = self.octaves
        data['persistence'] = self.persistence
        data['amplitude'] = self.amplitude
        data['frequency'] = self.frequency
        return data
    
    def noise(self, *args) -> float:
        return self.octave_perlin(*args)
    
    def octave_perlin(self, x, y, z) -> int:
        """Create the perlin noise with the given octaves and persistence.
    
        :param x: The x location of the pixel within the unit cube.
        :param y: The y location of the pixel within the unit cube.
        :param z: The z location of the pixel within the unit cube.
        :return: The color value of the pixel.
        :rtype: float
        """
        coords = [self._locate(n) for n in (x, y, z)]
        total = 0
        max_value = 0
        for i in range(self.octaves):
            amp = self.amplitude + (self.persistence * i)
            freq = self.frequency * 2 ** i
            total += self.perlin(*(n * freq for n in coords), True) * amp
            max_value += amp
        value = total / max_value
        return round(value)


if __name__ == '__main__':
    raise NotImplementedError