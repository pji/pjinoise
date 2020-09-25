"""
noise
~~~~~

These are the noise generation objects used by pjinoise.
"""
import math
from typing import Sequence, Union

from pjinoise.constants import X, Y, Z, AXES

# Classes.
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


class ValueNoise(Noise):
    """A class to generate value noise. Reference algorithms taken 
    from:
    
    https://www.scratchapixel.com/lessons/procedural-generation-virtual-worlds/procedural-patterns-noise-part-1/creating-simple-1D-noise
    """
    def __init__(self, 
                 permutation_table:Union[Sequence[int], None] = None,
                 unit_cube:int = 1024,
                 *args, **kwargs) -> None:
        """Initialize an instance of the object."""
        self.permutation_table = permutation_table
        self.unit_cube = unit_cube
        super().__init__(*args, **kwargs)
    
    # Public methods.
    def asdict(self) -> dict:
        data = super().asdict()
        data['permutation_table'] = self.permutation_table
        data['unit_cube'] = self.unit_cube
        return data
    
    def noise(self, x:float, y:float, z:float, located:bool = False) -> int:
        if not located:
            x = self._locate(x)
            z = self._locate(z)
        x_int = int(x) & 255
        z_int = int(z) & 255
        x_float = x - x_int
        z_float = z - z_int
        x1 = self._lerp(self.permutation_table[(x_int + z_int) % pt_len],
                        self.permutation_table[(x_int + z_int + 1) % pt_len],
                        x_float)
        x2 = self._lerp(self.permutation_table[(x_int + z_int + 1) % pt_len],
                        self.permutation_table[(x_int + z_int + 2) % pt_len],
                        x_float)
        value = self._lerp(x1, x2, z_float)
        return round(value)
    
    # Private methods.
    def _lerp(self, a:float, b:float, x:float) -> float:
        """Performs a linear interpolation."""
        return a * (1 - x) + b * x
    
    def _locate(self, n:float) -> float:
        """Locates the point in the unit cube."""
        return n // self.unit_cube + (n % self.unit_cube) / self.unit_cube


class CosineNoise(ValueNoise):
    """A class to produce cosine smoothed value noise."""
    # Public methods.
    def noise(self, x:float, y:float, z:float, located:bool = False) -> int:
        if not located:
            x = self._locate(x)
            z = self._locate(z)
        x_int = int(x) & 255
        z_int = int(z) & 255
        x_float = x - x_int
        z_float = z - z_int
        pt_len = len(self.permutation_table)
        x1 = self._lerp(self.permutation_table[(x_int + z_int) % pt_len],
                        self.permutation_table[(x_int + z_int + 1) % pt_len],
                        x_float)
        x2 = self._lerp(self.permutation_table[(x_int + z_int + 1) % pt_len],
                        self.permutation_table[(x_int + z_int + 2) % pt_len],
                        x_float)
        value = self._lerp(x1, x2, z_float)
        return round(value)
    
    # Private methods.
    def _lerp(self, a:float, b:float, x:float) -> float:
        """Eased linear interpolation function to smooth the noise."""
        x = (1 - math.cos(x * math.pi)) / 2
        return super()._lerp(a, b, x)


class OctaveCosineNoise(CosineNoise):
    def __init__(self, 
                 octaves:int = 6,
                 persistence:float = -4,
                 amplitude:float = 24,
                 frequency:float = 4,
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
    
    def noise(self, x:float, _:float, z:float, located:bool = False) -> int:
        coords = [self._locate(n) for n in (x, _, z)]
        total = 0
        max_value = 0
        for i in range(self.octaves):
            amp = self.amplitude + (self.persistence * i)
            freq = self.frequency * 2 ** i
            total += super().noise(*(n * freq for n in coords), True) * amp
            max_value += amp
        value = total / max_value
        return round(value)


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


