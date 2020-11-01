"""
generators
~~~~~~~~~~

Objects that generate values, both patterned and noise. These are not 
Python "generators," which is, I know, confusing. These are just 
generators in the sense they make things.
"""
from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np

from pjinoise.constants import X, Y, Z
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
        values = self.ease(values)
        return values


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
        return self.ease(a)


# Registration.
registered_generators = {
    'lines': Lines,
    'spheres': Spheres,
}


if __name__ == '__main__':
#     raise NotImplementedError
    
    ring = Spheres(5, 'l', 'x')
    val = ring.fill((1, 15, 15))
    
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
    else:
        for plane in val:
            for row in plane:
                for column in row:
                    column = int(column * 0xff)
                    print(f'{column:02x}', end=' ')
                print()
            print()
