"""
generators
~~~~~~~~~~

Objects that generate values, both patterned and noise. These are not 
Python "generators," which is, I know, confusing. These are just 
generators in the sense they make things.
"""
from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

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
    'spheres': Spheres,
    'spot': Spot,
}


if __name__ == '__main__':
#     raise NotImplementedError
    
#     ring = Spheres(5, 'l', 'x')
#     spot = Spot(5, 'l')
#     val = spot.fill((1, 15, 15), (0, 0, 0))
#     random = Random(.5, .02)
    gradient = Gradient('v', 'l', 0., 0., .5, 1., 1., 0.)
    val = gradient.fill((2, 5, 4), (0, 0, 0))
    
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