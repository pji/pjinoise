"""
pjinoise
~~~~~~~~

Functions to create Perlin noise and other bitmapped textures for 
the tessels. The code for the Perlin noise generator was adapted 
from:

    https://adrianb.io/2014/08/09/perlinnoise.html
    http://samclane.github.io/Perlin-Noise-Python/
"""
import argparse
from concurrent import futures
from itertools import chain
import json
import math
from operator import itemgetter
import random
import time
from typing import Callable, List, Mapping, Sequence, Union

from PIL import Image, ImageDraw, ImageOps


X, Y, Z = 0, 1, 2
AXES = (X, Y, Z)
P = [151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 
     140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 
     120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 
     177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 
     165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 
     211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 
     25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 
     196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 
     52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 
     207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 
     119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 
     129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 
     218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 
     241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 
     157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 
     93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 
     180]
P.extend(P)
SUPPORTED_FORMATS = {
    'bmp': 'BMP',
    'gif': 'GIF',
    'jpeg': 'JPEG',
    'jpg': 'JPEG',
    'png': 'PNG',
    'tif': 'TIFF',
    'tiff': 'TIFF',
}


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
        self.permutation_table = make_permutations(p=permutation_table)
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
        x_int = int(x)
        z_int = int(z)
        x_float = x - x_int
        z_float = z - z_int
        x1 = self._lerp(self.permutation_table[x_int + z_int],
                        self.permutation_table[x_int + z_int + 1],
                        x_float)
        x2 = self._lerp(self.permutation_table[x_int + z_int + 1],
                        self.permutation_table[x_int + z_int + 2],
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


class Perlin(Noise):
    """A class to generate Perlin noise."""
    def __init__(self,
                 permutation_table:Union[Sequence[int], None] = None,
                 unit_cube:int = 1024,
                 repeat:int = 0,
                 *args, **kwargs) -> None:
        """Initialize an instance of the object."""
        self.permutation_table = make_permutations(p=permutation_table)
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
        maxValue = 0
        for i in range(self.octaves):
            amp = self.amplitude + (self.persistence * i)
            freq = self.frequency * 2 ** i
            total += self.perlin(*(n * freq for n in coords), True) * amp
            maxValue += amp
        value = total / maxValue
        return round(value)


# Utility functions.
def get_format(filename:str) -> str:
    """Determine the image type based on the filename."""
    name_part = filename.split('.')[-1]
    extension = name_part.casefold()
    try:
        return SUPPORTED_FORMATS[extension]
    except KeyError:
        print(f'The file type {name_part} is not supported.')
        supported = ', '.join(SUPPORTED_FORMATS)
        print(f'The supported formats are: {supported}.')
        raise SystemExit


def make_noise_generator_config(args:argparse.Namespace) -> Sequence[dict]:
    """Get the attributes of new noise objects from command line 
    arguments.
    """
    workers = 6
    noises = []
    with futures.ProcessPoolExecutor(workers) as executor:
        actual_workers = executor._max_workers
        to_do = []
        for t in range(args.diff_layers):
            value = None
            if args.use_default_permutation_table:
                value = P
            job = executor.submit(make_permutations, 2, value)
            to_do.append(job)
        
        for future in futures.as_completed(to_do):
            kwargs = {
                'type': args.noise_type,
                'scale': 255,
                'unit_cube': args.unit_cube,
                'repeat': 0,
                'permutation_table': future.result(),
                'octaves': args.octaves,
                'persistence': args.persistence,
                'amplitude': args.amplitude,
                'frequency': args.frequency,
            }
            noises.append(kwargs)
    return noises


def make_permutations(copies:int = 2, p:Sequence = None) -> Sequence[int]:
    """Return a hash lookup table."""
    try:
        if not p:
            values = list(range(256))
            random.shuffle(values)
            p = values[:]
            while copies > 1:
                random.shuffle(values)
                p.extend(values[:])
                copies -= 1
        elif len(p) < 512:
            reason = ('Permutation tables must have more than 512 values. ', 
                      f'The given table had {len(p)} values.')
            raise ValueError(reason)
    except TypeError:
        reason = ('The permustation table must be a Sequence or None. ', 
                  f'The given table was type {type(p)}.')
        raise TypeError(reason)
    return p


def parse_command_line_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    p = argparse.ArgumentParser(description='Generate Perlin noise.')
    p.add_argument(
        '-a', '--autocontrast',
        action='store_true',
        required=False,
        help='Correct the contrast of the image.'
    )
    p.add_argument(
        '-A', '--amplitude',
        type=float,
        action='store',
        default=24.0,
        required=False,
        help='How much the first octave affects the noise.'
    )
    p.add_argument(
        '-c', '--config',
        type=str,
        action='store',
        default='',
        required=False,
        help='How much the first octave affects the noise.'
    )
    p.add_argument(
        '-d', '--diff_layers',
        type=int,
        action='store',
        default=1,
        required=False,
        help='Number of difference cloud layers.'
    )
    p.add_argument(
        '-D', '--direction',
        type=int,
        nargs=3,
        action='store',
        default=[0, 0, 1],
        required=False,
        help='The direction an animation travels through the noise volume.'
    )
    p.add_argument(
        '-f', '--frequency',
        type=float,
        action='store',
        default=4,
        required=False,
        help='How the scan changes between octaves.'
    )
    p.add_argument(
        '-F', '--filters',
        type=str,
        action='store',
        default='',
        required=False,
        help=('A comma separated list of filters to run on the image. '
              f'The available filters are: {", ".join(SUPPORTED_FILTERS)}')
    )
    p.add_argument(
        '-l', '--loops',
        type=int,
        action='store',
        default=1,
        required=False,
        help='The number of times the animation should loop.'
    )
    p.add_argument(
        '-n', '--noise_type',
        type=str,
        action='store',
        default='OctavePerlin',
        required=False,
        help='The noise generator to use.'
    )
    p.add_argument(
        '-o', '--octaves',
        type=int,
        action='store',
        default=6,
        required=False,
        help='The levels of detail in the noise.'
    )
    p.add_argument(
        '-p', '--persistence',
        type=float,
        action='store',
        default=-4.0,
        required=False,
        help='The size of major features in the noise.'
    )
    p.add_argument(
        '-r', '--frames',
        type=int,
        action='store',
        default=1,
        required=False,
        help='The number of frames of animation.'
    )
    p.add_argument(
        '-s', '--save_config',
        action='store_true',
        required=False,
        help='Whether to save the noise configuration to a file.'
    )
    p.add_argument(
        '-u', '--unit_cube',
        type=int,
        action='store',
        default=1024,
        required=False,
        help='The size of major features in the noise.'
    )
    p.add_argument(
        '-T', '--use_default_permutation_table',
        action='store_true',
        required=False,
        help='Use the default permutation table.'
    )
    p.add_argument(
        '-x', '--width',
        type=int,
        action='store',
        default=256,
        required=False,
        help='The width of the image.'
    )
    p.add_argument(
        '-y', '--height',
        type=int,
        action='store',
        default=256,
        required=False,
        help='The height of the image.'
    )
    p.add_argument(
        '-z', '--slice_depth',
        type=int,
        action='store',
        default=None,
        required=False,
        help='The Z axis point for the 2-D slice of 3-D noise.'
    )
    p.add_argument(
        'filename',
        type=str,
        action='store',
        help='The name for the output file.'
    )
    return p.parse_args()


def parse_filter_list(text) -> Sequence[Callable]:
    """Get the list of filters to run on the image."""
    if not text:
        return []
    filters = text.split(',')
    return [SUPPORTED_FILTERS[filter] for filter in filters]
    

def parse_noises_list(noises:Sequence) -> Sequence[Noise]:
    """Get the list of noise objects from the configuration."""
    results = []
    for noise in noises:
        if noise['type'] not in SUPPORTED_NOISES:
            reason = ('Can only deserialize to known subclasses of Noise.')
            raise ValueError(reason)
        kwargs = {key:noise[key] for key in noise if key != 'type'}
        cls = globals()[noise['type']]
        results.append(cls(**kwargs))
    return results


def read_config(filename:str) -> dict:
    """Read noise creation configuration from a file."""
    with open(filename) as fh:
        contents = fh.read()
    return json.loads(contents)


def save_config(filename:str, conf:dict) -> None:
    """Write the noise generation configuration to a file to allow 
    generated noise to be reproduced.
    """
    conffile = f'{filename.split(".")[0]}.conf'
    contents = conf.copy()
    text = json.dumps(contents, indent=4)
    with open(conffile, 'w') as fh:
        fh.write(text)


# Generation functions.
def make_diff_layers(size:Sequence[int], 
                     z:int, 
                     diff_layers:int,
                     noises:Sequence[Noise],
                     noise_type:Noise = OctavePerlin,
                     x:int = 0, y:int = 0,
                     with_z:bool = False,
                     **kwargs) -> List[List[int]]:
    """Manage the process of adding difference layers of additional 
    noise to create marbled or veiny patterns.
    """
    while len(noises) < diff_layers:
        noises.append(noise_type(**kwargs))
    matrix = [[0 for y in range(size[Y])] for x in range(size[X])]
    for layer, noise in zip(range(diff_layers), noises):
        slice = make_noise_slice(size, z, noise, x, y)
        matrix = [[abs(matrix[x][y] - slice[x][y]) for y in range(size[Y])]
                                                   for x in range(size[X])]
    if with_z:
        return z, matrix
    return matrix


def make_noise_slice(size:Sequence[int], 
                     z:int, 
                     noise_gen:Noise,
                     x_offset:int = 0, 
                     y_offset:int = 0) -> list:
    """Create a two dimensional slice of Perlin noise."""
    slice = []
    for x in range(x_offset, size[X] + x_offset):
        col = []
        for y in range(y_offset, size[Y] + y_offset):                
            value = noise_gen.noise(x, y, z)
            col.append(value)
        slice.append(col)
    return slice


def make_noise_volume(size:Sequence[int],
                      z:int,
                      direction:Sequence[int],
                      length:int,
                      diff_layers:int,
                      noises:Sequence[Noise] = None,
                      workers:int = 6) -> Sequence[int]:
    """Create the frames to animate travel through the noise space. 
    This should look like roiling or turbulent clouds.
    """
    location = [0, 0, z]
    with futures.ProcessPoolExecutor(workers) as executor:
        actual_workers = executor._max_workers
        to_do = []
        for t in range(length):
            job = executor.submit(make_diff_layers, 
                                  size, 
                                  location[Z], 
                                  diff_layers, 
                                  noises, 
                                  x=location[X],
                                  y=location[Y],
                                  with_z=True)
            to_do.append(job)
            location = [location[axis] + direction[axis] for axis in AXES]
        
        for future in futures.as_completed(to_do):
            yield future.result()


# Filters.
def cut_shadow(values:Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
    """Remove the lower half of colors from the image."""
    values = values[:]
    brightest = max(chain(col for col in values))
    darkest = 127
    for x in range(len(values)):
        for y in range(len(values[x])):
            if values[x][y] < 128:
                values[x][y] = 0
            else:
                new_color = 255 * (values[x][y] - 127) / 128
                values[x][y] = round(new_color)
    return values


def pixelate(matrix:list, size:int = 32) -> list:
    """Create squares of color from the image."""
    matrix = matrix[:]
    x_start, y_start = 0, 0
    while x_start < len(matrix):
        x_end = x_start + size
        if x_end > len(matrix):
            x_end = len(matrix) % size
        
        while y_start < len(matrix[x_start]):
            y_end = y_start + size
            if y_end > len(matrix[x_start]):
                y_end = len(matrix[x_start]) % size
            
            square = [[n for n in col[y_start:y_end]] 
                         for col in matrix[x_start:x_end]]
            average = sum(chain(*square)) / size ** 2
            color = round(average)
            
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    matrix[x][y] = color
            
            y_start +=size
        x_start += size
        y_start = 0
    return matrix


# Filters and noises must be registered here to be available to main.
SUPPORTED_FILTERS = {
    'cut_shadow': cut_shadow,
    'pixelate': pixelate,
}
SUPPORTED_NOISES = {
    'Noise': Noise,
    'ValueNoise': ValueNoise,
    'Perlin': Perlin,
    'OctavePerlin': OctavePerlin,
}


# Mainline.
def main(config:Mapping, filename:str = 'cloud.tiff') -> None:
    print('Creating image {filename}.')
    start_time = time.time()
    
    # Deserialize config.
    filters = parse_filter_list(config['filters'])
    noises = parse_noises_list(config['noises'])
    format = get_format(filename)
    
    # Make noise volume.
    print('Creating slices of noise.')
    volume = []
    for slice in make_noise_volume(size=config['size'],
                                   z=config['z'],
                                   direction=config['direction'],
                                   length=config['frames'],
                                   diff_layers=config['diff_layers'],
                                   noises=noises):
        print(f'Created slice {slice[0] + 1}')
        volume.append(slice)
    volume = sorted(volume, key=itemgetter(0))
    volume = [slice[1] for slice in volume]
    print(f'{len(volume)} slices of noise created and sorted.')
    
    # Postprocess and turn into images.
    print('Postprocessing noise.')
    images = []
    for i in range(len(volume)):
        for filter in filters:
            volume[i] = filter(volume[i])
        img = Image.new(config['mode'], config['size'])
        drw = ImageDraw.Draw(img)
        for x in range(config['size'][X]):
            for y in range(config['size'][Y]):
                drw.point([x, y], volume[i][x][y])
        if config['autocontrast']:
            img = ImageOps.autocontrast(img)
        images.append(img)
    print('Postprocessing complete.')
    
    # Save the image and the configuration.
    print('Saving.')
    if len(images) > 1:
        images[0].save(filename, 
                       save_all=True, 
                       append_images=images[1:],
                       loop=config['loops'])
    else:
        img.save(filename, format=format)
    if config['save_conf']:
        save_config(filename, config)
    print(f'Image saved as {filename}')
    duration = time.time() - start_time
    print(f'Time to complete: {duration // 60}min {round(duration % 60)}sec.')


if __name__ == '__main__':
    # Define and parse the arguments.
    args = parse_command_line_args()
    
    # Read script configuration from a given config file.
    if args.config:
        config = read_config(args.config)
    
    # Use the command line arguments to configure the script.
    else:
        noises = make_noise_generator_config(args)
        z = args.slice_depth
        if z is None:
            z = random.randint(0, args.unit_cube)    
        config = {
            'mode': 'L',
            'size': [args.width, args.height],
            'diff_layers': args.diff_layers,
            'autocontrast': args.autocontrast,
            'z': z,
            'filters': args.filters,
            'save_conf': args.save_config,
            'frames': args.frames,
            'direction': args.direction,
            'loops': args.loops,
            'workers': 6,
            'noises': noises,
        }
    
    # Generate the image.
    main(config, args.filename)
