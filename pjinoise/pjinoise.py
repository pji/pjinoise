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
import json
from operator import itemgetter
import random
import time
from typing import Callable, List, Mapping, Sequence, Union

from PIL import Image, ImageDraw, ImageOps

from pjinoise import filters
from pjinoise import noise
from pjinoise.constants import X, Y, Z, AXES, P


SUPPORTED_FORMATS = {
    'bmp': 'BMP',
    'gif': 'GIF',
    'jpeg': 'JPEG',
    'jpg': 'JPEG',
    'png': 'PNG',
    'tif': 'TIFF',
    'tiff': 'TIFF',
}
SUPPORTED_FILTERS = {
    'cut_shadow': filters.cut_shadow,
    'pixelate': filters.pixelate,
}
SUPPORTED_NOISES = {
    'Noise': noise.Noise,
    'ValueNoise': noise.ValueNoise,
    'CosineNoise': noise.CosineNoise,
    'Perlin': noise.Perlin,
    'OctavePerlin': noise.OctavePerlin,
}


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
    

def parse_noises_list(noises:Sequence) -> Sequence[noise.Noise]:
    """Get the list of noise objects from the configuration."""
    results = []
    for noise in noises:
        try:
            cls = SUPPORTED_NOISES[noise['type']]
        except KeyError:
            reason = ('Can only deserialize to known subclasses of Noise.')
            raise ValueError(reason)
        kwargs = {key:noise[key] for key in noise if key != 'type'}
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
                     noises:Sequence[noise.Noise],
                     noise_type:noise.Noise = noise.OctavePerlin,
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
                     noise_gen:noise.Noise,
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
                      noises:Sequence[noise.Noise] = None,
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


# Mainline.
def main() -> None:
    print('Creating image {filename}.')
    start_time = time.time()
    
    # Parse command line arguments.
    args = parse_command_line_args()
    filename = args.filename
    
    # Read script configuration from a given config file.
    if args.config:
        print('Reading configuration file.')
        config = read_config(args.config)
    
    # Use the command line arguments to configure the script.
    else:
        print('Creating noise generators.')
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
    main()
