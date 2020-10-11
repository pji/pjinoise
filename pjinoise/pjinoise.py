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
from copy import deepcopy
import json
from operator import itemgetter
import sys
from typing import List, Sequence, Union

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageColor

from pjinoise import noise
from pjinoise import ui
from pjinoise import filters
from pjinoise.constants import COLOR, SUPPORTED_FORMATS, WORKERS, X, Y, Z


# Registrations.
REGISTERED_FILTERS = {
    'rotate90': filters.rotate90,
    'skew': filters.skew,
}
SUPPORTED_NOISES = {
    'SolidNoise': noise.SolidNoise,
    'GradientNoise': noise.GradientNoise,
    'ValueNoise': noise.ValueNoise,
    'CosineNoise': noise.CosineNoise,
    'OctaveCosineNoise': noise.OctaveCosineNoise,
    'Perlin': noise.Perlin,
    'OctavePerlin': noise.OctavePerlin,
}


# Script configuration.
CONFIG = {
    # General script configuration,
    'filename': '',
    'format': '',
    'save_config': True,
    'difference_layers': 0,
    
    # Noise generation configuration.
    'ntypes': [],
    'size': (256, 256),
    'unit': (256, 256),
    
    # Octave noise configuration.
    'octaves': 0,
    'persistence': 0,
    'amplitude': 0,
    'frequency': 0,
    
    # Animation configuration.
    'loops': 0,
    
    # Postprocessing configuration.
    'autocontrast': False,
    'colorize': None,
    'filters': '',
}
FILTERS = []
FRAMERATE = 12
STATUS = None


# Script initialization.
def configure() -> None:
    """Configure the script from command line arguments."""
    epilog = ('COLORIZE: AVAILABLE COLORS\n'
              'The colors available for the --colorize option are:\r\n'
              '\n')
    color_temp = '{:>4}\t{}, {}\n'
    for color in COLOR:
        if color in ['t', 'T', '']:
            continue
        epilog += color_temp.format(color, COLOR[color][0], COLOR[color][1])
    epilog += ' \n'
    
    # Read the command line arguments.
    p = argparse.ArgumentParser(
        prog='PJINOISE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate noise.',
        epilog=epilog
    )
    p.add_argument(
        '-a', '--amplitude',
        type=float,
        action='store',
        required=False,
        default=24,
        help='The starting amplitude for octave noise generation.'
    )
    p.add_argument(
        '-A', '--autocontrast',
        action='store_true',
        required=False,
        default=False,
        help='Automatically adjust the contrast of the image.'
    )
    p.add_argument(
        '-c', '--save_config',
        action='store_true',
        required=False,
        help='Save the config to a file.'
    )
    p.add_argument(
        '-C', '--load_config',
        type=str,
        action='store',
        required=False,
        help='Read config from a file. Overrides most other arguments.'
    )
    p.add_argument(
        '-d', '--difference_layers',
        type=int,
        action='store',
        required=False,
        default=0,
        help='The number of noise spaces to difference.'
    )
    p.add_argument(
        '-f', '--frequency',
        type=float,
        action='store',
        required=False,
        default=4,
        help='The starting frequency for octave noise generation.'
    )
    p.add_argument(
        '-F', '--filters',
        type=str,
        action='store',
        required=False,
        default='',
        help='Filters for difference layers.'
    )
    p.add_argument(
        '-k', '--colorize',
        type=str,
        action='store',
        default='',
        required=False,
        help='Use the given color to colorize the noise.'
    )
    p.add_argument(
        '-n', '--ntypes',
        type=str,
        nargs='*',
        action='store',
        default=['GradientNoise',],
        required=False,
        help='The noise generators to use.'
    )
    p.add_argument(
        '-o', '--output_file',
        type=str,
        action='store',
        help='The name for the output file.'
    )
    p.add_argument(
        '-O', '--octaves',
        type=int,
        action='store',
        required=False,
        default=6,
        help='The octaves of noise for octave noise generation.'
    )
    p.add_argument(
        '-p', '--persistence',
        type=float,
        action='store',
        required=False,
        default=-4,
        help='How the impact of each octave changes in octave noise generation.'
    )
    p.add_argument(
        '-s', '--size',
        type=int,
        nargs='*',
        action='store',
        help='The dimensions of the output file.'
    )
    p.add_argument(
        '-u', '--unit',
        type=int,
        nargs='*',
        action='store',
        help='The dimensions in pixels of a unit of noise.'
    )
    args = p.parse_args()
    
    # Read the configuration from a file.
    if args.load_config:
        read_config(args.load_config)
        
        if args.colorize:
            CONFIG['colorize'] = COLOR[args.colorize]
        if args.size:
            CONFIG['size'] = args.size[::-1]
        if args.unit:
            CONFIG['unit'] = args.unit[::-1]
            for noise in CONFIG['noises']:
                noise['unit'] = args.unit[::-1]
        if not args.save_config:
            CONFIG['save_config'] = False
    
    # Turn the command line arguments into configuration.
    else:
        CONFIG['ntypes'] = args.ntypes
        CONFIG['size'] = args.size[::-1]
        CONFIG['unit'] = args.unit[::-1]
        CONFIG['difference_layers'] = args.difference_layers
        CONFIG['octaves'] = args.octaves
        CONFIG['persistence'] = args.persistence
        CONFIG['amplitude'] = args.amplitude
        CONFIG['frequency'] = args.frequency
        CONFIG['autocontrast'] = args.autocontrast
        CONFIG['colorize'] = COLOR[args.colorize]
        CONFIG['filters'] = args.filters
        CONFIG['noises'] = make_noises_from_config()
    
    # Configure arguments not overridden by the config file.
    CONFIG['filename'] = args.output_file
    CONFIG['format'] = get_format(args.output_file)
    
    # Deserialize serialized objects in the configuration.
    CONFIG['ntypes'] = [SUPPORTED_NOISES[item] for item in args.ntypes]
    noises = []
    for kwargs in CONFIG['noises']:
        cls = SUPPORTED_NOISES[kwargs['type']]
        n = cls(**kwargs)
        noises.append(n)
    CONFIG['noises'] = noises
    if CONFIG['filters']:
        global FILTERS
        FILTERS = parse_filter_command(CONFIG['filters'], 
                                       CONFIG['difference_layers'])


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


def make_noises_from_config() -> List[noise.BaseNoise]:
    """Make serialized noises from the command line configuration."""
    result = []
    for ntype in CONFIG['ntypes']:
        kwargs = {
            'type': ntype,
            'size': CONFIG['size'],
            'unit': CONFIG['unit'],
            'octaves': CONFIG['octaves'],
            'persistence': CONFIG['persistence'],
            'amplitude': CONFIG['amplitude'],
            'frequency': CONFIG['frequency'],
        }
        result.append(kwargs)
    
    while len(result) < CONFIG['difference_layers'] + 1:
        new_noise = deepcopy(result[0])
        result.append(new_noise)
    
    return result


def parse_filter_command(cmd:str, layers:int) -> List[List]:
    commands = [c.split('_') for c in cmd.split('+')]
    for c in commands:
        c[1] = c[1].split(':')
        if len(c) < 3:
            c.append('')
        c[2] = c[2].split(',')
        for i in range(len(c[2])):
            try:
                c[2][i] = int(c[2][i])
            except ValueError:
                pass
    
    parsed = []
    for layer in range(layers + 1):
        filters_ = []
        for c in commands:
            if layer % int(c[1][0]) == int(c[1][1]):
                filter = [REGISTERED_FILTERS[c[0]], c[2]]
                filters_.append(filter)
        parsed.append(filters_)
    return parsed


# File handling.
def read_config(filename:str) -> None:
    """Read the script configuration from a file."""
    # The global keyword has to be used here because I'll be changing 
    # the dictionary CONFIG points to. It doesn't need to be used in 
    # configure() because there I'm only changing the keys in the 
    # dictionary.
    global CONFIG
    
    with open(filename) as fh:
        contents = fh.read()
    CONFIG = json.loads(contents)


def save_config() -> None:
    """Save the current configuration to a file."""
    namepart = CONFIG["filename"].split(".")[0]
    filename = f'{namepart}.conf'
    config = deepcopy(CONFIG)
    config['ntypes'] = [cls.__name__ for cls in config['ntypes']]
    config['noises'] = [n.asdict() for n in config['noises']]
    with open(filename, 'w') as fh:
        fh.write(json.dumps(config, indent=4))


def save_image(n:'numpy.ndarray') -> None:
    """Save the given array as an image to disk."""
    # Ensure the values in the array are valid within the color 
    # space of the image.
    n = n.round()
    n = n.astype(np.uint8)
    
    if len(n.shape) == 2:
        img = Image.fromarray(n, mode='L')
        if CONFIG['autocontrast']:
            img = ImageOps.autocontrast(img)
        img.save(CONFIG['filename'], CONFIG['format'])
    
    if len(n.shape) == 3:
        frames = [Image.fromarray(n[i], mode='L') for i in range(n.shape[0])]
        if CONFIG['autocontrast']:
            frames = [ImageOps.autocontrast(frame) for frame in frames]
        if CONFIG['colorize']:
            black = CONFIG['colorize'][0]
            white = CONFIG['colorize'][1]
            frames = [ImageOps.colorize(f, black, white) for f in frames]
    
    
    if len(n.shape) == 3 and CONFIG['format'] not in ['AVI', 'MP4']:
        duration = (1 / FRAMERATE) * 1000
        frames[0].save(CONFIG['filename'], 
                       save_all=True,
                       append_images=frames[1:],
                       loop=CONFIG['loops'],
                       duration=duration)
    
    if len(n.shape) == 3 and CONFIG['format'] in ['AVI', 'MP4']:
        n = np.zeros((*n.shape, 3))
        for i in range(len(frames)):
            n[i] = np.asarray(frames[i])
        dim = (n.shape[1], n.shape[2])
        dim = dim[::-1]
        n = n.astype(np.uint8)
        n = np.flip(n, -1)
        if CONFIG['format'] == 'AVI':
            codec = 'MJPG'
        else:
            codec = 'mp4v'
        out = cv2.VideoWriter(CONFIG['filename'], 
                              cv2.VideoWriter_fourcc(*codec), 
                              FRAMERATE, dim, True)
        for frame in n:
            out.write(frame)
        out.release()


# Utility.
def _has_rotate90() -> bool:
    filter_fns = []
    for layer in FILTERS:
        for filter in layer:
            filter_fns.append(filter[0])
    if filters.rotate90 in filter_fns:
        return True
    return False


def _has_skew() -> bool:
    filter_fns = []
    for layer in FILTERS:
        for filter in layer:
            filter_fns.append(filter[0])
    if filters.skew in filter_fns:
        return True
    return False


# Noise creation.
def make_difference_noise(noises:Sequence[noise.BaseNoise],
                          size:Sequence[int]) -> 'numpy.ndarray':
    """Create a space filled with the difference of several 
    difference noise spaces.
    """
    if _has_skew():
        skews = []
        for layer in FILTERS:
            skews.extend(f for f in layer if f[0] == filters.skew)
        final_size = size[:]
        slopes = [abs(f[1][0]) for f in skews]
        slope = max(slopes)
        size = filters.skew_size_adjustment(size, slope)
    
    if _has_rotate90():
        rot_final_size = size[:]
        size = filters.rotate90_size_adjustment(size)
    
    spaces = [np.zeros(size) for _ in range(len(noises))]
    with futures.ProcessPoolExecutor(WORKERS) as executor:
        to_do = []
        for i in range(len(noises)):
            slice_loc = [0 for _ in range(len(size[:-2]))]
            while slice_loc[0] < size[0]:
                job = executor.submit(make_noise_slice, 
                                      noises[i],
                                      size[-2:],
                                      slice_loc[:],
                                      STATUS,
                                      i)
                to_do.append(job)
                slice_loc[-1] += 1
                for j in range(1, len(slice_loc))[::-1]:
                    if slice_loc[j] == size[j]:
                        slice_loc[j] = 0
                        slice_loc[j - 1] += 1
        
    for future in futures.as_completed(to_do):
        noise_loc, slice_loc, slice = future.result()
        spaces[noise_loc][slice_loc] = slice
        if STATUS:
            STATUS.update('slice_end', (noise_loc, slice_loc))
    
    result = np.zeros(spaces[0].shape)
    for i in range(len(spaces)):
        if STATUS:
            STATUS.update('diff', i)
        space = spaces[i]
        filters_ = FILTERS[i]
        for filter in filters_:
            if STATUS:
                STATUS.update('filter', filter[0].__name__)
            args = []
            if len(filter) > 1:
                args = filter[1]
            space = filter[0](space, *args)
            if STATUS:
                STATUS.update('filter_end', filter[0].__name__)
        result = abs(result - space)
    
    if _has_rotate90() and rot_final_size != size:
        diff = rot_final_size[Y] - rot_final_size[X]
        if diff > 0:
            start_x = diff // 2
            end_x = rot_final_size[X] + (diff - start_x)
            result = result[..., :, start_x:end_x]
        else:
            start_y = abs(diff // 2)
            end_y = rot_final_size[Y] + (abs(diff) - start_y)
            result = result[..., start_y:end_y, :]
    
    if _has_skew():
        start_x = (size[X] - final_size[X]) // 2
        end_x = final_size[X] + start_x
        result = result[..., :, start_x:end_x]
    
    if [filter for filter in FILTERS if filter]:
        return result[1:]
    else:
        return result


def make_noise(n:noise.BaseNoise, size:Sequence[int]) -> 'numpy.ndarray':
    """Create a space filled with noise."""
    if len(size) == 2:
        return n.fill(size)
    
    result = np.zeros(size)
    slice_loc = [0 for _ in range(len(size[:-2]))]
    with futures.ProcessPoolExecutor(WORKERS) as executor:
        to_do = []
        while slice_loc[0] < size[0]:
            job = executor.submit(make_noise_slice, 
                                  n, 
                                  size[-2:], 
                                  slice_loc[:],
                                  STATUS)
            to_do.append(job)
            slice_loc[-1] += 1
            for i in range(1, len(slice_loc))[::-1]:
                if slice_loc[i] == size[i]:
                    slice_loc[i] = 0
                    slice_loc[i - 1] += 1
    
    for future in futures.as_completed(to_do):
        loc, slice = future.result()
        result[loc] = slice
    
    return result
        

def make_noise_slice(n:noise.BaseNoise, 
                     size:Sequence[int],
                     slice_loc:Sequence[int],
                     status:ui.Status = None,
                     noise_loc:Union[int, None] = None) -> 'numpy.ndarray':
    """Create a two dimensional slice of noise."""
    if noise_loc is None:
        if status:
            status.update('slice', slice_loc)
        return (slice_loc, n.fill(size, slice_loc))
    
    if status:
        status.update('slice', [noise_loc, slice_loc])
    return (noise_loc, slice_loc, n.fill(size, slice_loc))


# Mainline.
def main() -> None:
    """Mainline."""
    global STATUS
    STATUS = ui.Status()
    configure()
    
    if CONFIG['difference_layers']:
        space = make_difference_noise(CONFIG['noises'], CONFIG['size'])
    else:
        space = make_noise(CONFIG['noises'][0], CONFIG['size'])
    
    save_image(space)
    STATUS.update('save_end', CONFIG['filename'])
    if CONFIG['save_config']:
        save_config()
    STATUS.end()


if __name__ == '__main__':
    main()