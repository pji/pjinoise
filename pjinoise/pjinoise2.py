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
import sys
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageOps

from pjinoise import noise
from pjinoise import ui
from pjinoise.constants import SUPPORTED_FORMATS, WORKERS


# Script configuration.
CONFIG = {
    # General script configuration,
    'filename': '',
    'format': '',
    'save_config': True,
    
    # Noise generation configuration.
    'ntypes': [],
    'size': (0, 0),
    'unit': (0, 0),
    
    # Octave noise configuration.
    'octaves': 0,
    'persistence': 0,
    'amplitude': 0,
    'frequency': 0,
    
    # Animation configuration.
    'loops': 0,
    
    # Postprocessing configuration.
    'autocontrast': False,
}
STATUS = None
SUPPORTED_NOISES = {
    'SolidNoise': noise.SolidNoise,
    'GradientNoise': noise.GradientNoise,
    'ValueNoise': noise.ValueNoise,
    'CosineNoise': noise.CosineNoise,
    'OctaveCosineNoise': noise.OctaveCosineNoise,
}


# Script initialization.
def configure() -> None:
    """Configure the script from command line arguments."""
    # Read the command line arguments.
    p = argparse.ArgumentParser('Generate noise.')
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
        '-f', '--frequency',
        type=float,
        action='store',
        required=False,
        default=4,
        help='The starting frequency for octave noise generation.'
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
        default=[256, 256],
        action='store',
        help='The dimensions of the output file.'
    )
    p.add_argument(
        '-u', '--unit',
        type=int,
        nargs='*',
        default=[256, 256],
        action='store',
        help='The dimensions in pixels of a unit of noise.'
    )
    args = p.parse_args()
    
    # Read the configuration from a file.
    if args.load_config:
        read_config(args.load_config)
    
    # Turn the command line arguments into configuration.
    else:
        CONFIG['ntypes'] = args.ntypes
        CONFIG['size'] = args.size[::-1]
        CONFIG['unit'] = args.unit[::-1]
        CONFIG['octaves'] = args.octaves
        CONFIG['persistence'] = args.persistence
        CONFIG['amplitude'] = args.amplitude
        CONFIG['frequency'] = args.frequency
        CONFIG['autocontrast'] = args.autocontrast
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
    return result


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
        fh.write(json.dumps(config))


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
        frames[0].save(CONFIG['filename'], 
                       save_all=True,
                       append_images=frames[1:],
                       loop=CONFIG['loops'])


# Noise creation.
def make_noise(n:noise.BaseNoise, size:Sequence[int]) -> 'np.ndarray':
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
                     status:ui.Status = None) -> 'np.array':
    """Create a two dimensional slice of noise."""
    if status:
        status.update('slice', slice_loc)
    return (slice_loc, n.fill(size, slice_loc))


# Mainline.
def main() -> None:
    """Mainline."""
    global STATUS
    STATUS = ui.Status()
    configure()
    space = make_noise(CONFIG['noises'][0], CONFIG['size'])
    save_image(space)
    STATUS.update('save_end', CONFIG['filename'])
    if CONFIG['save_config']:
        save_config()
    STATUS.end()


if __name__ == '__main__':
    main()    