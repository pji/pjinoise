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
import sys

from PIL import Image

from pjinoise import noise
from pjinoise.constants import SUPPORTED_FORMATS


# Script configuration.
CONFIG = {
    'filename': '',
    'format': '',
    'noise': '',
    'size': (0, 0),
    'unit': (0, 0),
}
SUPPORTED_NOISES = {
    'GradientNoise': noise.GradientNoise,
}


# Script initialization.
def configure() -> None:
    """Configure the script from command line arguments."""
    # Read the command line arguments.
    p = argparse.ArgumentParser('Generate noise.')
    p.add_argument(
        '-n', '--noise_type',
        type=str,
        nargs='*',
        action='store',
        default=['GradientNoise',],
        required=False,
        help='The noise generator to use.'
    )
    p.add_argument(
        '-o', '--output_file',
        type=str,
        action='store',
        help='The name for the output file.'
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
    
    # Turn the command line arguments into configuration.
    CONFIG['filename'] = args.output_file
    CONFIG['format'] = get_format(args.output_file)
    CONFIG['noise'] = [SUPPORTED_NOISES[item] for item in args.noise_type]
    CONFIG['size'] = args.size
    CONFIG['unit'] = args.unit


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


# Image handling.
def save_image(noise:'numpy.array') -> None:
    """Save the given array as an image to disk."""
    img = Image.fromarray(noise)
    img.save(CONFIG['filename'], CONFIG['format'])


if __name__ == '__main__':
    print(sys.argv)
    raise NotImplementedError