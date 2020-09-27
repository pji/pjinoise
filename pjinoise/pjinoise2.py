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

from pjinoise.constants import SUPPORTED_FORMATS


# Script configuration.
CONFIG = {
    'filename': '',
    'format': '',
}


# Script initialization.
def configure() -> None:
    """Configure the script from command line arguments."""
    # Read the command line arguments.
    p = argparse.ArgumentParser('Generate noise.')
    p.add_argument(
        '-s', '--size',
        type=int,
        nargs=2,
        default=[256, 256],
        action='store',
        help='The dimensions of the output file.'
    )
    p.add_argument(
        'filename',
        type=str,
        action='store',
        help='The name for the output file.'
    )
    args = p.parse_args()
    
    # Turn the command line arguments into configuration.
    CONFIG['filename'] = args.filename
    CONFIG['format'] = get_format(args.filename)
    CONFIG['size'] = args.size


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