"""
pjinoise
~~~~~~~~

Functions to create Perlin noise and other bitmapped textures for 
the tessels. The code for the Perlin noise generator was adapted 
from:

    https://adrianb.io/2014/08/09/perlinnoise.html
    http://samclane.github.io/Perlin-Noise-Python/
"""
from PIL import Image


# Script configuration.
CONFIG = {
    'filename': '',
    'format': '',
}


def save_image(noise:'numpy.array') -> None:
    """Save the given array as an image to disk."""
    img = Image.fromarray(noise)
    img.save(CONFIG['filename'], CONFIG['format'])