"""
io
~~

Input/output for the pjinoise module.
"""
import json
from typing import Mapping, MutableMapping, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from pjinoise import model as m
from pjinoise.__version__ import __version__
from pjinoise.common import get_format
from pjinoise.constants import VIDEO_FORMATS


# Constants.
X, Y, Z = 2, 1, 0


def _update_location(map: MutableMapping, loc: Sequence) -> None:
    """Offset the location of the image generation for a given 
    amount across every layer used to make the image. This only 
    works on mutable mappings because it changes the data in place.
    """
    if 'location' in map:
        map['location'] = [n + m for n, m in zip(map['location'], loc)]
    if 'source' in map:
        if isinstance(map['source'], Sequence):
            for item in map['source']:
                _update_location(item, loc)
        else:
            _update_location(map['source'], loc)

def load_conf(filename: str, 
              args: Union[None, 'argparse.Namespace'] = None) -> m.Image:
    """Load a configuration file."""
    # Get the configuration from the file.
    with open(filename, 'r') as fh:
        conf_json = fh.read()
    conf = json.loads(conf_json)
    
    # Deserialize configuration based on the given version.
    if conf['Version'] == '0.2.0':
        # Allow CLI arguments to change or override values in the 
        # loaded config.
        if args and 'filename' in vars(args):
            conf['Image']['filename'] = args.filename
            conf['Image']['format'] = get_format(args.filename)
        if args and 'size' in vars(args):
            conf['Image']['size'] = args.size
        if args and 'location' in vars(args):
            _update_location(conf['Image'], args.location)
        
        # Deserialize and return the configuration object.
        return m.Image(**conf['Image'])
    
    # Otherwise, the version isn't recognized, so throw an error.
    else:
        raise ValueError(f'Version {conf["Version"]} not supported.')


def save_conf(conf: m.Image) -> None:
    """Save the configuration file."""
    # Determine the name of the config file.
    parts = conf.filename.split('.')[:-1]
    parts.append('json')
    conffile = '.'.join(parts)
    
    # Serialize the config.
    confmap = {
        'Version': __version__,
        'Image': conf.asdict(),
    }
    confjson = json.dumps(confmap, indent=4)
    
    # Save the config.
    with open(conffile, 'w') as fh:
        fh.write(confjson)


def save_image(a: np.ndarray, 
               filename: str, 
               format: str, 
               mode: str,
               framerate: Union[None, float] = None) -> None:
    """Save image data to disk."""
    # Save a still image.
    if format not in VIDEO_FORMATS:
        # If the image data is grayscale, it should be in the pjinoise 
        # default grayscale space, which is single values between zero 
        # and one. This can be detected by checking the shape of the 
        # incoming array. The Image.save function in pillow need these 
        # images to be two-dimensional and in the 'L' color space.
        if len(a.shape) == 3:
            a = a[0].copy()
            a = (a * 0xff).astype(np.uint8)
        
        # If the data is in color, it should be in the RGB color 
        # space. However, Image.save still needs it to be a three-
        # dimensional image: Y, X, color channel.
        if len(a.shape) == 4:
            a = a[0].copy()
        
        image = Image.fromarray(a, mode=mode)
        image.save(filename, format)
    
    else:
        dim = (a.shape[X], a.shape[Y])
        if len(a.shape) == 3:
            a = a * 0xff
        a = a.astype(np.uint8)
        a = np.flip(a, -1)
        codec = VIDEO_FORMATS[format]
        videowriter = cv2.VideoWriter(*[filename, 
                                        cv2.VideoWriter_fourcc(*codec),
                                        framerate,
                                        dim, 
                                        True])
        for i in range(a.shape[Z]):
            videowriter.write(a[i])
        videowriter.release()
