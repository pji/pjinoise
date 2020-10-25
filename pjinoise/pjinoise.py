"""
pjinoise
~~~~~~~~

Functions to create Perlin noise and other bitmapped textures for 
the tessels. The code for the Perlin noise generator was adapted 
from:

    https://adrianb.io/2014/08/09/perlinnoise.html
    http://samclane.github.io/Perlin-Noise-Python/
"""
from concurrent import futures
from copy import deepcopy
import json
from typing import Any, List, Mapping, Sequence, Union

import cv2
import numpy as np
from PIL import Image, ImageColor

from pjinoise import cli
from pjinoise import filters
from pjinoise import noise
from pjinoise import ui
from pjinoise.constants import SUPPORTED_FORMATS, VIDEO_FORMATS, WORKERS


# Registrations and defaults.
DEFAULT_CONFIG = {
    # General script configuration,
    'filename': '',
    'format': '',
    'save_config': True,
    'difference_layers': 0,
    
    # Noise generation configuration.
    'ntypes': ['GradientNoise',],
    'size': (256, 256),
    'unit': (256, 256),
    'start': [],
    
    # Octave noise configuration.
    'octaves': 6,
    'persistence': -4,
    'amplitude': 24,
    'frequency': 4,
    
    # Animation configuration.
    'framerate': 12,
    'loops': 0,
    
    # Postprocessing configuration.
    'autocontrast': False,
    'blur': None,
    'colorize': '',
    'curve': '',
    'filters': '',
    'grain': None,
    'overlay': False,
}


# Script initialization.
def configure() -> None:
    """Configure the script from command line arguments."""
    args = cli.parse_arguments()
    config = deepcopy(DEFAULT_CONFIG)
    
    # Read the configuration from a file.
    if args.load_config:
        config.update(read_config(args.load_config))
        
    # Turn the command line arguments into configuration, overriding 
    # anything from a config file.
    config.update(cli.make_config_from_arguments(args))
    config.setdefault('noises', [])
    for n in config['noises']:
        n_config = {k: config[k] for k in n if k in config}
        n.update(n_config)
    
    # Deserialize serialized objects in the configuration.
    config['format'] = get_format(args.output_file)
    config['ntypes'] = [noise.SUPPORTED_NOISES[item] 
                        for item in config['ntypes']]
    noises = []
    for kwargs in config['noises']:
        cls = noise.SUPPORTED_NOISES[kwargs['type']]
        n = cls(**kwargs)
        noises.append(n)
    config['noises'] = noises
    layer_filters = []
    if config['filters']:
        layer_filters = parse_filter_command(config['filters'], 
                                             config['difference_layers'])
    image_filters = []
    if config['autocontrast']:
        image_filters.append(filters.Autocontrast())
    if config['overlay']:
        image_filters.append(filters.Overlay(.2))
    if config['curve']:
        image_filters.append(filters.Curve(config['curve']))
    if config['colorize']:
        white = config['colorize'][0]
        black = config['colorize'][1]
        image_filters.append(filters.Colorize(white, black))
    if config['blur']:
        image_filters.append(filters.Blur(config['blur']))
    if config['grain']:
        image_filters.append(filters.Grain(config['grain']))
    
    # Set the global config.
    config['_layer_filters'] = layer_filters
    config['_image_filters'] = image_filters
    return config


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
                filter = filters.make_filter(c[0], c[2])
                filters_.append(filter)
        parsed.append(filters_)
    return parsed


def postprocess_image(a:np.ndarray, 
                      ifilter:Sequence[filters.ForImage],
                      status:ui.Status = None) -> Image.Image:
    """Run the configured post-creation filters and other post 
    processing steps.
    """
    a = a.round()
    a = a.astype(np.uint8)
    img = Image.fromarray(a, mode='L')
    return filters.process_image(img, ifilter, status)


# File handling.
def read_config(filename:str) -> None:
    """Read the script configuration from a file."""
    # The global keyword has to be used here because I'll be changing 
    # the dictionary CONFIG points to. It doesn't need to be used in 
    # configure() because there I'm only changing the keys in the 
    # dictionary.    
    with open(filename) as fh:
        contents = fh.read()
    config = json.loads(contents)
    return config


def save_config(config:Mapping[str, Any]) -> None:
    """Save the current configuration to a file.
    
    :param config: The configuration to save.
    :return: None.
    :rtype: NoneType
    """
    # Since mappings can be mutable, create a defensive copy.
    config = deepcopy(config)
    
    # Construct the filename for the saved config.
    namepart = config["filename"].split(".")[0]
    filename = f'{namepart}.conf'
    
    # Serialize any objects in the configuration.
    config['ntypes'] = [cls.__name__ for cls in config['ntypes']]
    config['noises'] = [n.asdict() for n in config['noises']]
    
    # Remove any private keys in the config. These are generally 
    # used for objects that aren't serialized.
    config = {k: config[k] for k in config if not k.startswith('_')}
    
    # Save the configuration.
    with open(filename, 'w') as fh:
        fh.write(json.dumps(config, indent=4))


def save_image(n:np.ndarray, 
               ifilter:Sequence[filters.ForImage],
               filename:str,
               format:int, 
               framerate:float,
               loops:int) -> None:
    """Save the given array as an image to disk."""
    # Ensure the values in the array are valid within the color 
    # space of the image.
    if len(n.shape) == 2:
        img = postprocess_image(n, ifilter)
        img.save(filename, format)
    
    if len(n.shape) == 3:
        frames = [postprocess_image(n[i], ifilter) for i in range(n.shape[0])]
    
    if len(n.shape) == 3 and format not in VIDEO_FORMATS:
        duration = (1 / framerate) * 1000
        frames[0].save(filename, 
                       save_all=True,
                       append_images=frames[1:],
                       loop=loops,
                       duration=duration)
    
    if len(n.shape) == 3 and format in VIDEO_FORMATS:
        n = np.zeros((*n.shape, 3))
        for i in range(len(frames)):
            n[i] = np.asarray(frames[i])
        dim = (n.shape[1], n.shape[2])
        dim = dim[::-1]
        n = n.astype(np.uint8)
        n = np.flip(n, -1)
        if format == 'AVI':
            codec = 'MJPG'
        else:
            codec = 'mp4v'
        out = cv2.VideoWriter(filename, 
                              cv2.VideoWriter_fourcc(*codec), 
                              framerate, dim, True)
        for frame in n:
            out.write(frame)
        out.release()


# Noise creation.
def make_difference_noise(noises:Sequence[noise.BaseNoise],
                          size:Sequence[int],
                          start_loc:Sequence[int] = None,
                          lfilters:Sequence[filters.ForLayer] = None,
                          status:ui.Status = None) -> np.ndarray:
    """Create a space filled with the difference of several 
    difference noise spaces.
    """
    # If no starting location was passed, start at the beginning.
    if not start_loc:
        start_loc = [0 for _ in range(len(size[:-2]))]
    
    # Adjust the size of the image to avoid filter artifacts.
    size = filters.preprocess(size, lfilters)
    
    # Create the jobs to generate the noise.
    spaces = np.zeros((len(noises), *size))
    with futures.ProcessPoolExecutor(WORKERS) as executor:
        to_do = []
        for i in range(len(noises)):
            slice_loc = start_loc[:]
            while slice_loc[0] < size[0] + start_loc[0]:
                job = executor.submit(make_noise_slice, 
                                      noises[i],
                                      size[-2:],
                                      slice_loc[:],
                                      status,
                                      i)
                to_do.append(job)
                slice_loc[-1] += 1
                for j in range(1, len(slice_loc))[::-1]:
                    if slice_loc[j] == size[j]:
                        slice_loc[j] = 0
                        slice_loc[j - 1] += 1
    
    # Gather the generated noise and put it into the correct spot 
    # in the generated volume of noise.
    for future in futures.as_completed(to_do):
        noise_loc, slice_loc, slice = future.result()
        slice_loc = [loc - offset for loc, offset in zip(slice_loc, start_loc)]
        spaces[noise_loc][slice_loc] = slice
        if status:
            status.update('slice_end', (noise_loc, slice_loc))
    
    # Run the filters on the noise.
    spaces = filters.process(spaces, lfilters, status)
    
    # Apply the difference layers.
    result = np.zeros(spaces.shape[1:])
    for i in range(len(spaces)):
        if status:
            status.update('diff', i)
        space = spaces[i]
        result = abs(result - space)
    
    # Crop the noise to remove the padding added to avoid artifacting 
    # from the filters.
    result = filters.postprocess(result, lfilters)
    
    # Return the result.
    return result


def make_noise(n:noise.BaseNoise, 
               size:Sequence[int],
               lfilters:Sequence[filters.ForLayer] = None,
               status:ui.Status = None) -> np.ndarray:
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
                                  status)
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
                     noise_loc:Union[int, None] = None) -> np.ndarray:
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
    status = ui.Status()
    config = configure()
    
    if config['difference_layers']:
        space = make_difference_noise(config['noises'], 
                                      config['size'], 
                                      config['start'],
                                      config['_layer_filters'],
                                      status)
    else:
        space = make_noise(config['noises'][0], config['size'], status)
    
    save_image(space, 
               config['_image_filters'], 
               config['filename'],
               config['format'],
               config['framerate'],
               config['loops'])
    status.update('save_end', config['filename'])
    if config['save_config']:
        save_config(config)
    status.end()


if __name__ == '__main__':
    main()