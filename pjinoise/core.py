"""
core
~~~~

The core image generation functions for pjinoise.
"""
import argparse
import json
from typing import (Any, Callable, List, Mapping, NamedTuple, Sequence, 
                    Tuple, Union)

import cv2
import numpy as np
from PIL import Image, ImageOps

from pjinoise.constants import COLOR, SUPPORTED_FORMATS, VIDEO_FORMATS
from pjinoise import filters
from pjinoise import generators as g
from pjinoise import operations as op


# Configuration objects.
class FilterConfig(NamedTuple):
    filter: str
    args: Sequence[Any]
    
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = {k: getattr(self, k) for k in self._fields}
        return attrs


class Layer(NamedTuple):
    """A layer of an image or animation."""
    mode: Callable
    data: np.ndarray


class LayerConfig(NamedTuple):
    generator: str
    args: List[Any]
    mode: str
    location: Sequence[int]
    filters: Sequence[FilterConfig]
    
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = {k: getattr(self, k) for k in self._fields}
        if attrs['filters']:
            attrs['filters'] = [filter.asdict() for filter in attrs['filters']]
        return attrs


class ImageConfig(NamedTuple):
    size: Sequence[int]
    layers: Sequence[LayerConfig]
    filters: Sequence[FilterConfig]
    color: Sequence[str]
    
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = {k: getattr(self, k) for k in self._fields}
        attrs['layers'] = [layer.asdict() for layer in attrs['layers']]
        attrs['filters'] = [filter.asdict() for filter in attrs['filters']]
        return attrs


class SaveConfig(NamedTuple):
    filename: str
    format: str
    mode: str
    framerate: Union[None, float] = 12
    
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = {k: getattr(self, k) for k in self._fields}
        return attrs


# Image creation functions.
def bake_image(array:np.ndarray, 
               scale:int, 
               mode:str, 
               color:Sequence[str]) -> np.ndarray:
    """Convert the values in the array from grayscale into a color 
    space.
    """
    if mode == 'L':
        array = np.around(array * scale).astype(np.uint8)
        return array
    
    if len(array.shape) == 3:
        arrays = [bake_image(array[i], scale, mode, color) 
                  for i in range(array.shape[0])]
        new_array = np.zeros((len(arrays), *arrays[0].shape))
        for i, a in enumerate(arrays):
            new_array[i] = a
        return new_array
    
    array = np.around(array * scale).astype(np.uint8)
    image = Image.fromarray(array)
    if color:
        colorize = filters.Colorize(*color)
        image = colorize.process(image)
    if image.mode != mode:
        image = image.convert(mode)
    array = np.array(image)
    return array


def blend_layers(layers:Sequence[Layer]) -> np.ndarray:
    """Blend the layers of an image or animation together.
    
    :param layers: The layers to blend together.
    :return: The result of the layer blending as an array.
    :rtype: numpy.ndarray
    """
    layers = [layer for layer in layers]
    result = np.zeros_like(layers[0].data)
    for layer in layers:
        result = layer.mode(result, layer.data)
    return result


def make_filters(fconfs:Sequence[FilterConfig]) -> Sequence[filters.ForLayer]:
    """Create filter objects based on the given configuration."""
    for fconf in fconfs:
        f_cls = filters.REGISTERED_FILTERS[fconf.filter]
        filter = f_cls(*fconf.args)
        yield filter
    

def make_image(conf:ImageConfig) -> np.ndarray:
    """Create the image from the given configuration."""
    layers = make_layers(conf.layers, conf.size)
    image = blend_layers(layers)
    ifilters = make_filters(conf.filters)
    image = filters.process(image, ifilters)
    return image


def make_layers(lconfs:Sequence[LayerConfig], 
                size:Sequence[int]) -> Sequence[Layer]:
    """Generate each layer of the image."""
    size = size[:]
    for conf in lconfs:
        gen_cls = g.registered_generators[conf.generator]
        gen = gen_cls(*conf.args)
        data = gen.fill(size, conf.location)

        lfilters = [filter for filter in make_filters(conf.filters)]
        for filter in lfilters:
            size = filters.preprocess(size, lfilters)
        data = filters.process(data, lfilters)
        data = filters.postprocess(data, lfilters)

        mode = op.registered_ops[conf.mode]
        yield Layer(mode, data)


def save_image(array:np.ndarray, config:SaveConfig) -> None:
    """Save the image file to disk."""
    if config.format not in VIDEO_FORMATS:
        image = Image.fromarray(array, mode=config.mode)
        image.save(config.filename, config.format)
    
    elif config.format in VIDEO_FORMATS:
        dim = (array.shape[1], array.shape[2])
        dim = dim[::-1]
        array = array.astype(np.uint8)
        array = np.flip(array, -1)
        codec = VIDEO_FORMATS[config.format]
        videowriter = cv2.VideoWriter(config.filename, 
                                      cv2.VideoWriter_fourcc(*codec),
                                      config.framerate,
                                      dim, True)
        for i in range(array.shape[0]):
            videowriter.write(array[i])
        videowriter.release()


# Interface.
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


def load_config(filename:str) -> Tuple[ImageConfig, SaveConfig]:
    """Load configuration from a file."""
    # Load the configuration file.
    with open(filename, 'r') as fh:
        text = fh.read()
    
    # Deserialize the configuration objects.
    config = json.loads(text)
    version = config['Version']
    if version == '0.0.1':
        def make_fconf(filter:Mapping) -> FilterConfig:
            return FilterConfig(filter['filter'], filter['args'])
        
        def make_lconf(layer:Mapping) -> LayerConfig:
            return LayerConfig(
                layer['generator'],
                layer['args'],
                layer['mode'],
                layer['location'],
                [make_fconf(filter) for filter in layer['filters']]
            )
        
        iconf = ImageConfig(
            config['ImageConfig']['size'],
            [make_lconf(layer) for layer in config['ImageConfig']['layers']],
            [make_fconf(filter) for filter in config['ImageConfig']['filters']],
            config['ImageConfig']['color']
        )
        sconf = SaveConfig(
            config['SaveConfig']['filename'],
            config['SaveConfig']['format'],
            config['SaveConfig']['mode'],
            config['SaveConfig']['framerate'],
        )
        return iconf, sconf
    else:
        raise ValueError(f'pjinoise.core does not recognize version {version}')


def make_config(args:argparse.Namespace) -> Tuple[ImageConfig, SaveConfig]:
    """Parse command line arguments into configuration."""
    def parse_filters(s:str) -> List[FilterConfig]:
        filters_ = []
        for filter in s.split('!'):
            if filter:
                name, *args_ = filter.split(':')
                filters_.append(FilterConfig(name, args_))
        return filters_
    
    size = tuple(args.size[::-1])
    layers = []
    for layer in args.noise:
        name, args_, loc, filters_, mode = layer.split('_')
        args_ = [arg for arg in args_.split(':') if arg]
        loc = tuple(int(n) for n in loc.split(':'))
        lfilters = parse_filters(filters_)
        layers.append(LayerConfig(name, args_, mode, loc, lfilters))
    
    color = COLOR[args.color]
    filters_ = parse_filters(args.filters)
    iconf = ImageConfig(size, layers, filters_, color)
    
    format = get_format(args.filename)
    sconf = SaveConfig(args.filename, format, args.mode)
    return iconf, sconf


def parse_cli_arguments() -> argparse.Namespace:
    """Parse the command line arguments into a namespace so they can 
    be turned into configuration.
    """
    # Define the command line options.
    options = {
        'color': {
            'args': ('-k', '--color',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': '',
                'help': 'The color for colorizing the image.'
            },
        },
        'filename': {
            'args': ('-o', '--filename',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The name for the output file.'
            },
        },
        'filters': {
            'args': ('-f', '--filters',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': '',
                'help': 'The filters to run on the final image.'
            },
        },
        'load_config': {
            'args': ('-l', '--load_config',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': '',
                'help': 'Load configuration from the given file.'
            },
        },
        'mode': {
            'args': ('-m', '--mode',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': 'L',
                'required': False,
                'help': 'The color space for the image.'
            },
        },
        'noise': {
            'args': ('-n', '--noise',),
            'kwargs': {
                'type': str,
                'action': 'append',
                'help': 'The configuration of a layer of noise.'
            },
        },
        'size': {
            'args': ('-s', '--size',),
            'kwargs': {
                'type': int,
                'nargs': '*',
                'action': 'store',
                'help': 'The dimensions of the output file.'
            },
        },
    }
    
    # Read the command line arguments.
    p = argparse.ArgumentParser(
        prog='PJINOISE',
        description='Generate noise.',
    )
    for option in options:
        args = options[option]['args']
        kwargs = options[option]['kwargs']
        p.add_argument(*args, **kwargs)
    return p.parse_args()


def save_config(iconf:ImageConfig, sconf:SaveConfig) -> None:
    """Serialize the current configuration to disk."""
    # Construct the configuration filename.
    name = sconf.filename.split('.')
    filename = '.'.join(name[:-1]) + '.json'
    
    # Serialize the configuration for storage.
    config = {
        'ImageConfig': iconf.asdict(),
        'SaveConfig': sconf.asdict(),
    }
    config = json.dumps(config, indent=4)
    
    # Write the serialized configuration to disk.
    with open(filename, 'w') as fh:
        fh.write(config)


# Main.
def main() -> None:
    """Mainline."""
    args = parse_cli_arguments()
    if args.load_config:
        iconf, sconf = load_config(args.load_config)
    else:
        iconf, sconf = make_config(args)
    layers = make_layers(iconf.layers, iconf.size)
    image = blend_layers(layers)
    image = bake_image(image, 0xff, sconf.mode, iconf.color)
    save_image(image, sconf)
    save_config(iconf, sconf)


if __name__ == '__main__':
    main()