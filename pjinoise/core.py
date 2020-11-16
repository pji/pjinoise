"""
core
~~~~

The core image generation functions for pjinoise.
"""
import argparse
import json
from queue import Queue
from threading import Thread
from typing import (Any, Callable, List, Mapping, NamedTuple, Sequence, 
                    Tuple, Union)

import cv2
import numpy as np
from PIL import Image, ImageOps

from pjinoise.constants import COLOR, SUPPORTED_FORMATS, VIDEO_FORMATS
from pjinoise import filters
from pjinoise import generators as g
from pjinoise import operations as op
from pjinoise import ui


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
    args: Sequence[Any] = tuple()


class LayerConfig(NamedTuple):
    generator: g.ValueGenerator
    mode: str
    location: Sequence[int]
    filters: Sequence[FilterConfig]
    
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = {k: getattr(self, k) for k in self._fields}
        attrs['args'] = attrs['generator'].asargs()
        attrs['generator'] = g.get_regname_for_class(attrs['generator'])
        if attrs['filters']:
            attrs['filters'] = [filter.asdict() for filter in attrs['filters']]
        return attrs


class ImageConfig(NamedTuple):
    size: Sequence[int]
    layers: Sequence[LayerConfig]
    filters: Sequence[FilterConfig]
    color: Sequence[str]
    mode: str = 'difference'
    
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = {k: getattr(self, k) for k in self._fields}
        attrs['layers'] = [layer.asdict() for layer in attrs['layers']]
        attrs['filters'] = [filter.asdict() for filter in attrs['filters']]
        return attrs


class SaveConfig(NamedTuple):
    filename: str
    format: str
    mode: str = 'L'
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


def blend_images(images:Sequence[Image.Image], 
                 iconfs:Sequence[ImageConfig],
                 mode:str) -> Image.Image:
    if len(images) < 2:
        return images[0]
    a = None
    for image, conf in zip(images, iconfs):
        if a is None:
            a = np.zeros(image.shape, np.float64)
        amount = 1
        mode = conf.mode
        if ':' in mode:
            mode, amount = mode.split(':')
        a = op.registered_ops[mode](a, image, amount)
    return a


def blend_layers(layers:Sequence[Layer]) -> np.ndarray:
    """Blend the layers of an image or animation together.
    
    :param layers: The layers to blend together.
    :return: The result of the layer blending as an array.
    :rtype: numpy.ndarray
    """
    layers = [layer for layer in layers]
    result = np.zeros_like(layers[0].data)
    for layer in layers:
        result = layer.mode(result, layer.data, *layer.args)
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
    ifilters = tuple(make_filters(conf.filters))
    image = filters.process(image, ifilters)
    return image


def make_layers(lconfs:Sequence[LayerConfig], 
                size:Sequence[int]) -> Sequence[Layer]:
    """Generate each layer of the image."""
    for conf in lconfs:
        lsize = size[:]
        lfilters = [filter for filter in make_filters(conf.filters)]
        for filter in lfilters:
            lsize = filters.preprocess(size, lfilters)

        data = conf.generator.fill(lsize, conf.location)
        data = filters.process(data, lfilters)
        data = filters.postprocess(data, lfilters)
        
        # The image data should be between 0 and 1. Sometimes it ends 
        # up outside of that range. I'm not sure why. It could be 
        # I have a logical error somewhere, but it could be normal 
        # problems with floating point math. This step renormalizes 
        # the image data for the layer.
        if np.max(data) > 1 or np.min(data) < 0:
            data_min = np.min(data)
            data_max = np.max(data)
            scale = data_max - data_min
            data = data + data_min
            data = data / scale

        mode, *args = conf.mode.split(':')
        mode = op.registered_ops[mode]
        yield Layer(mode, data, args)


def save_image(array:np.ndarray, config:SaveConfig) -> None:
    """Save the image file to disk."""
    if config.format not in VIDEO_FORMATS:
        if len(array.shape) == 4 and config.mode != 'L':
            array = array[0].copy()
        if len(array.shape) == 3 and config.mode == 'L':
            array = array[0].copy()
        array = array.astype(np.uint8)
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


def load_config(filename:str, 
                args:argparse.Namespace) -> Tuple[ImageConfig, SaveConfig]:
    """Load configuration from a file."""
    # Load the configuration file.
    with open(filename, 'r') as fh:
        text = fh.read()
    
    # Deserialize the configuration objects.
    config = json.loads(text)
    version = config['Version']
    if version == '0.0.0':
        # Modify the loaded configuration based on the CLI options.
        if args.filename:
            config['filename'] = args.filename
            config['format'] = get_format(args.filename)
        
        # Parse the image filter configuration.
        filters_ = []
        for filter in config['filters'].split('+'):
            part = filter.split('_')
            place = part[1].split(':')
            place = [int(s) for s in place]
            args = part[2].split(',')
            filters_.append((*place, FilterConfig(part[0], args)))
        
        # Turn the noises into generators.
        noise_to_gen = {
            'OctaveCosineNoise': g.OldOctaveCosineCurtains,
        }
        layers = []
        for i, noise in enumerate(config['noises']):
            gencls = noise_to_gen[noise['type']]
            del noise['type']
            gen = gencls(**noise)
            loc = [0, 0, 0]
            if 'start' in noise:
                loc[0] = noise['start'][0]
            lconf = LayerConfig(
                gencls(**noise),
                'difference',
                loc,
                []
            )
            for filter in filters_:
                if i % filter[0] == filter[1]:
                    lconf.filters.append(filter[2])
            layers.append(lconf)
        
        # Construct the image filters.
        ifconf = []
        config.setdefault('autocontrast', None)
        config.setdefault('overlay', None)
        config.setdefault('curve', None)
        config.setdefault('blur', None)
        config.setdefault('grain', None)
        if config['autocontrast']:
            ifconf.append(FilterConfig('contrast', []))
        if config['overlay']:
            ifconf.append(FilterConfig('overlay', [.2,]))
        if config['curve']:
            ifconf.append(FilterConfig('curve'), [config['curve'],])
        if config['blur']:
            ifconf.append(FilterConfig('blur', [config['blur'],]))
        if config['grain']:
            ifconf.append(FilterConfig('grain', [config['grain'],]))
        
        # Now it's time to construct the image config.
        iconf = ImageConfig(
            config['size'],
            layers,
            ifconf,
            config['colorize'],
            'difference'
        )
        sconf = SaveConfig(config['filename'], config['format'], 'RGB', 12)
        return [iconf,], sconf
    
    elif version == '0.0.1':
        # Modify the loaded configuration based on the CLI options.
        if args.color:
            config['ImageConfig']['color'] = COLOR[args.color]
        if args.location:
            for layer in config['ImageConfig']['layers']:
                layer['location'] = [n + m for n, m in zip(layer['location'], 
                                                           args.location[::-1])]
        if args.size:
            config['ImageConfig']['size'] = args.size[::-1]
        
        # Deserialize the modified configuration.
        def make_fconf(filter:Mapping) -> FilterConfig:
            return FilterConfig(filter['filter'], filter['args'])
        
        def make_lconf(layer:Mapping) -> LayerConfig:
            gencls = g.registered_generators[layer['generator']]
            return LayerConfig(
                gencls(*layer['args']),
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
        return [iconf,], sconf
    
    elif version == '0.0.2':
        # Modify the loaded configuration based on the CLI options.
        if args.color:
            config['ImageConfig']['color'] = COLOR[args.color]
        if args.location:
            locmod = args.location[::-1]
            for image in config['ImageConfig']:
                for layer in image['layers']:
                    loc = layer['location']
                    layer['location'] = [n + m for n, m in zip(loc, locmod)]
        if args.size:
            config['ImageConfig']['size'] = args.size[::-1]
        if args.filename:
            config['SaveConfig']['filename'] = args.filename
            config['SaveConfig']['format'] = get_format(args.filename)
        
        # Deserialize the modified configuration.
        def make_fconf(filter:Mapping) -> FilterConfig:
            return FilterConfig(filter['filter'], filter['args'])
        
        def make_lconf(layer:Mapping) -> LayerConfig:
            gencls = g.registered_generators[layer['generator']]
            return LayerConfig(
                gencls(*layer['args']),
                layer['mode'],
                layer['location'],
                [make_fconf(filter) for filter in layer['filters']]
            )
        
        iconfs = []
        for item in config['ImageConfig']:
            iconf = ImageConfig(
                item['size'],
                [make_lconf(layer) for layer in item['layers']],
                [make_fconf(filter) for filter in item['filters']],
                item['color'],
                item['mode']
            )
            iconfs.append(iconf)
        sconf = SaveConfig(
            config['SaveConfig']['filename'],
            config['SaveConfig']['format'],
            config['SaveConfig']['mode'],
            config['SaveConfig']['framerate'],
        )
        return iconfs, sconf
    
    else:
        raise ValueError(f'pjinoise.core does not recognize version {version}')


def make_config(args:argparse.Namespace) -> Tuple[ImageConfig, SaveConfig]:
    """Parse command line arguments into configuration."""
    def parse_filters(s:str) -> List[FilterConfig]:
        filters_ = []
        for filter in s.split('+'):
            if filter:
                name, *args_ = filter.split(':')
                filters_.append(FilterConfig(name, args_))
        return filters_
    
    size = tuple(args.size[::-1])
    layers = []
    for layer in args.noise:
        name, args_, loc, filters_, mode = layer.split('_')
        args_ = [arg for arg in args_.split(':') if arg]
        gencls = g.registered_generators[name]
        if loc:
            loc = tuple(int(n) for n in loc.split(':'))
        else:
            loc = (0, 0, 0)
        if args.location:
            loc = tuple(n + int(m) for n, m in zip(loc, args.location[::-1]))
        lfilters = parse_filters(filters_)
        layers.append(LayerConfig(gencls(*args_), mode, loc, lfilters))
    
    color = COLOR[args.color]
    filters_ = parse_filters(args.filters)
    iconf = ImageConfig(size, layers, filters_, color)
    
    kwargs = {
        'filename': args.filename,
        'format': get_format(args.filename),
    }
    if args.mode:
        kwargs['mode'] = args.mode
    if args.framerate:
        kwargs['framerate'] = args.framerate
    sconf = SaveConfig(**kwargs)
    return [iconf,], sconf


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
        'framerate': {
            'args': ('-r', '--framerate',),
            'kwargs': {
                'type': float,
                'action': 'store',
                'help': 'The framerate of the animation.'
            },
        },
        'load_config': {
            'args': ('-c', '--load_config',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': '',
                'help': 'Load configuration from the given file.'
            },
        },
        'location': {
            'args': ('-l', '--location',),
            'kwargs': {
                'type': int,
                'nargs': '*',
                'action': 'store',
                'help': 'Offset the starting location of each generator.'
            },
        },
        'mode': {
            'args': ('-m', '--mode',),
            'kwargs': {
                'type': str,
                'action': 'store',
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


def save_config(iconfs:Sequence[ImageConfig], sconf:SaveConfig) -> None:
    """Serialize the current configuration to disk."""
    # Construct the configuration filename.
    name = sconf.filename.split('.')
    filename = '.'.join(name[:-1]) + '.json'
    
    # Serialize the configuration for storage.
    config = {
        'Version': "0.0.2",
        'ImageConfig': [iconf.asdict() for iconf in iconfs],
        'SaveConfig': sconf.asdict(),
    }
    config = json.dumps(config, indent=4)
    
    # Write the serialized configuration to disk.
    with open(filename, 'w') as fh:
        fh.write(config)


# Main.
def main(silent=True):
    """Mainline."""
    try:
        status = None
        args = parse_cli_arguments()
        if args.load_config:
            iconfs, sconf = load_config(args.load_config, args)
        else:
            iconfs, sconf = make_config(args)
        if not silent:
            stages = 2 + len(iconfs)
            status = Queue()
            t = Thread(target=ui.status_writer, args=(status, stages))
            t.start()
            status.put((ui.INIT,))
    
        if not silent:
            status.put((ui.STATUS, 'Generating images...'))
        images = []
        for i, iconf in enumerate(iconfs):
            image = make_image(iconf)
            image = bake_image(image, 0xff, sconf.mode, iconf.color)
            images.append(image)
            if not silent:
                status.put((ui.PROG, f'Image {i} of {len(iconfs)} generated.'))
    
        if not silent:
            status.put((ui.STATUS, 'Blending images...'))
        image = blend_images(images, iconfs, sconf.mode)
        if not silent:
            status.put((ui.PROG, 'Images blended.'))
    
        if not silent:
            status.put((ui.STATUS, 'Saving...'))
        save_image(image, sconf)
        save_config(iconfs, sconf)
        if not silent:
            status.put((ui.PROG, f'Saved as {sconf.filename}.'))
            status.put((ui.END, 'Good-bye.'))
    
    # Since the status updates run in an independent thread, letting 
    # exceptions bubble up from this thread clobbers causes the last 
    # status updates to clobber the last few lines of the exception. 
    # To avoid that, send the exception through the status update 
    # thread. This also ensures the status update thread is terminated. 
    except Exception as e:
        if status:
            status.put((ui.KILL, e))
        else:
            raise e


if __name__ == '__main__':
    main(False)