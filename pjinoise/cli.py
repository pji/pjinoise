"""
cli
~~~

Run the command line for the pjinoise module.
"""
import argparse

from pjinoise import model as m
from pjinoise import operations as op
from pjinoise import sources as s
from pjinoise.constants import SUPPORTED_FORMATS


def parse_cli_args() -> None:
    """Parse the command line arguments."""
    # Define the command line options.
    options = {
        'filename': {
            'args': ('-o', '--filename',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The name for the output file.'
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


def build_config(args: argparse.Namespace) -> m.Image:
    """Turn CLI arguments into a configuration argument."""
    layers = []
    for noise in args.noise:
        parts = noise.split('_')
        cls = s.registered_sources[parts[0]]
        args_ = parts[1].split(':')
        layer = m.Layer(**{
            'source': cls(*args_),
            'filters': [],
            'mask': None,
            'mask_filters': [],
            'blend': op.registered_ops[parts[-2]],
            'blend_amount': float(parts[-1])
        })
        layers.append(layer)
    if len(layers) == 1:
        layers = layers[0]
    
    return m.Image(**{
        'source': layers,
        'size': [int(n) for n in args.size[::-1]],
        'filename': args.filename,
        'format': get_format(args.filename),
        'mode': args.mode,
#         'framerate': args.framerate,
    })


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
