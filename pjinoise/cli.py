"""
cli
~~~

Run the command line for the pjinoise module.
"""
import argparse
from typing import Callable, List

from pjinoise import filters as f
from pjinoise import model as m
from pjinoise import operations as op
from pjinoise import sources as s
from pjinoise.common import get_format


def _build_source(name: str, args: str) -> s.ValueSource:
    cls = s.registered_sources[name]
    args = args.split(':')
    return cls(*args)


def _build_location(loc: str) -> List[int]:
    if not loc:
        return [0, 0, 0]
    return [int(n) for n in loc.split(':')[::-1]]


def _build_blend(blend: str) -> Callable:
    blend, *_ = blend.split(':')
    return op.registered_ops[blend]


def _build_blend_amount(blend: str) -> float:
    if ':' not in blend:
        return 1.0
    _, amount = blend.split(':')
    return float(amount)


def _build_filters(filters: str) -> List[f.ForLayer]:
    def _build_filter(filter: str) -> f.ForLayer:
        name, *args = filter.split(':')
        cls = f.registered_filters[name]
        return cls(*args)

    if not filters:
        return []
    return [_build_filter(filter) for filter in filters.split('+')]


def build_config(args: argparse.Namespace) -> m.Image:
    """Turn CLI arguments into a configuration argument."""
    layers = []
    for noise in args.noise:
        name, args_, loc, filters, blend = noise.split('_')
        layer = m.Layer(**{
            'source': _build_source(name, args_),
            'blend': _build_blend(blend),
            'blend_amount': _build_blend_amount(blend),
            'location': _build_location(loc),
            'filters': _build_filters(filters),
            'mask': None,
            'mask_filters': [],
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
        'framerate': args.framerate,
    })


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
        'framerate': {
            'args': ('-r', '--framerate',),
            'kwargs': {
                'type': float,
                'action': 'store',
                'help': 'The framerate for animations.'
            },
        },
        'location': {
            'args': ('-l', '--location',),
            'kwargs': {
                'type': int,
                'nargs': '*',
                'action': 'store',
                'help': 'The distance to offset image generation.'
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
