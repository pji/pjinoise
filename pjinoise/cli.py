"""
cli
~~~

The command line interface for the pjinoise module.
"""
import argparse
from copy import deepcopy
from typing import List, Mapping

from pjinoise.constants import COLOR
from pjinoise import noise


# Command line option configuration.
OPTIONS = {
    'amplitude': {
        'args': ('-a', '--amplitude',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'required': False,
            'help': 'The starting amplitude for octave noise generation.'
        },
    },
    'autocontrast': {
        'args': ('-A', '--autocontrast',),
        'kwargs': {
            'action': 'store_true',
            'required': False,
            'help': 'Automatically adjust the contrast of the image.'
        },
    },
    'blur': {
        'args': ('-b', '--blur',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'required': False,
            'help': 'Blur the image by the given amount.'
        },
    },
    'save_config': {
        'args': ('-c', '--save_config',),
        'kwargs': {
            'action': 'store_true',
            'required': False,
            'help': 'Save the config to a file.'
        },
    },
    'load_config': {
        'args': ('-C', '--load_config',),
        'kwargs': {
            'type': str,
            'action': 'store',
            'required': False,
            'help': 'Read config from a file. Overrides most other arguments.'
        },
    },
    'difference_layers': {
        'args': ('-d', '--difference_layers',),
        'kwargs': {
            'type': int,
            'action': 'store',
            'required': False,
            'help': 'The number of noise spaces to difference.'
        },
    },
    'frequency': {
        'args': ('-f', '--frequency',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'required': False,
            'help': 'The starting frequency for octave noise generation.'
        },
    },
    'filters': {
        'args': ('-F', '--filters',),
        'kwargs': {
            'type': str,
            'action': 'store',
            'required': False,
            'help': 'Filters for difference layers.'
        },
    },
    'grain': {
        'args': ('-g', '--grain',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'required': False,
            'help': 'Apply gaussian noise over the image.'
        },
    },
    'colorize': {
        'args': ('-k', '--colorize',),
        'kwargs': {
            'type': str,
            'action': 'store',
            'default': '',
            'required': False,
            'help': 'Use the given color to colorize the noise.'
        },
    },
    'curve': {
        'args': ('-K', '--curve',),
        'kwargs': {
            'type': str,
            'action': 'store',
            'default': '',
            'required': False,
            'help': 'Perform an easing function on the colors.'
        },
    },
    'ntypes': {
        'args': ('-n', '--ntypes',),
        'kwargs': {
            'type': str,
            'nargs': '*',
            'action': 'store',
            'required': False,
            'help': 'The noise generators to use.'
        },
    },
    'output_file': {
        'args': ('-o', '--output_file',),
        'kwargs': {
            'type': str,
            'action': 'store',
            'help': 'The name for the output file.'
        },
    },
    'octaves': {
        'args': ('-O', '--octaves',),
        'kwargs': {
            'type': int,
            'action': 'store',
            'required': False,
            'help': 'The octaves of noise for octave noise generation.'
        },
    },
    'persistence': {
        'args': ('-p', '--persistence',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'required': False,
            'help': 'How the impact of each octave changes in octave noise generation.'
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
    'start': {
        'args': ('-t', '--start',),
        'kwargs': {
            'type': int,
            'nargs': '*',
            'action': 'store',
            'default': [],
            'help': 'The frame to start the animation at.'
        },
    },
    'unit': {
        'args': ('-u', '--unit',),
        'kwargs': {
            'type': int,
            'nargs': '*',
            'action': 'store',
            'help': 'The dimensions in pixels of a unit of noise.'
        },
    },
    'overlay': {
        'args': ('-V', '--overlay',),
        'kwargs': {
            'action': 'store_true',
            'required': False,
            'help': 'Overlay the image with itself to increase contrast.'
        },
    },
}


def make_config_from_arguments(args:argparse.Namespace) -> dict:
    """Convert the command line arguments into a dictionary used 
    to control the execution of the application.
    """
    config = {}
    if args.output_file:
        config['filename'] = args.output_file
        config['format'] = None
    if args.ntypes:
        config['ntypes'] = args.ntypes
    if args.size:
        config['size'] = args.size[::-1]
    if args.unit:
        config['unit'] = args.unit[::-1]
    if args.start:
        config['start'] = args.start[::-1]
    if args.difference_layers:
        config['difference_layers'] = args.difference_layers
    if args.octaves:
        config['octaves'] = args.octaves
    if args.persistence:
        config['persistence'] = args.persistence
    if args.amplitude:
        config['amplitude'] = args.amplitude
    if args.frequency:
        config['frequency'] = args.frequency
    if args.autocontrast:
        config['autocontrast'] = args.autocontrast
    if args.blur:
        config['blur'] = args.blur
    if args.colorize:
        config['colorize'] = COLOR[args.colorize]
    if args.curve:
        config['curve'] = args.curve
    if args.filters:
        config['filters'] = args.filters
    if args.grain:
        config['grain'] = args.grain
    if args.overlay:
        config['overlay'] = args.overlay
    if args.ntypes:
        config['noises'] = make_noises_from_arguments(config)
    return config


def make_noises_from_arguments(config:Mapping) -> List[noise.BaseNoise]:
    """Make serialized noises from the command line arguments."""
    result = []
    for ntype in config['ntypes']:
        if ntype == 'LineNoise':
            kwargs = {
                'type': ntype,
            }
        else:
            kwargs = {
                'type': ntype,
                'size': config['size'],
                'unit': config['unit'],
                'octaves': config['octaves'],
                'persistence': config['persistence'],
                'amplitude': config['amplitude'],
                'frequency': config['frequency'],
            }
        result.append(kwargs)
    
    while len(result) < config['difference_layers'] + 1:
        new_noise = deepcopy(result[0])
        result.append(new_noise)
    
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse arguments from the command line."""
    epilog = ('COLORIZE: AVAILABLE COLORS\n'
              'The colors available for the --colorize option are:\r\n'
              '\n')
    color_temp = '{:>4}\t{}, {}\n'
    for color in COLOR:
        if color in ['t', 'T', '']:
            continue
        epilog += color_temp.format(color, COLOR[color][0], COLOR[color][1])
    epilog += ' \n'
    
    # Read the command line arguments.
    p = argparse.ArgumentParser(
        prog='PJINOISE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate noise.',
        epilog=epilog
    )
    for option in OPTIONS:
        args = OPTIONS[option]['args']
        kwargs = OPTIONS[option]['kwargs']
        p.add_argument(*args, **kwargs)
    return p.parse_args()


if __name__ == '__main__':
    raise NotImplementedError