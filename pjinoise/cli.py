"""
cli
~~~

The command line interface for the pjinoise module.
"""
import argparse

from pjinoise.constants import COLOR

# Command line option configuration.
OPTIONS = {
    'amplitude': {
        'args': ('-a', '--amplitude',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'required': False,
            'default': 24,
            'help': 'The starting amplitude for octave noise generation.'
        },
    },
    'autocontrast': {
        'args': ('-A', '--autocontrast',),
        'kwargs': {
            'action': 'store_true',
            'required': False,
            'default': False,
            'help': 'Automatically adjust the contrast of the image.'
        },
    },
    'blur': {
        'args': ('-b', '--blur',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'required': False,
            'default': None,
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
            'default': 0,
            'help': 'The number of noise spaces to difference.'
        },
    },
    'frequency': {
        'args': ('-f', '--frequency',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'required': False,
            'default': 4,
            'help': 'The starting frequency for octave noise generation.'
        },
    },
    'filters': {
        'args': ('-F', '--filters',),
        'kwargs': {
            'type': str,
            'action': 'store',
            'required': False,
            'default': '',
            'help': 'Filters for difference layers.'
        },
    },
    'grain': {
        'args': ('-g', '--grain',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'default': None,
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
    'ntypes': {
        'args': ('-n', '--ntypes',),
        'kwargs': {
            'type': str,
            'nargs': '*',
            'action': 'store',
            'default': ['GradientNoise',],
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
            'default': 6,
            'help': 'The octaves of noise for octave noise generation.'
        },
    },
    'persistence': {
        'args': ('-p', '--persistence',),
        'kwargs': {
            'type': float,
            'action': 'store',
            'required': False,
            'default': -4,
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
            'default': False,
            'help': 'Overlay the image with itself.'
        },
    },
}


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