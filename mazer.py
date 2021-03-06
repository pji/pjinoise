#! .venv/bin/python
"""
mazer
~~~~~

Create a printable maze using the pjinoise module.
"""
import argparse
from datetime import date
from os import path

from pjinoise import filters as f
from pjinoise import model as m
from pjinoise import operations as op
from pjinoise import pjinoise as pn
from pjinoise import sources as s
from pjinoise.__version__ import __version__
from pjinoise.constants import X, Y


def get_today(delim='.'):
    today = date.today()
    year, month, day, *_ = today.timetuple()
    return f'{year}{delim}{month:02d}{delim}{day:02d}'


def main(seed=None, origin=(0, 0, 0), solve=False, save_dir='', unit=20):
    # Make sure the version of pjinoise supports mazer.
    assert __version__ == '0.3.1'

    # Set up the size and structure of the maze.
    size = (1, 720, 560)
    units = (1, unit, unit)
    exit_size = (1, int(unit * .8), unit)
    title = seed.replace('_', ' ').upper()

    # The maze interior.
    path = m.Layer(**{
        'filters': [],
        'source': s.Path(width=.4, origin=origin, unit=units, seed=seed),
        'blend': op.replace,
    })

    # The solution.
    sol = m.Layer(**{
        'source': s.Solid(1),
        'filters': [
            f.Color('s'),
        ],
        'mask': s.SolvedPath(width=.1, origin=origin, unit=units, seed=seed),
        'blend': op.replace,
    })

    # The maze entrance.
    entrance = m.Layer(**{
        'source': s.Box((0, int(unit * .6), 0), exit_size, 1.0),
        'blend': op.lighter,
    })

    # The maze exit.
    exit = m.Layer(**{
        'source': s.Box(*[
            (0, int(size[Y] - unit * 1.4), size[X] - unit),
            exit_size,
            1.0
        ]),
        'blend': op.lighter,
    })

    if unit >= 16:
        ft_origin = (int(unit * .6), 1)
        ft_size = int(unit * .5)
        title = m.Layer(**{
            'source': s.Text(
                title,
                size=ft_size,
                origin=ft_origin,
                font='Helvetica',
                face=1
            ),
            'blend': op.lighter,
        })

    # Put it all together and you get the maze.
    layers = [path, entrance, exit,]
    if unit >= 16:
        layers.append(title)
    if solve:
        layers.append(sol)
    maze = m.Layer(**{
        'source': layers,
        'blend': op.replace,
    })

    # Image output configuration.
    mode = 'L'
    name = f'{save_dir}maze_{seed}_{origin}'
    if unit != 20:
        name = name + f'_{unit}'
    if solve:
        mode = 'RGB'
        name = name + '_solved'
    name = f'{name}.png'
    conf = m.Image(**{
        'source': maze,
        'size': size,
        'filename': name,
        'format': 'PNG',
        'mode': mode,
    })

    # Create image.
    pn.main(False, conf)


if __name__ == '__main__':
    # Define the command line options.
    options = {
        'middle': {
            'args': ('-m', '--middle',),
            'kwargs': {
                'action': 'store_true',
                'help': 'Start generation in the middle of the maze.'
            },
        },
        'origin': {
            'args': ('-o', '--origin'),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': 'tl',
                'help': 'Where in the maze generation should start.'
            },
        },
        'save_dir': {
            'args': ('-d', '--save_dir',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': '',
                'help': 'Where to save the image file.'
            },
        },
        'seed': {
            'args': ('-s', '--seed',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': None,
                'help': 'The seed used to generate the maze.'
            },
        },
        'solve': {
            'args': ('-S', '--solve',),
            'kwargs': {
                'action': 'store_true',
                'help': 'Add the solution to the maze.'
            },
        },
        'today': {
            'args': ('-t', '--today',),
            'kwargs': {
                'action': 'store_true',
                'help': 'Use the current date for the seed.'
            },
        },
        'unit': {
            'args': ('-u', '--unit',),
            'kwargs': {
                'type': int,
                'action': 'store',
                'default': 20,
                'help': 'The unit side for the maze grid.'
            },
        },
    }

    # Set up the argument parser.
    p = argparse.ArgumentParser(
        prog='MAZER',
        description='Create a printable maze.',
    )
    for option in options:
        args = options[option]['args']
        kwargs = options[option]['kwargs']
        p.add_argument(*args, **kwargs)

    # Parse the command line arguments.
    args = p.parse_args()
    if args.today or args.seed is None:
        args.seed = get_today()
    if args.middle:
        args.origin = 'm'
    if args.save_dir and not args.save_dir.endswith(path.sep):
        args.save_dir = f'{args.save_dir}{path.sep}'
    main(args.seed, args.origin, args.solve, args.save_dir, args.unit)
