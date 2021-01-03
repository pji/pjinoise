#! .venv/bin/python
"""
mazer
~~~~~

Create a printable maze using the pjinoise module.
"""
import argparse

from pjinoise import filters as f
from pjinoise import model as m
from pjinoise import operations as op
from pjinoise import pjinoise as pn
from pjinoise import sources as s


def main(seed=None, origin=(0, 0, 0)):
    # Make sure the version of pjinoise supports mazer.
    assert pn.__version__ == '0.3.1'

    # Set up the size and structure of the maze.
    size = (1, 720, 560)
    units = (1, 20, 20)
    title = seed.replace('_', ' ').upper()

    # The maze interior.
    path = m.Layer(**{
        'filters': [],
        'source': s.Path(width=.4, origin=origin, unit=units, seed=seed),
        'blend': op.replace,
    })

    # The maze entrance.
    entrance = m.Layer(**{
        'source': s.Box((0, 12, 0), (1, 16, 20), 1.0),
        'blend': op.lighter,
    })

    # The maze exit.
    exit = m.Layer(**{
        'source': s.Box((0, 692, 540), (1, 16, 20), 1.0),
        'blend': op.lighter,
    })

    title = m.Layer(**{
        'source': s.Text(title, origin=(12, 1), font='Helvetica', face=1),
        'blend': op.lighter,
    })

    # Put it all together and you get the maze.
    maze = m.Layer(**{
        'source': [path, entrance, exit, title],
        'blend': op.replace,
    })

    # Image output configuration.
    conf = m.Image(**{
        'source': maze,
        'size': size,
        'filename': f'maze_{seed}_{origin}.png',
        'format': 'PNG',
        'mode': 'L',
    })

    # Create image.
    pn.main(False, conf)


if __name__ == '__main__':
    # Define the command line options.
    options = {
        'origin': {
            'args': ('-o', '--origin'),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': 'tl',
                'help': 'The seed used to generate the maze.'
            },
        },
        'seed': {
            'args': ('seed',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'default': None,
                'help': 'The seed used to generate the maze.'
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
    main(args.seed, args.origin)
