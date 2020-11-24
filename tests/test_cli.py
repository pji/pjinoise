"""
test_cli
~~~~~~~~

Unit tests for the CLI for the pjinoise module.
"""
import sys
import unittest as ut
from typing import Mapping

from pjinoise import cli
from pjinoise import model as m
from pjinoise import operations as op
from pjinoise import sources as s
from tests.common import map_compare


class CLITestCase(ut.TestCase):
    def test_create_single_layer_image(self):
        """Given the proper CLI options, cli.build_config should 
        return the config as a model.Image object.
        """
        # Back up initial state.
        argv_bkp = sys.argv
        try:
            
            # Set up data for expected values.
            format = 'JPEG'
            filename = 'spam.json'
            framerate = None
            imagefile = 'spam.jpeg'
            location = [5, 0, 0]
            mode = 'L'
            size = [1, 1280, 720]
            
            # Expected values.
            exp = m.Image(**{
                'source': m.Layer(**{
                    'source': s.Spot(**{
                        'radius': 128,
                        'ease': 'l',
                    }),
                    'location': location,
                    'filters': [],
                    'mask': None,
                    'mask_filters': [],
                    'blend': op.difference,
                    'blend_amount': .5,
                }),
                'size': size,
                'filename': imagefile,
                'format': format,
                'mode': mode,
                'framerate': None
            })
            
            # Set up test data and state.
            sys.argv = [
                'python3.8 -m pjinoise.core',
                '-s', str(size[-1]), str(size[-2]), str(size[-3]),
                '-n', 'spot_128:l_0:0:5__difference:.5',
                '-o', imagefile,
                '-m', mode,
            ]
            
            # Run tests.
            args = cli.parse_cli_args()
            act = cli.build_config(args)
            
            # Determine if test passed.
            try:
                self.assertEqual(exp, act)
            except AssertionError as e:
                exp = exp.asdict()
                act = act.asdict()
                result = map_compare(exp, act)
                if result is not True:
                    msg = f'Path to bad key(s): {result}'
                    raise ValueError(msg)
                else:
                    raise e
        
        # Restore initial state.
        finally:
            sys.argv = argv_bkp
