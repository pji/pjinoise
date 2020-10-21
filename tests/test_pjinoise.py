"""
test_pjinoise
~~~~~~~~~~~~~

Unit tests for pjinoise.pjinoise.
"""
from copy import deepcopy
import json
import unittest as ut
from unittest.mock import call, mock_open, patch
import sys

import numpy as np
from PIL import Image

from pjinoise import filters
from pjinoise import noise
from pjinoise import pjinoise as pn


CONFIG = {
    # General script configuration.
    'filename': 'spam.tiff',
    'format': 'TIFF',
    'save_config': True,
    'difference_layers': 6,
    
    # General noise generation configuration.
    'ntypes': [noise.ValueNoise,],
    'size': [3, 3],
    'unit': [2, 2, 2],
    'start': [3,],
    
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
    'colorize': '',
    'blur': None,
    'filters': 'rotate90_2:1_r+skew_3:1_10+skew_3:2_-10',
    'grain': 2.0,
    'overlay': False,
}
CONFIG['noises'] = [
    CONFIG['ntypes'][0](unit=CONFIG['unit'], table=[0 for _ in range(512)]),
    CONFIG['ntypes'][0](unit=CONFIG['unit'], table=[0 for _ in range(512)]),
    CONFIG['ntypes'][0](unit=CONFIG['unit'], table=[0 for _ in range(512)]),
    CONFIG['ntypes'][0](unit=CONFIG['unit'], table=[0 for _ in range(512)]),
    CONFIG['ntypes'][0](unit=CONFIG['unit'], table=[0 for _ in range(512)]),
    CONFIG['ntypes'][0](unit=CONFIG['unit'], table=[0 for _ in range(512)]),
    CONFIG['ntypes'][0](unit=CONFIG['unit'], table=[0 for _ in range(512)]),
]
FILTERS = [
    [],
    [
        filters.Rotate90('r'),
        filters.Skew(10),
    ],
    [
        filters.Skew(-10),
    ],
    [
        filters.Rotate90('r'),
    ],
    [
        filters.Skew(10),
    ],
    [
        filters.Rotate90('r'),
        filters.Skew(-10),
    ],
    [],
]


class CLITestCase(ut.TestCase):
    def test_configure_from_command_line(self):
        """When the script is invoked with command line arguments, 
        pjinoise.configure should update the script configuration 
        based on those arguments.
        """
        CONFIG_backup = deepcopy(pn.CONFIG)
        
        try:
            exp = CONFIG
            exp_filters = FILTERS
        
            sys.argv = [
                'python3.8 -m pjinoise.pjinoise', 
                '-g',
                str(exp['grain']),
                '-n',
                'ValueNoise',
                '-s',
                str(exp['size'][0]),
                str(exp['size'][1]),
                '-u',
                str(exp['unit'][0]),
                str(exp['unit'][1]),
                str(exp['unit'][2]),
                '-O',
                str(exp['octaves']),
                '-p',
                str(exp['persistence']),
                '-a',
                str(exp['amplitude']),
                '-f',
                str(exp['frequency']),
                '-F',
                exp['filters'],
                '-d',
                str(exp['difference_layers']),            
                '-t',
                str(exp['start'][0]),
                '-o',
                exp['filename'],
            ]
            pn.configure()
            for i in range(len(CONFIG['noises'])):
                pn.CONFIG['noises'][i].table = np.array([0 for _ in range(512)])
            act = pn.CONFIG
            act_filters = pn.FILTERS
        
            for key in act:
                try:
                    self.assertEqual(exp[key], act[key])
                except ValueError as e:
                    print(key)
                    raise e
                except AssertionError as e:
                    print(key)
                    raise e
            self.assertListEqual(exp_filters, act_filters)
        
        finally:
            pn.CONFIG = CONFIG_backup
            pn.IFILTERS = []
    
    def test_parse_filter_command(self):
        """Given a string containing a filter command and a number 
        of difference layers, pjinoise.parse_filter_command should 
        construct a list of tuples containing the filters and their 
        arguments in the proper periods.
        """
        CONFIG_backup = deepcopy(pn.CONFIG)
        
        try:
            exp = FILTERS
        
            cmd = CONFIG['filters']
            layers = 6
            act = pn.parse_filter_command(cmd, layers)
        
            self.maxDiff = None
            self.assertListEqual(exp, act)
        
        finally:
            pn.CONFIG = deepcopy(CONFIG_backup)
            pn.IFILTERS = []


class FileTestCase(ut.TestCase):
    def test_read_configuration_file(self):
        """When given a filename, pjinoise.read_config should 
        configure the script using the details in the given file.
        """
        CONFIG_backup = deepcopy(pn.CONFIG)
        
        try:
            namepart = CONFIG["filename"].split(".")[0]
            filename = f'{namepart}.conf'
            exp_conf = deepcopy(CONFIG)
            exp_conf['ntypes'] = [cls.__name__ for cls in exp_conf['ntypes']]
            exp_conf['noises'] = [n.asdict() for n in exp_conf['noises']]
            exp_open = (filename,)
        
            conf = deepcopy(exp_conf)
            contents = json.dumps(conf, indent=4)
            open_mock = mock_open()
            with patch("pjinoise.pjinoise.open", open_mock, create=True):
                open_mock.return_value.read.return_value = contents
                act_conf = pn.read_config(filename)
            
            open_mock.assert_called_with(*exp_open)
            for key in exp_conf:
                self.assertEqual(exp_conf[key], act_conf[key])
        
        finally:
            pn.CONFIG = deepcopy(CONFIG_backup)
            pn.IFILTERS = []
    
    def test_override_configuration_file(self):
        """If an argument is passed by the command line, it should 
        override the value that is in the configuration file.
        """
        # Save the original state.
        CONFIG_backup = deepcopy(pn.CONFIG)
        
        # Prepare and run the test.
        try:
            # Build the expected values.
            namepart = CONFIG["filename"].split(".")[0]
            filename = f'{namepart}.conf'
            exp_conf = deepcopy(CONFIG)
            exp_open = (filename,)
            
            # Build the input and state for the test.
            conf = deepcopy(exp_conf)
            conf['grain'] = exp_conf['grain'] + 2
            conf['ntypes'] = [cls.__name__ for cls in exp_conf['ntypes']]
            conf['noises'] = [n.asdict() for n in exp_conf['noises']]
            contents = json.dumps(conf, indent=4)
            sys.argv = [
                'python3.8 -m pjinoise.pjinoise', 
                '-g',
                str(exp_conf['grain']),
                '-C',
                filename,
                '-o',
                exp_conf['filename'],
            ]
            open_mock = mock_open()
            with patch("pjinoise.pjinoise.open", open_mock, create=True):
                open_mock.return_value.read.return_value = contents
                
                # Run the test.
                pn.configure()
                
            # Extract the actual values from the output or state.
            act_conf = pn.CONFIG
            
            # Determine the success of the test.
            open_mock.assert_called_with(*exp_open)
            for key in exp_conf:
                self.assertEqual(exp_conf[key], act_conf[key])
        
        # Restore the original state after the test, even in the case 
        # of an exception.
        finally:
            pn.CONFIG = deepcopy(CONFIG_backup)
            pn.IFILTERS = []
    
    def test_override_noise_configuration(self):
        """If an argument is passed by the command line, it should 
        override the value that is each of the noises serialized in 
        the configuration file.
        """
        # Save the original state.
        CONFIG_backup = deepcopy(pn.CONFIG)
        
        # Prepare and run the test.
        try:
            # Build the expected values.
            namepart = CONFIG["filename"].split(".")[0]
            filename = f'{namepart}.conf'
            exp_unit = [32, 256, 256]
            exp_open = (filename,)
            
            # Build the input and state for the test.
            conf = deepcopy(CONFIG)
            conf['ntypes'] = [cls.__name__ for cls in conf['ntypes']]
            conf['noises'] = [n.asdict() for n in conf['noises']]
            contents = json.dumps(conf, indent=4)
            sys.argv = [
                'python3.8 -m pjinoise.pjinoise', 
                '-u',
                str(exp_unit[2]),
                str(exp_unit[1]),
                str(exp_unit[0]),
                '-C',
                filename,
                '-o',
                conf['filename'],
            ]
            open_mock = mock_open()
            with patch("pjinoise.pjinoise.open", open_mock, create=True):
                open_mock.return_value.read.return_value = contents
                
                # Run the test.
                pn.configure()
                
            # Extract the actual values from the output or state.
            act_units = [n.unit for n in pn.CONFIG['noises']]
            
            # Determine the success of the test.
            open_mock.assert_called_with(*exp_open)
            for act_unit in act_units:
                self.assertListEqual(exp_unit, act_unit)
        
        # Restore the original state after the test, even in the case 
        # of an exception.
        finally:
            pn.CONFIG = deepcopy(CONFIG_backup)
            pn.IFILTERS = []
    
    def test_save_configuration_file(self):
        """When called, pjinoise.save_config should write the 
        current configuration to a file.
        """
        namepart = CONFIG["filename"].split(".")[0]
        filename = f'{namepart}.conf'
        exp_conf = deepcopy(CONFIG)
        CONFIG_backup = deepcopy(pn.CONFIG)
        pn.CONFIG = deepcopy(CONFIG)
        exp_conf['ntypes'] = [cls.__name__ for cls in exp_conf['ntypes']]
        exp_conf['noises'] = [n.asdict() for n in exp_conf['noises']]
        exp_open = (filename, 'w')
        
        open_mock = mock_open()
        with patch('pjinoise.pjinoise.open', open_mock, create=True):
            pn.save_config()
        pn.CONFIG = deepcopy(CONFIG_backup)
        
        open_mock.assert_called_with(*exp_open)
        
        text = open_mock.return_value.write.call_args[0][0]
        act_conf = json.loads(text)
        for key in exp_conf:
            self.assertEqual(exp_conf[key], act_conf[key])
        
    @patch('PIL.Image.Image.save')
    @patch('PIL.Image.fromarray')
    def test_save_image(self, mock_fromarray, mock_save):
        """Given a two dimensional numpy.array, pjinoise.save_image 
        should create PIL.Image for the array and save it to disk.
        """
        # Save the original state.
        CONFIG_backup = deepcopy(pn.CONFIG)
        
        # Run the test.
        try:
            # Build the expected values.
            array = np.array([[0, 127, 255], [0, 127, 255]])
            filename = 'spam'
            format = 'TIFF'
            exp = [
                ['', [array.tolist(),], {'mode': 'L',}], 
                ['().save', [filename, format], {}],
            ]
            
            # Build input and state for the test.
            pn.CONFIG['filename'] = filename
            pn.CONFIG['format'] = format
            pn.CONFIG['grain'] = None
            
            # Run the test.
            pn.save_image(array)
            
            # Extract the actual values from the output.
            calls = mock_fromarray.mock_calls
            act = [list(item) for item in calls]
            for item in act:
                item[1] = [thing for thing in item[1]]
            act[0][1][0] = act[0][1][0].tolist()
            
            # Determine if the test succeeded.
            self.assertListEqual(exp, act)
            
        # Restore the original state after the test, even in the case 
        # of an exception.
        finally:
            pn.CONFIG = deepcopy(CONFIG_backup)
    
    @patch('PIL.Image.Image.save')
    @patch('PIL.Image.fromarray')
    def test_save_animation(self, mock_fromarray, mock_save):
        """Given a three dimensional numpy.array, pjinoise.save_image 
        should create PIL.Image that is an animation, with each two 
        dimensional slice being a frame, and save it to disk.
        """
        # Save the original state.
        CONFIG_backup = deepcopy(pn.CONFIG)
        
        # Run the test.
        try:
            # Build the expected values.
            array = np.array([
                [
                    [0, 127, 255], 
                    [0, 127, 255]
                ],
                [
                    [0, 127, 255], 
                    [0, 127, 255]
                ],
            ]
            )
            filename = 'spam'
            format = 'GIF'
            loop = 0
            img = Image.fromarray(array[1])
            exp = [
                ['', [array.tolist()[0],], {}],     # Artifact from two lines up.
                ['', [array.tolist()[0],], {'mode': 'L',}], 
                ['', [array.tolist()[1],], {'mode': 'L',}], 
                [
                    '().save', 
                    [filename], 
                    {
                        'save_all': True,
                        'append_images': [img,],
                        'loop': loop,
                        'duration': (1 / pn.CONFIG['framerate']) * 1000,
                    }
                ],
            ]
        
            # Build input and state for the test.
            pn.CONFIG['filename'] = filename
            pn.CONFIG['format'] = format
            pn.CONFIG['loop'] = 0
            pn.CONFIG['grain'] = None
            
            # Run the test.
            pn.save_image(array)
            
            # Extract the actual values from the output.
            calls = mock_fromarray.mock_calls
            act = [list(item) for item in calls]
            for item in act:
                item[1] = [thing for thing in item[1]]
            act[0][1][0] = act[0][1][0].tolist()
            act[1][1][0] = act[1][1][0].tolist()
            act[2][1][0] = act[2][1][0].tolist()
            
            # Determine if the test succeeded.
            self.assertListEqual(exp, act)
        
        # Restore the original state after the test, even in the case 
        # of an exception.
        finally:
            pn.CONFIG = deepcopy(CONFIG_backup)
            pn.IFILTERS = []


class NoiseTestCase(ut.TestCase):
    """Test cases for the creation of noise."""
    def test_create_single_noise_volume(self):
        """Given a noise object and a size of the noise to generate, 
        pjinoise.make_noise should return a numpy.ndarray object 
        containing the noise.
        """
        CONFIG_backup = deepcopy(pn.CONFIG)
        
        try:
            exp = [
                [0.0, 63.5, 127.0, 191.0, 255.0],
                [0.0, 63.5, 127.0, 191.0, 255.0],
                [0.0, 63.5, 127.0, 191.0, 255.0],
            ]
        
            size = (3, 5)
            table = [
                    [
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                    ],
                    [
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                    ],
                    [
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                    ],
                ]
            unit = (2, 2, 2)
            obj = noise.GradientNoise(table=table, unit=unit)
            array = pn.make_noise(obj, size)
            act = array.tolist()
        
            self.assertListEqual(exp, act)
        
        finally:
            pn.CONFIG = deepcopy(CONFIG_backup)
            pn.IFILTERS = []


    def test_create_difference_noise_volume(self):
        """Given a sequence of noise objects and a size of the 
        noise to generate, pjinoise.make_difference_noise should 
        return a numpy.ndarray object containing noise that is 
        the difference of the noise generated from each object.
        """
        CONFIG_backup = deepcopy(pn.CONFIG)
        
        try:
            exp = [[
                [127.0, 63.5, 0.0, 64.0, 128.0],
                [127.0, 63.5, 0.0, 64.0, 128.0],
                [127.0, 63.5, 0.0, 64.0, 128.0],
            ],]
        
            size = (1, 3, 5)
            table1 = [
                    [
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                    ],
                    [
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                    ],
                    [
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                        [0, 127, 255, 255],
                    ],
                ]
            table2 = [
                    [
                        [127, 127, 127, 255],
                        [127, 127, 127, 255],
                        [127, 127, 127, 255],
                        [127, 127, 127, 255],
                    ],
                    [
                        [127, 127, 127, 255],
                        [127, 127, 127, 255],
                        [127, 127, 127, 255],
                        [127, 127, 127, 255],
                    ],
                    [
                        [127, 127, 127, 255],
                        [127, 127, 127, 255],
                        [127, 127, 127, 255],
                        [127, 127, 127, 255],
                    ],
                ]
            unit = (2, 2, 2)
            objs = [
                noise.GradientNoise(table=table1, unit=unit),
                noise.GradientNoise(table=table2, unit=unit),
            ]
            pn.FILTERS = [[] for _ in  range(pn.CONFIG['difference_layers'])]
            array = pn.make_difference_noise(objs, size)
            act = array.tolist()
        
            self.assertListEqual(exp, act)
        
        finally:
            pn.CONFIG = deepcopy(CONFIG_backup)
            pn.IFILTERS = []


if __name__ == '__main__':
    raise NotImplementedError