"""
io
~~

Input/output for the pjinoise module.
"""
import json
from typing import Mapping

from pjinoise import model as m
from pjinoise.__version__ import __version__


def load_conf(filename: str) -> m.Image:
    """Load a configuration file."""
    # Get the configuration from the file.
    with open(filename, 'r') as fh:
        conf_json = fh.read()
    conf = json.loads(conf_json)
    
    # Deserialize configuration based on the given version.
    if conf['Version'] == '0.2.0':
        return m.Image(**conf['Image'])
    
    # Otherwise, the version isn't recognized, so throw an error.
    else:
        raise ValueError(f'Version {conf["Version"]} not supported.')


def save_conf(conf: m.Image) -> None:
    """Save the configuration file."""
    # Determine the name of the config file.
    parts = conf.filename.split('.')[:-1]
    parts.append('json')
    conffile = '.'.join(parts)
    
    # Serialize the config.
    confmap = {
        'Version': __version__,
        'Image': conf.asdict(),
    }
    confjson = json.dumps(confmap, indent=4)
    
    # Save the config.
    with open(conffile, 'w') as fh:
        fh.write(confjson)