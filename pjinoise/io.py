"""
io
~~

Input/output for the pjinoise module.
"""
import json
from typing import Mapping

from pjinoise import model as m


def load_conf(filename: str) -> m.Image:
    """Load a configuration file."""
    # Get the configuration from the file.
    with open(filename, 'r') as fh:
        conf_json = fh.read()
    conf = json.loads(conf_json)
    
    # Deserialize configuration based on the given version.
    if conf['Version'] == '0.2.0':
        return m.Image(**conf['Image'])
    
    else:
        raise ValueError(f'Version {conf["Version"]} not supported.')