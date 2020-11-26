"""
common
~~~~~~

Utilities and other commonly reused functions for pjinoise.
"""
from typing import Sequence, Tuple, Union

from pjinoise.constants import SUPPORTED_FORMATS


def deserialize_sequence(value:Union[Sequence[float], str]) -> Tuple[float]:
    """Deserialize a set of coordinates that could have come from 
    command line input.
    """
    if not value:
        return (0, 0, 0)
    if isinstance(value, str):
        value = value.split(',')[::-1]
    return tuple(float(n) for n in value)


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
