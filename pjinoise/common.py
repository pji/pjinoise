"""
common
~~~~~~

Utilities and other commonly reused functions for pjinoise.
"""
from typing import List, Sequence, Tuple, Union

import numpy as np

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


def grayscale_to_ints_list(a: np.ndarray) -> List[int]:
    """pjinoise grayscale stores color values as floats between 
    zero and one. This is a pain to read on a screen or type 
    expected values for. This function converts that to lists 
    of integers for easier test comparison and printing.
    """
    a = a.copy()
    a = np.around(a * 0xff)
    a = a.astype(np.uint8)
    return a.tolist()


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
