"""
common
~~~~~~

Utilities and other commonly reused functions for pjinoise.
"""
from typing import Sequence, Tuple, Union


def deserialize_sequence(value:Union[Sequence[float], str]) -> Tuple[float]:
    """Deserialize a set of coordinates that could have come from 
    command line input.
    """
    if not value:
        return (0, 0, 0)
    if isinstance(value, str):
        value = value.split(',')[::-1]
    return tuple(float(n) for n in value)
