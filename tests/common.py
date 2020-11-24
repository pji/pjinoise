"""
common
~~~~~~

Common functions used across multiple tests.
"""
from typing import Mapping


def map_compare(a, b):
    """Compare two mappings and return the first difference."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        ab = set(a) - set(b)
        ba = set(b) - set(a)
        if ab:
            return f'b missing keys: {ab}'
        if ba:
            return f'a missing keys: {ba}'
        for k in a:
            result = map_compare(a[k], b[k])
            if result is False:
                return f'{k} {a[k]} {b[k]}'
            if result is not True:
                return f'{k}/{result}'
        return True
    return a == b
