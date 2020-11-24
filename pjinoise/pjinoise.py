"""
pjinoise
~~~~~~~~

Core image generation and mainline for the pjinoise module.
"""
from typing import Sequence


def render_source(source: 'pjinoise.sources.ValueSource',
                  size: Sequence[int], 
                  location: Sequence[int] = (0, 0, 0)) -> 'numpy.ndarray':
    """Create image data from a ValueSource."""
    return source.fill(size, location)