"""
pjinoise
~~~~~~~~

Core image generation and mainline for the pjinoise module.
"""
from typing import Sequence

from pjinoise import filters as f


def render_source(source: 'pjinoise.sources.ValueSource',
                  size: Sequence[int],
                  location: Sequence[int] = (0, 0, 0),
                  filters: Sequence[f.ForLayer] = None) -> 'numpy.ndarray':
    """Create image data from a ValueSource."""
    # You don't want to set the default value for a parameter to a
    # mutable value because it will remember any changes to it leading
    # to unexpected behavior.
    if filters is None:
        filters = []

    # Pad the image size and adjust the location so that the padding
    # doesn't change where the image data is generaterated within the
    # source.
    new_size = f.preprocess(size, filters)
    if new_size != size:
        padding = [n - o // 2 for n, o in zip(new_size, size)]
        location = [l + p for l, p in zip(location, padding)]

    # Generate the image data from the source.
    a = source.fill(new_size, location)

    # Apply any filters and remove any padding before returning the
    # image data.
    a = f.process(a, filters)
    a = f.postprocess(a, filters)
    return a