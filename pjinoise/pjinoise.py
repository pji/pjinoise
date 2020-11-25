"""
pjinoise
~~~~~~~~

Core image generation and mainline for the pjinoise module.
"""
from typing import Sequence, Union

import numpy as np

from pjinoise import filters as f
from pjinoise.model import Layer
from pjinoise.sources import ValueSource


# Image generation functions.
def process_layers(size: Sequence[int], 
                   layers: Union[ValueSource, Layer, Sequence[Layer]],
                   a: Union[None, np.ndarray] = None) -> np.ndarray: 
    """Create image data from the layers."""
    # If no destination array was sent, it means we are either at the 
    # start of the layer processing or we are starting the processing 
    # of a layer group. If we are starting processing, then this will 
    # contain the final image data. If we are starting the processing 
    # of a layer group, this allows the layers in the group to only 
    # blend with the other layers in the group before they blend with 
    # the layers outside the group.
    if a is None:
        a = np.zeros(size, dtype=float)
    
    # If we got a sequence of layers, we process them recursively and 
    # return the result.
    if isinstance(layers, Sequence):
        for layer in layers:
            a = process_layers(size, layer, a)
        return a
    
    # If we got a source layer, return it.
    if isinstance(layers.source, ValueSource):
        kwargs = {
            "source": layers.source, 
            "size": size,
            "location": layers.location,
            "filters": layers.filters,
        }
        b = render_source(**kwargs)
        a = layers.blend(a, b, layers.blend_amount)
        return a
    
    # Otherwise we got a container layer, process its source and 
    # blend the result with the destination array.
    b = process_layers(size, layers.source)
    return layers.blend(a, b, layers.blend_amount)


def render_source(source: 'pjinoise.sources.ValueSource',
                  size: Sequence[int],
                  location: Sequence[int] = (0, 0, 0),
                  filters: Sequence[f.ForLayer] = None) -> np.ndarray: 
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
    try:
        a = source.fill(new_size, location)
    except AttributeError as e:
        print(source)
        raise e

    # Apply any filters and remove any padding before returning the
    # image data.
    a = f.process(a, filters)
    a = f.postprocess(a, filters)
    return a