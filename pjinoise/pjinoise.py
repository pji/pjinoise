"""
pjinoise
~~~~~~~~

Core image generation and mainline for the pjinoise module.


Basic Usage
===========
The pjinoise.main() function drives the rendering of images and
animations made with objects from the pjinoise module. Calling
main() works like calling any other Python function. To create an
image, it needs to be passed configuration through a model.Image
object.

Usage::

    >>> from pjinoise import model as m
    >>> from pjinoise import operations as op
    >>> from pjinoise import pjinoise as pn
    >>> from pjinoise import sources as s
    >>>
    >>> src = s.Solid(.5)
    >>> layer = m.Layer(src, op.replace)
    >>> image = m.Image(layer, (1, 8, 8), '_test.jpg', 'JPEG', 'L')
    >>>
    >>> pn.main(False, image)
    PJINOISE: Pattern and Noise Generation
    ┌   ┐
    │███│
    └   ┘
    00:00:00 Image generated.
    00:00:00 Saving...
    00:00:01 Saved as _test.jpg.
    00:00:01 Good-bye.

"""
from queue import Queue
from threading import Thread
from typing import Sequence, Tuple, Union

import numpy as np

from pjinoise import cli
from pjinoise import filters as f
from pjinoise import io
from pjinoise import ui
from pjinoise.__version__ import __version__
from pjinoise.common import convert_color_space as _convert_color_space
from pjinoise.model import Image, Layer
from pjinoise.sources import Source


# Image generation functions.
def _normalize_color_space(*arrays) -> Tuple[np.ndarray]:
    """If one of the arrays is in RGB, convert both to RGB."""
    # Assuming the working spaces are either grayscale or RGB, if all
    # the two arrays are the same shape, they should be in the same
    # space.
    shapes = [len(a.shape) for a in arrays]
    if all(shape == shapes[0] for shape in shapes):
        return arrays

    # Grayscale has three dimensions. RGB has four. To preserve color
    # in blending operations, grayscale has to be converted to RGB.
    converted = []
    for a in arrays:
        if len(a.shape) == 3:
            assert np.max(a) <= 1.0
            a = _convert_color_space(a)
            assert a.dtype == np.uint8
        converted.append(a)
    return tuple(converted)


def process_layers(size: Sequence[int],
                   layers: Union[Source, Layer, Sequence[Layer]],
                   a: Union[None, np.ndarray] = None,
                   status: Union[None, Queue] = None) -> np.ndarray:
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
            a = process_layers(size, layer, a, status)
        return a

    # If we got a source layer, process it.
    if isinstance(layers.source, Source):
        if status is not None:
            src_name = layers.source.__class__.__name__
            status.put((ui.STATUS, f'Rendering {src_name}...'))
        kwargs = {
            'source': layers.source,
            'size': size,
            'location': layers.location,
            'filters': layers.filters,
        }
        b = render_source(**kwargs)
        if status is not None:
            src_name = layers.source.__class__.__name__
            status.put((ui.PROG, f'Rendered {src_name}.'))

    # Otherwise we got a container layer, process its source and run
    # any filters that are set on the layer.
    else:
        new_size = f.preprocess(size, layers.filters)
        b = process_layers(new_size, layers.source, None, status)
        b = f.process(b, layers.filters)
        b = f.postprocess(b, layers.filters)

    # There are two possibilities for how the layers should be
    # blended: masked or unmasked. Masked blends will have a
    # Source in the mask attribute, which needs to be sent
    # to the blending operation.
    if layers.mask is not None:
        if status is not None:
            src_name = layers.mask.__class__.__name__
            status.put((ui.STATUS, f'Rendering {src_name}...'))
        kwargs = {
            'source': layers.mask,
            'size': size,
            'filters': layers.mask_filters,
        }
        mask = render_source(**kwargs)
        if status is not None:
            src_name = layers.mask.__class__.__name__
            status.put((ui.PROG, f'Rendered {src_name}.'))
        a, b, mask = _normalize_color_space(a, b, mask)
        return layers.blend(a, b, layers.blend_amount, mask)

    # Unmasked layers have no mask, so Blend the image data and return.
    a, b = _normalize_color_space(a, b)
    return layers.blend(a, b, layers.blend_amount)


def render_source(source: Source,
                  size: Sequence[int],
                  location: Sequence[int] = (0, 0, 0),
                  filters: Sequence[f.Filter] = None) -> np.ndarray:
    """Create image data from a Source."""
    # You don't want to set the default value for a parameter to a
    # mutable value because it will remember any changes to it leading
    # to unexpected behavior.
    if filters is None:
        filters = []

    # Pad the image size and adjust the location so that the padding
    # doesn't change where the image data is generated within the
    # source.
    new_size = f.preprocess(size, filters)
    if new_size != size:
        padding = [n - o // 2 for n, o in zip(new_size, size)]
        location = [loc + p for loc, p in zip(location, padding)]

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


# Mainline.
def main(silent: bool = True, conf: Image = None) -> None:
    """Mainline."""
    status = None
    try:
        if not conf:
            args = cli.parse_cli_args()
            if args.load_config:
                conf = io.load_conf(args.load_config, args)
            else:
                conf = cli.build_config(args)
        if not silent:
            status = Queue()
            stages = 2 + conf.count_sources()
            t = Thread(target=ui.status_writer, args=(status, stages))
            t.start()
            status.put((ui.INIT,))

        if not silent:
            status.put((ui.STATUS, 'Generating image...'))
        a = process_layers(conf.size, conf.source, None, status)
        if not silent:
            status.put((ui.PROG, 'Image generated.'))

        if not silent:
            status.put((ui.STATUS, 'Saving...'))
        io.save_image(a, conf.filename, conf.format, conf.mode, conf.framerate)
        io.save_conf(conf)
        if not silent:
            status.put((ui.PROG, f'Saved as {conf.filename}.'))
            status.put((ui.END, 'Good-bye.'))

    # Since the status updates run in an independent thread, letting
    # exceptions bubble up from this thread causes the last status
    # updates to clobber the last few lines of the exception.
    # To avoid that, send the exception through the status update
    # thread. This also ensures the status update thread is terminated.
    except Exception as e:
        if status:
            status.put((ui.KILL, e))
        else:
            raise e


if __name__ == '__main__':
    main(False)
