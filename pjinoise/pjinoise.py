"""
pjinoise
~~~~~~~~

Core image generation and mainline for the pjinoise module.
"""
from queue import Queue
from threading import Thread
from typing import Sequence, Tuple, Union

import numpy as np
from PIL import Image

from pjinoise import cli
from pjinoise import filters as f
from pjinoise import io
from pjinoise import ui
from pjinoise.common import convert_color_space as _convert_color_space
from pjinoise.model import Layer
from pjinoise.sources import ValueSource


# Constants.
X, Y, Z = 2, 1, 0


# Image generation functions.
# def _convert_color_space(a: np.ndarray,
#                          src_space: str = '',
#                          dst_space: str = 'RGB') -> np.ndarray:
#     """Convert an array to the given color space."""
#     # The shape of the output is based on the space, so we can't
#     # build out until we do the first conversion. However, setting
#     # it to None here makes the process of detecting whether we've
#     # set up the output array a little smoother later.
#     out = None
# 
#     # Most of pjinoise tries to work with grayscale color values
#     # that go from zero to one. However, pillow's grayscale mode
#     # is 'L', which represents the color as an unsigned 8 bit
#     # integer. The data will need to at least be in mode 'L' for
#     # pillow to be able to convert the color space.
#     if src_space == '':
#         a = (a * 0xff).astype(np.uint8)
#         src_space = 'L'
# 
#     # PIL.image.convert can only convert two-dimensional (or three,
#     # with color channel being the third) images. So, for animations
#     # we have to iterate through the Z axis, coverting one frame at
#     # a time. Since pjinoise thinks of still images as single frame
#     # animations, this means we're always going to have to handle
#     # the Z axis like this.
#     for i in range(a.shape[Z]):
#         img = Image.fromarray(a[i], mode=src_space)
#         img = img.convert(dst_space)
#         a_img = np.array(img)
#         if out is None:
#             out = np.zeros((a.shape[Z], *a_img.shape), dtype=a.dtype)
#         out[i] = a_img
#     return out
# 
# 
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

    # If we got a source layer, process it.
    if isinstance(layers.source, ValueSource):
        kwargs = {
            'source': layers.source,
            'size': size,
            'location': layers.location,
            'filters': layers.filters,
        }
        b = render_source(**kwargs)

    # Otherwise we got a container layer, process its source and run
    # any filters that are set on the layer.
    else:
        new_size = f.preprocess(size, layers.filters)
        b = process_layers(size, layers.source)
        b = f.process(b, layers.filters)
        b = f.postprocess(b, layers.filters)

    # There are two possibilities for how the layers should be
    # blended: masked or unmasked. Masked blends will have a
    # ValueSource in the mask attribute, which needs to be sent
    # to the blending operation.
    if layers.mask is not None:
        kwargs = {
            'source': layers.mask,
            'size': size,
            'filters': layers.mask_filters,
        }
        mask = render_source(**kwargs)
        a, b, mask = _normalize_color_space(a, b, mask)
        return layers.blend(a, b, layers.blend_amount, mask)

    # Unmasked layers have no mask, so Blend the image data and return.
    a, b = _normalize_color_space(a, b)
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
        location = [loc + p for loc, p in zip(location, padding)]

    # Generate the image data from the source.
    try:
        a = source.fill(new_size, location)
        assert np.max(a) <= 1.0
    except AttributeError as e:
        print(source)
        raise e

    # Apply any filters and remove any padding before returning the
    # image data.
    a = f.process(a, filters)
    a = f.postprocess(a, filters)
    return a


# Mainline.
def main(silent=True):
    """Mainline."""
    try:
        status = None
        args = cli.parse_cli_args()
        if args.load_config:
            conf = io.load_conf(args.load_config, args)
        else:
            conf = cli.build_config(args)
        if not silent:
            stages = 2
            status = Queue()
            t = Thread(target=ui.status_writer, args=(status, stages))
            t.start()
            status.put((ui.INIT,))

        if not silent:
            status.put((ui.STATUS, 'Generating image...'))
        a = process_layers(conf.size, conf.source)
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
    # exceptions bubble up from this thread clobbers causes the last
    # status updates to clobber the last few lines of the exception.
    # To avoid that, send the exception through the status update
    # thread. This also ensures the status update thread is terminated.
    except Exception as e:
        if status:
            status.put((ui.KILL, e))
        else:
            raise e


if __name__ == '__main__':
    main(False)
