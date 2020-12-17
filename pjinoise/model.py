"""
model
~~~~~

Data objects to used to construct an image.
"""
import typing as t

from pjinoise import filters as f
from pjinoise import operations as op
from pjinoise import sources as s
from pjinoise.base import Serializable


_Source = t.Union[s.ValueSource, 'Layer']
_SerializedSource = t.Union[t.Mapping, t.Sequence[t.Mapping]]


# Classes.
class Layer(Serializable):
    """A layer of image data to blend with other layers.

    :param source: The image source(s) used to create the image data
        of the layer.
    :param blend: The blending operation used to blend this layer into
        the image.
    :param blend_amount: (Optional.) How much this layer's blend should
        impact the overall image. It's equivalent to opacity. The value
        is a percentage in the range 0 <= x <= 1. The default value is
        one.
    :param location: (Optional.) The amount to offset the generation
        of image data from the layer's source(s). This is passed to
        the source(s) as the location parameter to their fill methods.
        The default value is no offset.
    :param filters: (Optional.) A list of filter objects to run on the
        image data generated by the source(s). The default value is
        no filters.
    :param mask: (Optional.) The image source(s) used to create an
        opacity mask for the layer. The default value results in a
        fully opaque mask.
    :param mask_filters: (Optional.) A list of filter objects to run
        on the image data of the opacity mask. The default value is
        no filters.
    :return: A :class:Layer object.
    :rtype: pjinoise.model.Layer

    Usage::

        >>> from pjinoise import operations as op
        >>> from pjinoise import sources as s
        >>>
        >>> layer = Layer(s.Solid(.5), op.replace)
        >>> layer                                       #doctest: +ELLIPSIS
        Layer(blend='replace'...mask=None)

    """
    def __init__(self,
                 source: t.Union[_Source,
                                 t.Sequence[_Source],
                                 _SerializedSource],
                 blend: t.Union[str, t.Callable],
                 blend_amount: float = 1,
                 location: t.Sequence[int] = (0, 0, 0),
                 filters: t.Sequence[f.Filter] = None,
                 mask: t.Union[None, s.ValueSource] = None,
                 mask_filters: t.Sequence[f.Filter] = None) -> None:
        self.source = source
        if isinstance(blend, str):
            blend = op.registered_ops[blend]
        self.blend = blend
        self.blend_amount = blend_amount
        self.location = location
        if not filters:
            filters = []
        elif isinstance(filters[0], t.Mapping):
            filters = [f.deserialize_filter(filter) for filter in filters]
        self.filters = filters
        self.mask = mask
        if not mask_filters:
            mask_filters = []
        elif isinstance(mask_filters[0], t.Mapping):
            mask_filters = [f.deserialize_filter(f_) for f_ in mask_filters]
        self.mask_filters = mask_filters

    @property
    def source(self) -> _Source:
        return self._source

    @source.setter
    def source(self, value) -> None:
        self._source = self._process_source(value)

    @property
    def mask(self) -> _Source:
        return self._mask

    @mask.setter
    def mask(self, value) -> None:
        if value is None:
            self._mask = None
        else:
            self._mask = self._process_source(value)

    # Public methods.
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        source = self._serialize_object(self._source)
        mask = self._serialize_object(self._mask)
        attrs = super().asdict()
        attrs['source'] = source
        attrs['mask'] = mask
        attrs['filters'] = self._serialize_object(self.filters)
        attrs['mask_filters'] = self._serialize_object(self.mask_filters)
        attrs['blend'] = op.get_regname_for_function(attrs['blend'])
        del attrs['type']
        return attrs

    # Private methods.
    def _process_source(self, value) -> t.Union[_Source, t.Sequence[_Source]]:
        # If passed a sequence, process recursively.
        if isinstance(value, t.Sequence):
            return tuple(self._process_source(item) for item in value)

        # If passed a valid object, return it.
        if isinstance(value, (s.ValueSource, self.__class__)):
            return value

        # Duck type for a serialized source.ValueSource. If that's
        # it, then deserialize and return.
        if 'type' in value:
            return s.deserialize_source(value)

        # Otherwise, assume it's a Layer, deserialize, and return.
        return self.__class__(**value)

    def _serialize_object(self, obj) -> _SerializedSource:
        if not obj:
            return None
        if isinstance(obj, t.Sequence):
            return tuple(self._serialize_object(item) for item in obj)
        return obj.asdict()


class Image():
    def __init__(self,
                 source: t.Union[Layer, t.Sequence[Layer], _SerializedSource],
                 size: t.Sequence[int],
                 filename: str,
                 format: str,
                 mode: str,
                 framerate: t.Union[None, float] = None) -> None:
        self.source = source
        self.size = size
        self.filename = filename
        self.format = format
        self.mode = mode
        self.framerate = framerate

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.asdict() == other.asdict()

    @property
    def source(self) -> _Source:
        return self._source

    @source.setter
    def source(self, value) -> None:
        self._source = self._process_source(value)

    # Public methods.
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = self.__dict__.copy()
        try:
            attrs['source'] = attrs['_source'].asdict()
        except AttributeError:
            attrs['source'] = [item.asdict() for item in attrs['_source']]
        del attrs['_source']
        return attrs

    def count_sources(self) -> int:
        """Return the number of ValueSources contained by the Image."""
        return count_sources(self.source)

    # Private methods.
    def _process_source(self, value) -> t.Union[Layer, t.Sequence[Layer]]:
        # If passed a sequence, process recursively.
        if isinstance(value, t.Sequence):
            return tuple(self._process_source(item) for item in value)

        # If passed a valid object, return it.
        if isinstance(value, Layer):
            return value

        # Otherwise, assume it's a serialized Layer, deserialize, and return.
        return Layer(**value)


# Utility functions.
def count_sources(obj: t.Union[_Source, t.Sequence[Layer]]) -> int:
    """Find the number of ValueSources contained in the object."""
    if isinstance(obj, t.Sequence):
        counts = [count_sources(item) for item in obj]
        return sum(counts)
    if isinstance(obj, Layer):
        return count_sources(obj.source) + count_sources(obj.mask)
    if isinstance(obj, s.ValueSource):
        return 1
    if obj is None:
        return 0
    else:
        msg = f'Unexpected type in Image: {obj.__class__.__name__}'
        raise TypeError(msg)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
