"""
model
~~~~~

Data objects to use for the pjinoise module.
"""
import typing as t

from pjinoise import filters as f
from pjinoise import operations as op
from pjinoise import sources as s


_Source = t.Union[s.ValueSource, 'Layer']
_SerializedSource = t.Union[t.Mapping, t.Sequence[t.Mapping]]


class Layer():
    def __init__(self,
                 source: t.Union[_Source,
                                 t.Sequence[_Source],
                                 _SerializedSource],
                 blend: t.Union[str, t.Callable],
                 blend_amount: float = 1,
                 location: t.Sequence[int] = (0, 0, 0),
                 filters: t.Sequence[f.ForLayer] = None,
                 mask: t.Union[None, s.ValueSource] = None,
                 mask_filters: t.Sequence[f.ForLayer] = None) -> None:
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
        if mask_filters is None:
            mask_filters = []
        self.mask_filters = mask_filters

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
        attrs['filters'] = [item.asdict() for item in attrs['filters']]
        if attrs['mask']:
            attrs['mask'] = attrs['mask'].asdict()
        attrs['mask_filters'] = [f.asdict() for f in attrs['mask_filters']]
        attrs['blend'] = op.op_names[attrs['blend']]
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
