"""
base
~~~~

Core base classes for the pjinoise module.
"""
from abc import ABC, abstractmethod
import inspect
import typing as t

from pjinoise import common as c
from pjinoise import ease as e


# Base classes.
class Serializable(ABC):
    """A class that can be compared and serialized to a dictionary."""
    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.asdict() == other.asdict()

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        attrs = self.asdict()
        if 'type' in attrs:
            del attrs['type']
        params = []
        for key in attrs:
            val = attrs[key]
            if isinstance(val, str):
                val = f"'{val}'"
            if isinstance(val, bytes):
                val = f"b'{val}'"
            params.append(f'{key}={val}')
        params_str = ', '.join(params)
        return f'{cls}({params_str})'

    # Public methods.
    def asargs(self) -> t.List[t.Any]:
        sig = inspect.signature(self.__init__)
        kwargs = self.asdict()
        args = [kwargs[key] for key in sig.parameters if key in kwargs]
        return args

    def asdict(self) -> t.Dict[str, t.Any]:
        """Serialize the object to a dictionary."""
        attrs = self.__dict__.copy()
        cls = self.__class__.__name__
        attrs['type'] = cls.casefold()
        attrs = c.remove_private_attrs(attrs)
        return attrs


class Filter(Serializable):
    """The base class for filter objects."""
    padding = None

    # Public methods.
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = super().asdict()
        if 'ease' in attrs:
            attrs['ease'] = e.get_regname_for_func(attrs['ease'])
        if 'padding' in attrs:
            del attrs['padding']
        return attrs

    def preprocess(self, size: t.Sequence[int],
                   original_size: t.Sequence[int]) -> t.Tuple[int]:
        """Determine the size the filter needs the image to be during
        processing.
        """
        return tuple(size)

    @abstractmethod
    def process(self, values: 'numpy.array') -> 'numpy.array':
        """Run the filter over the image."""

    def postprocess(self, size: t.Sequence[int], *args) -> t.Sequence[int]:
        """Return the original size of the image."""
        if self.padding is None:
            return size
        return tuple(n - pad for n, pad in zip(size, self.padding))
