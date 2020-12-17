"""
base
~~~~

Core base classes for the pjinoise module.
"""
from abc import ABC
import inspect
import typing as t

from pjinoise import common as c


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
