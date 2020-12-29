"""
base
~~~~

Core base classes for the pjinoise module.
"""
from abc import ABC, abstractmethod
from functools import wraps
import inspect
import typing as t

from pjinoise import common as c
from pjinoise import ease as e


# Common decorators.
def eased(fn: t.Callable) -> t.Callable:
    @wraps(fn)
    def wrapper(obj, *args, **kwargs) -> 'numpy.ndarray':
        return obj._ease(fn(obj, *args, **kwargs))
    return wrapper


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


class ValueSource(Serializable):
    """Base class to define common features of noise classes.

    :param ease: (Optional.) The easing function to use on the
        generated noise.
    :return: ABCs cannot be instantiated.
    :rtype: ABCs cannot be instantiated.
    """
    def __init__(self, ease: str = 'l', *args, **kwargs) -> None:
        self.ease = ease

    @property
    def ease(self) -> str:
        return e.get_regname_for_func(self._ease)

    @ease.setter
    def ease(self, value: str) -> None:
        self._ease = e.registered_functions[value]

    # Public methods.
    def asargs(self) -> t.List[t.Any]:
        sig = inspect.signature(self.__init__)
        kwargs = self.asdict()
        args = [kwargs[key] for key in sig.parameters if key in kwargs]
        return args

    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = self.__dict__.copy()
        cls = self.__class__.__name__
        attrs['type'] = cls.casefold()
        attrs['ease'] = self.ease
        attrs = c.remove_private_attrs(attrs)
        return attrs

    @abstractmethod
    def fill(self, size: t.Sequence[int],
             location: t.Sequence[int] = None) -> 'numpy.ndarray':
        """Return a space filled with noise."""

    def noise(self, coords: t.Sequence[float]) -> int:
        """Generate the noise value for the given coordinates."""
        size = [1 for n in range(len(coords))]
        value = self.fill(size, coords)
        index = [0 for n in range(len(coords))]
        return value[index]
