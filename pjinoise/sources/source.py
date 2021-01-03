"""
source
~~~~~~

Supplies the base class for the pjinoise.sources submodule.
"""
from abc import abstractmethod
from functools import wraps
import typing as t

from pjinoise import ease as e
from pjinoise.base import Serializable


# Common decorators.
def eased(fn: t.Callable) -> t.Callable:
    @wraps(fn)
    def wrapper(obj, *args, **kwargs) -> 'numpy.ndarray':
        return obj._ease(fn(obj, *args, **kwargs))
    return wrapper


# Base classes.
class Source(Serializable):
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
    def asdict(self) -> dict:
        """Serialize the object to a dictionary."""
        attrs = super().asdict()
        attrs['ease'] = self.ease
        return attrs

    @abstractmethod
    def fill(self, size: t.Sequence[int],
             location: t.Sequence[int] = None) -> 'numpy.ndarray':
        """Return a space filled with noise."""

    def noise(self, coords: t.Sequence[int]) -> int:
        """Generate the noise value for the given coordinates."""
        size = [1 for _ in range(len(coords))]
        value = self.fill(size, coords)
        index = [0 for _ in range(len(coords))]
        return value[index]
