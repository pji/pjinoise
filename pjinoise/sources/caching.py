"""
caching
~~~~~~~

Provides the Caching mixin, used to create sources that can cache
generated fill data.
"""
from typing import Sequence


class CachingMixin():
    """Cache fill results."""
    # This sets up a cache in the CachingMixin class, which, if used
    # would cause all instances with the same key to retur the same
    # cached value regardless of subtype. To avoid this, classes
    # using the CachingMixin should override this with their own
    # dictionary.
    _cache = {}

    def __init__(self, key: str = '_default', *args, **kwargs) -> None:
        self.key = key
        super().__init__(*args, **kwargs)

    # Public methods.
    def fill(self, size: Sequence[int],
             location: Sequence[int] = (0, 0, 0)) -> 'numpy.ndarray':
        """On first call, generate and return image data, caching
        the data. On subsequent calls, return the cached data
        rather than generating the same data again.
        """
        if self.key not in self._cache:
            self._cache[self.key] = super().fill(size, location)
        return self._cache[self.key].copy()
