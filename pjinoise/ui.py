"""
ui
~~

User interface elements for noise generation.
"""
import time
from typing import Tuple

from pjinoise.constants import TEXT


class Status():
    def __init__(self) -> None:
        self.t0 = time.time()
        self.last_key = None
        self.last_line = None
        msg = TEXT['start'].format(min=0, sec=0)
        print(msg)
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    def end(self) -> None:
        min, sec = self._duration()
        msg = TEXT['end'].format(min=min, sec=sec)
        print(msg)
    
    def update(self, key:str, *args) -> None:
        min, sec = self._duration()
        msg = TEXT[key].format(min, sec, *args)
        print(msg)
    
    def _duration(self) -> Tuple[int]:
        now = time.time() - self.t0
        min = int(now // 60)
        sec = int(now % 60)
        return min, sec
        