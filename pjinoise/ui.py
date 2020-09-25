"""
ui
~~

User interface elements for noise generation.
"""
import time
from typing import Tuple

from pjinoise.constants import TEXT


class Status():
    def __init__(self, filename:str) -> None:
        self.t0 = time.time()
        msg = TEXT['start'].format(min=0, sec=0, filename=filename)
        print(msg)
    
    def end(self, filename:str) -> None:
        min, sec = self._duration()
        msg = TEXT['end'].format(min=min, sec=sec, filename=filename)
        print(msg)
    
    def _duration(self) -> Tuple[int]:
        now = time.time() - self.t0
        min = int(now // 60)
        sec = int(now % 60)
        return min, sec
        