"""
ui
~~

User interface elements for noise generation.
"""
from queue import Queue
import sys
import time
from typing import Tuple

from pjinoise.constants import TEXT


# Status message commands.
INIT = 0x0
STATUS = 0x1
PROG = 0x2
END = 0xf


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


def split_time(duration:float) -> Tuple[int]:
    s = duration % 60
    duration -= s
    m = duration % 3600
    duration -= m
    h = int(duration / 3600)
    m = int(m / 60)
    s = int(s)
    return h, m, s


def status_writer(msg_queue:Queue, stages:int) -> None:
    write, flush = sys.stdout.write, sys.stdout.flush
    t0 = time.time()
    stages_done = 0
    title = 'PJINOISE: Pattern and Noise Generation\n'
    progress = '\u2591' * stages + '\n'
    status_tmp = '{h:02d}:{m:02d}:{s:02d} {msg}'
    msg = 'Starting...'
    status = ''
    runflag = False
    
    while True:
        h, m, s = split_time(time.time() - t0)
        if not msg_queue.empty():
            cmd, *args = msg_queue.get()
            if cmd == INIT:
                status = status_tmp.format(h=h, m=m, s=s, msg=msg)
                write(title)
                write(progress)
                write(status)
                runflag = True
            elif cmd == STATUS:
                write('\x08' * len(status))
                write(' ' * len(status))
                write('\x08' * len(status))
                msg=args[0]
                status = status_tmp.format(h=h, m=m, s=s, msg=msg)
                write(status)
            elif cmd == PROG:
                write('\x08' * len(status))
                write(' ' * len(status))
                write('\x08' * len(status))
                stages_done += 1
                stages -= 1
                progress = '\u2588' * stages_done + '\u2591' * stages + '\n'
                write(f'\033[A{progress}')
                msg=args[0]
                status = status_tmp.format(h=h, m=m, s=s, msg=msg)
                write(status)
            elif cmd == END:
                write('\x08' * len(status))
                write(' ' * len(status))
                write('\x08' * len(status))
                msg=args[0]
                status = status_tmp.format(h=h, m=m, s=s, msg=msg)
                write(status + '\n')
                flush()
                break
        elif runflag:
            time.sleep(1)
            write('\x08' * len(status))
            write(' ' * len(status))
            write('\x08' * len(status))
            status = status_tmp.format(h=h, m=m, s=s, msg=msg)
            write(status)
        flush()


if __name__ == '__main__':
    from threading import Thread
    
    msg_queue = Queue()
    t = Thread(target=status_writer, args=(msg_queue, 3))
    t.start()
    msg_queue.put((INIT,))
    time.sleep(5)
    msg_queue.put((STATUS, 'Stage 1...'))
    time.sleep(2)
    msg_queue.put((PROG, 'Stage 2...'))
    time.sleep(5)
    msg_queue.put((PROG, 'Stage 3...'))
    time.sleep(2)
    msg_queue.put((PROG, 'Done.'))
    time.sleep(1)
    msg_queue.put((END, 'Good-bye.'))