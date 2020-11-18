"""
ui
~~

User interface elements for noise generation.
"""
from collections import deque
from queue import Queue
import sys
import time
from typing import Tuple

from pjinoise.constants import TEXT


# Shortcut names for writing to standard output.
write, flush = sys.stdout.write, sys.stdout.flush

# Status message commands.
INIT = 0x0
STATUS = 0x1
PROG = 0x2
KILL = 0xe
END = 0xf


def split_time(duration:float) -> Tuple[int]:
    s = duration % 60
    duration -= s
    m = duration % 3600
    duration -= m
    h = int(duration / 3600)
    m = int(m / 60)
    s = int(s)
    return h, m, s


def update_progress(progress:str, stages_done:int, status:deque) -> None:
    progress = list(progress)
    progress[stages_done] = '\u2588'
    progress = ''.join(progress)
    
    write('\033[A' * (len(status) + 2) + '\r')
    write(progress)
    write('\n' * (len(status) + 1) + '\r')
    
    return progress


def update_status(status:deque, newline:str, 
                  maxlines:int, roll:bool = True) -> None:
    if roll:
        for i in range(len(status))[::-1]:
            write('\r\033[A' + ' ' * len(status[i]))
        if len(status) >= maxlines:
            status.popleft()
        status.append(newline)
        for line in status:
            write('\r' + line + '\n')
    else:
        write('\r\033[A' + ' ' * len(status[-1]))
        status[-1] = newline
        write('\r' + status[-1] + '\n')


def status_writer(msg_queue:Queue, stages:int, maxlines:int = 4) -> None:
    t0 = time.time()
    stages_done = 0
    title = 'PJINOISE: Pattern and Noise Generation\n'
    bar_top = '\u250c' + ' ' * stages + '\u2510\n'
    progress = '\u2502' + '\u2591' * stages + '\u2502\n'
    bar_bot = '\u2514' + ' ' * stages + '\u2518\n'
    status_tmp = '{h:02d}:{m:02d}:{s:02d} {msg}'
    msg = 'Starting...'
    status = deque()
    runflag = False
    
    while True:
        h, m, s = split_time(time.time() - t0)
        if not msg_queue.empty():
            cmd, *args = msg_queue.get()
            if cmd == INIT:
                newline = status_tmp.format(h=h, m=m, s=s, msg=msg)
                write(title)
                write(bar_top)
                write(progress)
                write(bar_bot + '\r')
                update_status(status, newline, maxlines)
                runflag = True
            elif cmd == STATUS:
                msg=args[0]
                newline = status_tmp.format(h=h, m=m, s=s, msg=msg)
                update_status(status, newline, maxlines)
            elif cmd == PROG:
                stages_done += 1
                progress = update_progress(progress, stages_done, status)
                msg=args[0]
                newline = status_tmp.format(h=h, m=m, s=s, msg=msg)
                update_status(status, newline, maxlines)
            elif cmd == KILL:
                msg='Exception raised by core.'
                newline = status_tmp.format(h=h, m=m, s=s, msg=msg)
                update_status(status, newline, maxlines)
                raise args[0]
            elif cmd == END:
                write('\r' + ' ' * len(status) + '\r')
                msg=args[0]
                newline = status_tmp.format(h=h, m=m, s=s, msg=msg)
                update_status(status, newline, maxlines)
                flush()
                break
        elif runflag:
            time.sleep(1)
            newline = status_tmp.format(h=h, m=m, s=s, msg=msg)
            update_status(status, newline, maxlines, False)
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