"""
ELVE_TIMING=1 环境变量开启的耗时累计器：分桶累计多卡通信（nccl init/broadcast/
send/recv/barrier）与 inversion 特征换入换出（D2H/落盘写/预取读/主线程等待/H2D）。
关闭时 add/timer 为近零开销空操作，不影响常规基准计时。
"""
import json
import os
import threading
import time
from contextlib import contextmanager

ENABLED = os.environ.get('ELVE_TIMING', '').strip() == '1'
_lock = threading.Lock()
_acc = {}  # key -> {n, total, max, first, bytes}


def add(key, dt, nbytes=0):
    if not ENABLED:
        return
    with _lock:
        d = _acc.setdefault(key, dict(n=0, total=0.0, max=0.0, first=None, bytes=0))
        if d['first'] is None:
            d['first'] = dt
        d['n'] += 1
        d['total'] += dt
        d['max'] = max(d['max'], dt)
        d['bytes'] += nbytes


def add_bytes(key, nbytes):
    """只累计数据量不记次数（配合 timer 使用，避免 n 被重复计数）。"""
    if not ENABLED or not nbytes:
        return
    with _lock:
        d = _acc.setdefault(key, dict(n=0, total=0.0, max=0.0, first=None, bytes=0))
        d['bytes'] += nbytes


@contextmanager
def timer(key):
    if not ENABLED:
        yield
        return
    t0 = time.perf_counter()
    yield
    add(key, time.perf_counter() - t0)


def format_report(tag):
    lines = [f'[ELVE-TIMING {tag}] {"key":<20} {"n":>6} {"total_s":>9} '
             f'{"avg_ms":>9} {"max_ms":>9} {"first_ms":>9} {"MB":>10}']
    with _lock:
        rows = sorted(_acc.items(), key=lambda kv: -kv[1]['total'])
        for k, d in rows:
            lines.append(f'[ELVE-TIMING {tag}] {k:<20} {d["n"]:6d} {d["total"]:9.3f} '
                         f'{d["total"] / d["n"] * 1000:9.3f} {d["max"] * 1000:9.3f} '
                         f'{(d["first"] or 0) * 1000:9.3f} {d["bytes"] / 1e6:10.1f}')
    return '\n'.join(lines)


def dump(path):
    if not ENABLED:
        return
    with _lock:
        with open(path, 'w') as f:
            json.dump(_acc, f, indent=1)
