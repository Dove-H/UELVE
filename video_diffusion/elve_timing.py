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
