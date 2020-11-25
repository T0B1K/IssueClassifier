"""
This type stub file was generated by pyright.
"""

import os
import platform
import weakref
from collections import Counter
from kombu.utils.objects import cached_property
from celery import __version__
from celery.utils.collections import LimitedSet

"""Internal worker state (global).

This includes the currently active and reserved tasks,
statistics, and revoked tasks.
"""
SOFTWARE_INFO = { 'sw_ident': 'py-celery','sw_ver': __version__,'sw_sys': platform.system() }
REVOKES_MAX = 50000
REVOKE_EXPIRES = 10800
requests = {  }
reserved_requests = weakref.WeakSet()
active_requests = weakref.WeakSet()
total_count = Counter()
all_total_count = [0]
revoked = LimitedSet(maxlen=REVOKES_MAX, expires=REVOKE_EXPIRES)
should_stop = None
should_terminate = None
def reset_state():
    ...

def maybe_shutdown():
    """Shutdown if flags have been set."""
    ...

def task_reserved(request, add_request=..., add_reserved_request=...):
    """Update global state when a task has been reserved."""
    ...

def task_accepted(request, _all_total_count=..., add_active_request=..., add_to_total_count=...):
    """Update global state when a task has been accepted."""
    ...

def task_ready(request, remove_request=..., discard_active_request=..., discard_reserved_request=...):
    """Update global state when a task is ready."""
    ...

C_BENCH = os.environ.get('C_BENCH') or os.environ.get('CELERY_BENCH')
C_BENCH_EVERY = int(os.environ.get('C_BENCH_EVERY') or os.environ.get('CELERY_BENCH_EVERY') or 1000)
if C_BENCH:
    all_count = 0
    bench_first = None
    bench_start = None
    bench_last = None
    bench_every = C_BENCH_EVERY
    bench_sample = []
    __reserved = task_reserved
    __ready = task_ready
    def task_reserved(request):
        """Called when a task is reserved by the worker."""
        ...
    
    def task_ready(request):
        """Called when a task is completed."""
        ...
    
class Persistent:
    """Stores worker state between restarts.

    This is the persistent data stored by the worker when
    :option:`celery worker --statedb` is enabled.

    Currently only stores revoked task id's.
    """
    storage = ...
    protocol = ...
    compress = ...
    decompress = ...
    _is_open = ...
    def __init__(self, state, filename, clock=...) -> None:
        ...
    
    def open(self):
        ...
    
    def merge(self):
        ...
    
    def sync(self):
        ...
    
    def close(self):
        ...
    
    def save(self):
        ...
    
    @cached_property
    def db(self):
        ...
    

