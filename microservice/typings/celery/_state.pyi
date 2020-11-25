"""
This type stub file was generated by pyright.
"""

import os
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack

"""Internal state.

This is an internal module containing thread state
like the ``current_app``, and ``current_task``.

This module shouldn't be used directly.
"""
default_app = None
app_or_default = None
_apps = weakref.WeakSet()
_on_app_finalizers = set()
_task_join_will_block = False
def connect_on_app_finalize(callback):
    """Connect callback to be called when any app is finalized."""
    ...

def task_join_will_block():
    ...

class _TLS(threading.local):
    current_app = ...


_tls = _TLS()
_task_stack = LocalStack()
push_current_task = _task_stack.push
pop_current_task = _task_stack.pop
def set_default_app(app):
    """Set default app."""
    ...

if os.environ.get('C_STRICT_APP'):
    def get_current_app():
        """Return the current app."""
        ...
    
else:
    def get_current_app():
        ...
    
    get_current_app = _get_current_app
def get_current_task():
    """Currently executing task."""
    ...

def get_current_worker_task():
    """Currently executing task, that was applied by the worker.

    This is used to differentiate between the actual task
    executed by the worker and any task that was called within
    a task (using ``task.__call__`` or ``task.apply``)
    """
    ...

current_app = Proxy(get_current_app)
current_task = Proxy(get_current_task)
def enable_trace():
    """Enable tracing of app instances."""
    ...

def disable_trace():
    """Disable tracing of app instances."""
    ...

if os.environ.get('CELERY_TRACE_APP'):
    ...
else:
    ...