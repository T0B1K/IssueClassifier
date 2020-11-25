"""
This type stub file was generated by pyright.
"""

import os
import sys
from contextlib import contextmanager

"""Utilities related to importing modules and symbols by name."""
MP_MAIN_FILE = os.environ.get('MP_MAIN_FILE')
class NotAPackage(Exception):
    """Raised when importing a package, but it's not a package."""
    ...


if sys.version_info > (3, 3):
    def qualname(obj):
        """Return object name."""
        ...
    
else:
    ...
def instantiate(name, *args, **kwargs):
    """Instantiate class by name.

    See Also:
        :func:`symbol_by_name`.
    """
    ...

@contextmanager
def cwd_in_path():
    """Context adding the current working directory to sys.path."""
    ...

def find_module(module, path=..., imp=...):
    """Version of :func:`imp.find_module` supporting dots."""
    ...

def import_from_cwd(module, imp=..., package=...):
    """Import module, temporarily including modules in the current directory.

    Modules located in the current directory has
    precedence over modules located in `sys.path`.
    """
    ...

def reload_from_cwd(module, reloader=...):
    """Reload module (ensuring that CWD is in sys.path)."""
    ...

def module_file(module):
    """Return the correct original file name of a module."""
    ...

def gen_task_name(app, name, module_name):
    """Generate task name from name/module pair."""
    ...

def load_extension_class_names(namespace):
    ...

def load_extension_classes(namespace):
    ...
