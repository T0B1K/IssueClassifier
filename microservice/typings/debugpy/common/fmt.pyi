"""
This type stub file was generated by pyright.
"""

import string
import types

class Formatter(string.Formatter, types.ModuleType):
    """A custom string.Formatter with support for JSON pretty-printing.

    Adds {!j} format specification. When used, the corresponding value is converted
    to string using json_encoder.encode().

    Since string.Formatter in Python <3.4 does not support unnumbered placeholders,
    they must always be numbered explicitly - "{0} {1}" rather than "{} {}". Named
    placeholders are supported.
    """
    def __init__(self) -> None:
        ...
    
    def __call__(self, format_string, *args, **kwargs):
        """Same as self.format().
        """
        ...
    
    def convert_field(self, value, conversion):
        ...
    

