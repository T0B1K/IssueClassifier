"""
This type stub file was generated by pyright.
"""

import debugpy

TARGET = "<filename> | -m <module> | -c <code> | --pid <pid>"
HELP = """debugpy {0}
See https://aka.ms/debugpy for documentation.

Usage: debugpy --listen | --connect
               [<host>:]<port>
               [--wait-for-client]
               [--configure-<name> <value>]...
               [--log-to <path>] [--log-to-stderr]
               {1}
               [<arg>]...
""".format(debugpy.__version__, TARGET)
class Options(object):
    mode = ...
    address = ...
    log_to = ...
    log_to_stderr = ...
    target = ...
    target_kind = ...
    wait_for_client = ...
    adapter_access_token = ...


options = Options()
def in_range(parser, start, stop):
    ...

pid = in_range(int, 0, None)
def print_help_and_exit(switch, it):
    ...

def print_version_and_exit(switch, it):
    ...

def set_arg(varname, parser=...):
    ...

def set_const(varname, value):
    ...

def set_address(mode):
    ...

def set_config(arg, it):
    ...

def set_target(kind, parser=..., positional=...):
    ...

switches = [("-(\\?|h|-help)", None, print_help_and_exit), ("-(V|-version)", None, print_version_and_exit), ("--log-to", "<path>", set_arg("log_to")), ("--log-to-stderr", None, set_const("log_to_stderr", True)), ("--listen", "<address>", set_address("listen")), ("--connect", "<address>", set_address("connect")), ("--wait-for-client", None, set_const("wait_for_client", True)), ("--configure-.+", "<value>", set_config), ("--adapter-access-token", "<token>", set_arg("adapter_access_token")), ("", "<filename>", set_target("file", positional=True)), ("-m", "<module>", set_target("module")), ("-c", "<code>", set_target("code")), ("--pid", "<pid>", set_target("pid", pid))]
def consume_argv():
    ...

def parse_argv():
    ...

def start_debugging(argv_0):
    ...

def run_file():
    ...

def run_module():
    ...

def run_code():
    ...

def attach_to_pid():
    ...

def main():
    ...
