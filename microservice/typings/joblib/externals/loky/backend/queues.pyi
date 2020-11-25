"""
This type stub file was generated by pyright.
"""

import sys
from multiprocessing.queues import Queue as mp_Queue, SimpleQueue as mp_SimpleQueue

class Queue(mp_Queue):
    def __init__(self, maxsize=..., reducers=..., ctx=...) -> None:
        ...
    
    def __getstate__(self):
        ...
    
    def __setstate__(self, state):
        ...
    
    if sys.version_info[: 2] < (3, 4):
        ...


class SimpleQueue(mp_SimpleQueue):
    def __init__(self, reducers=..., ctx=...) -> None:
        ...
    
    def close(self):
        ...
    
    def __getstate__(self):
        ...
    
    def __setstate__(self, state):
        ...
    
    if sys.version_info[: 2] < (3, 4):
        def get(self):
            ...
        
    def put(self, obj):
        ...
    

