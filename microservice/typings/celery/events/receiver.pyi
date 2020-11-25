"""
This type stub file was generated by pyright.
"""

from operator import itemgetter
from kombu.mixins import ConsumerMixin

"""Event receiver implementation."""
CLIENT_CLOCK_SKEW = - 1
_TZGETTER = itemgetter('utcoffset', 'timestamp')
class EventReceiver(ConsumerMixin):
    """Capture events.

    Arguments:
        connection (kombu.Connection): Connection to the broker.
        handlers (Mapping[Callable]): Event handlers.
            This is  a map of event type names and their handlers.
            The special handler `"*"` captures all events that don't have a
            handler.
    """
    app = ...
    def __init__(self, channel, handlers=..., routing_key=..., node_id=..., app=..., queue_prefix=..., accept=..., queue_ttl=..., queue_expires=...) -> None:
        ...
    
    def process(self, type, event):
        """Process event by dispatching to configured handler."""
        ...
    
    def get_consumers(self, Consumer, channel):
        ...
    
    def on_consume_ready(self, connection, channel, consumers, wakeup=..., **kwargs):
        ...
    
    def itercapture(self, limit=..., timeout=..., wakeup=...):
        ...
    
    def capture(self, limit=..., timeout=..., wakeup=...):
        """Open up a consumer capturing events.

        This has to run in the main process, and it will never stop
        unless :attr:`EventDispatcher.should_stop` is set to True, or
        forced via :exc:`KeyboardInterrupt` or :exc:`SystemExit`.
        """
        ...
    
    def wakeup_workers(self, channel=...):
        ...
    
    def event_from_message(self, body, localize=..., now=..., tzfields=..., adjust_timestamp=..., CLIENT_CLOCK_SKEW=...):
        ...
    
    @property
    def connection(self):
        ...
    

