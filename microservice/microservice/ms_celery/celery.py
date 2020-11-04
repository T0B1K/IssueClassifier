"""Primary Celery application instance.

This is the entry-point for any and all operations that should be carried out by the celery workers instead of the running process/thread. Since classification and vectorisation computations take a relatively long amount of time and such computations should be carried out asynchronously, Celery is used for this purpose.

A Docker container with a running celery worker can indeed contain several processes/threads. See https://docs.docker.com/config/containers/multi-service_container/.
"""
from celery import Celery

app = Celery("celery")

app.config_from_object("microservice.config.celeryconfig")
