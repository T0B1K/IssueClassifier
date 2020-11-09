"""Celery configuration for the Issue Classifier Microservice.

This contains various configurations values for the celery application, most of which are defined by environment variables. The rationale for choosing envirionment variables is to be able to customise various options on starting the microservice using Docker Compose.

For more information on what these configuration attributes mean, see https://docs.celeryproject.org/en/stable/userguide/configuration.html
"""
from os import getenv

imports = ["microservice.ms_celery.tasks"]

broker_url = getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672")
result_backend = getenv("RESULT_BACKEND_URL", "redis://localhost")

task_routes = {
    "celery.classify_issues": getenv("CLASSIFY_QUEUE", "classify_queue"),
    "celery.vectorise_issues": getenv("VECTORISE_QUEUE", "vectorise_queue"),
}

result_serializer = "pickle"
task_serializer = "pickle"
accept_content = ["pickle", "json"]
task_acks_late = True
