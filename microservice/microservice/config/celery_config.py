"""Celery configuration for the Issue Classifier Microservice.

This contains various configurations values for the celery application, most of
which are defined by environment variables. The rationale for choosing
environment variables is to be able to customise various options on starting the
microservice using Docker Compose.

For more information on what these configuration attributes mean, see
https://docs.celeryproject.org/en/stable/userguide/configuration.html
"""
from os import getenv

imports = ["microservice.classifier_celery.tasks"]

broker_url = getenv("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672")

task_routes = {
    "microservice.classifier_celery.tasks.classify_issues": getenv(
        "CLASSIFY_QUEUE", "classify_queue"
    ),
    "microservice.classifier_celery.tasks.vectorise_issues": getenv(
        "VECTORISE_QUEUE", "vectorise_queue"
    ),
}

result_serializer = "pickle"
task_serializer = "pickle"
accept_content = ["pickle"]

task_acks_late = True
task_ignore_result = True

worker_prefetch_multiplier = 1
