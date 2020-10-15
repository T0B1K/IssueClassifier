import os
import pickle
from typing import Any

from celery import Celery

from microservice.vectoriser.main import vectorise_issues
from microservice.classifier.main import classify_issues

CELERY_BROKER_URL = os.getenv(
    'CELERY_BROKER_URL', "amqp://guest:guest@rabbitmq:5672")
RESULT_BACKEND_URL = os.getenv('RESULT_BACKEND_URL', "redis://localhost")
VECTORISER_QUEUE = os.getenv('VECTORISER_QUEUE', "vectorise_queue")
CLASSIFIER_QUEUE = os.getenv('CLASSIFIER_QUEUE', "classify_queue")

app = Celery("celery_app", broker=CELERY_BROKER_URL,
             backend=RESULT_BACKEND_URL)

app.conf.task_routes = {
    'celery_app.vectorise': VECTORISER_QUEUE,
    'celery_app.classify': CLASSIFIER_QUEUE
}

app.conf.result_serializer = 'pickle'
app.conf.task_serializer = 'pickle'
app.conf.accept_content = ['pickle', 'json']


@app.task
def vectorise(issues: Any) -> Any:
    return vectorise_issues(issues)


@app.task(reply_to='ic_microservice_output_queue')
def classify(vectorised_issues: Any) -> Any:
    return classify_issues(vectorised_issues)
