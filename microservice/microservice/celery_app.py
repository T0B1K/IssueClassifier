import os
from typing import List

from celery import Celery
from numpy import array as np_array

from microservice.vectoriser.main import vectoriser as vectorise_string_list

CELERY_BROKER_URL = os.environ['CELERY_BROKER_URL'] or "amqp://guest:guest@rabbitmq:5672"
RESULT_BACKEND_URL = os.environ['RESULT_BACKEND_URL'] or "redis://localhost"

app = Celery("celery_app", broker=CELERY_BROKER_URL,
             backend=RESULT_BACKEND_URL)

app.conf.task_routes = {
    'celery_app.vectorise': 'vectorise_queue',
    'celery_app.classify': 'classify_queue'
}


@app.task
def vectorise(issues: List[str]) -> np_array:
    string_array = np_array(issues)
    return vectorise_string_list(string_array)


@app.task(reply_to='result_queue')
def classify(issue: str) -> str:
    print("Classifying issue: " + issue)
    return issue
