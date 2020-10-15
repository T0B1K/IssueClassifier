import os
from typing import List

from celery import Celery
from numpy import array as np_array

from microservice.vectoriser.main import vectoriser as vectorise_string_list

CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', "amqp://guest:guest@rabbitmq:5672")
RESULT_BACKEND_URL = os.getenv('RESULT_BACKEND_URL', "redis://localhost")
VECTORISER_QUEUE = os.getenv('VECTORISER_QUEUE', "vectorise_queue")
CLASSIFIER_QUEUE = os.getenv('CLASSIFIER_QUEUE', "classify_queue")

app = Celery("celery_app", broker=CELERY_BROKER_URL,
             backend=RESULT_BACKEND_URL)

app.conf.task_routes = {
    'celery_app.vectorise': VECTORISER_QUEUE,
    'celery_app.classify': CLASSIFIER_QUEUE
}


@app.task
def vectorise(issues: List[str]) -> np_array:
    # FIXME Add working vectorisation logic
    # string_array = np_array(issues)
    # return vectorise_string_list(string_array)
    print("Celery vectoriser received issues: " + str(issues))
    return issues


@app.task(reply_to='result_queue')
def classify(issue: str) -> str:
    # FIXME Add working classification logic
    print("Celery classifier recieved issue: " + issue)
    return issue
