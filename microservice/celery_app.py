from typing import List

from celery import Celery
from numpy import array as np_array

from vectoriser.main import get_from_queue as vectorise_string_list

app = Celery("celery_app", broker="amqp://localhost",
             backend="redis://localhost")

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
