from __future__ import absolute_import, unicode_literals

from celery import Celery

from web_server.models import ClassifiedIssue

app = Celery('celery', backend='redis://localhost:6379/0',
             broker='pyamqp://guest@localhost//')


@app.task
def classify(hashed_issue: dict) -> dict:
    classified_issue: ClassifiedIssue = ClassifiedIssue(
        digest=hashed_issue["digest"], labels=["Bug", "API"])
    # TODO Add classification logic here
    return classified_issue.dict()
