from __future__ import absolute_import, unicode_literals
from typing import List

from pydantic import BaseModel, Field

from celery_app import app as server_celery_app

# TODO Find a way to cleanly allow Celery tasks to import Pydantic models also used by server
class ClassifiedIssue(BaseModel):
    digest: str = Field(
        None, title="Issue digest", description="The first 10 characters of the SHA-1 digest of the issue body")
    labels: List[str] = Field(
        None, title="Issue labels", description="The labels predicted by the classifier for this issue"
    )


@server_celery_app.task
def classify(hashed_issue: dict) -> dict:
    classified_issue: ClassifiedIssue = ClassifiedIssue(
        digest=hashed_issue["digest"], labels=["Bug", "API"])
    # TODO Add classification logic here
    return classified_issue.dict()
