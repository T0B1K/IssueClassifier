from web_server.celery import app as celery_app
from web_server.models import HashedIssue


def print_classify_response(classified_issue: dict):
    print("Labels of issue with hash " + classified_issue["result"]["digest"] +
          ": " + str(classified_issue["labels"]))


async def send_issue_to_celery(hashed_issue: HashedIssue):
    task_name = "tasks.classify"
    task = celery_app.send_task(task_name, args=[hashed_issue.dict()])
    print("Sent issue with hash " + hashed_issue.digest + " for classification.")
    result = task.get(timeout=5)
    print("Labels of issue with hash " + result["digest"] +
          ": " + str(result["labels"]))
