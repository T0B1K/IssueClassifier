from hashlib import sha1
from typing import List

from fastapi import BackgroundTasks, FastAPI, Header, status
import uvicorn

# Only needed for debugging. Remove during deployment.
from resources.models import ClassifiedIssue, HashedIssue, Issue
from server_celery.celery_app import app as server_celery_app

app = FastAPI()

'''
BUG Callback not supported while RabbitMQ is set as celery
result backend https://github.com/celery/celery/issues/3625
'''
def print_classify_response(classified_issue: dict):
    print("Labels of issue with hash " + classified_issue["result"]["digest"] +
          ": " + str(classified_issue["labels"]))


async def send_issue_to_celery(hashed_issue: HashedIssue):
    task_name = "tasks.classify"
    task = server_celery_app.send_task(task_name, args=[hashed_issue.dict()])
    print("Sent issue with hash " + hashed_issue.digest + " for classification.")
    result = task.get(timeout=5)
    print("Labels of issue with hash " + result["digest"] +
          ": " + str(result["labels"]))



# TODO properly handle errors (e.g. during validation check)
@app.post("/classification/classify", response_model=List[HashedIssue], status_code=status.HTTP_202_ACCEPTED)
async def receive_issues(issues: List[Issue], background_tasks: BackgroundTasks, content_type: str = Header("application/json")):
    response: List[HashedIssue] = []
    for issue in issues:
        issue_hash = sha1((issue.body).encode('utf-8')).hexdigest()[:10]
        hashed_issue = HashedIssue(**issue.dict(), digest=issue_hash)
        background_tasks.add_task(send_issue_to_celery, hashed_issue)
        response.append(hashed_issue)
    return response

# Only needed for debugging. Remove during deployment.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
