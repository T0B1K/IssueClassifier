from hashlib import sha1
from typing import List

from fastapi import BackgroundTasks, FastAPI, Header, status
import uvicorn

from web_server.models import HashedIssue, Issue
from web_server.server_methods import send_issue_to_celery

app = FastAPI()


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
