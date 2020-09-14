from hashlib import sha1
from fastapi import FastAPI, BackgroundTasks, status
from resources.models import Issue, HashedIssue
from typing import List
import uvicorn  # Only needed for debugging. Remove during deployment.

app = FastAPI()


def publish_to_broker(issue: HashedIssue):
    '''
    TODO implement publish functionality from server to MQTT topic
    INFO Will most likely be implemented in a separate module for readability and cleaner code
    Furhtermore, may not 
    '''
    print("Published issue with hash: " + issue.digest)


@app.post("/classification/classify", response_model=List[HashedIssue], status_code=status.HTTP_202_ACCEPTED)
async def receive_issues(issues: List[Issue], background_tasks: BackgroundTasks):
    # TODO properly handle errors (e.g. during validation check)
    response: List[HashedIssue] = []
    for issue in issues:
        issue_hash = sha1((issue.body).encode('utf-8')).hexdigest()[:10]
        hashed_issue = HashedIssue(**issue.dict(), digest=issue_hash)
        background_tasks.add_task(publish_to_broker, hashed_issue)
        response.append(hashed_issue)
    return response

# Only needed for debugging. Remove during deployment.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
