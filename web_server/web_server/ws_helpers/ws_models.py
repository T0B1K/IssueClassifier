from typing import List

from pydantic import BaseModel, Field

class BaseIssue(BaseModel):
    issue_id: int = Field(...,
                          title="Issue ID",
                          description="The unique ID for the current issue",
                          min_length=1)

class UnclassifiedIssue(BaseIssue):
    body: str = Field(..., title="Issue body text",
                      description="The body text of the issue", min_length=1)


class ClassifiedIssue(BaseIssue):
    labels: List[str] = Field(
        None, title="Issue labels", description="The labels predicted by the classifier for this issue"
    )