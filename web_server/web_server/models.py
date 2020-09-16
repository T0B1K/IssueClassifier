from pydantic import BaseModel, Field
from typing import List


class Issue(BaseModel):
    body: str = Field(..., title="Issue body text", description="The body text of the issue", min_length=1)

    class Config:
        schema_extra = {
            "example": {
                "body": "This issue represents a bug with regards to an API"
            }
        }


class HashedIssue(Issue):
    digest: str = Field(
        None, title="Issue digest", description="The first 10 characters of the SHA-1 digest of the issue body")

    class Config:
        schema_extra = {
            "example": {
                "body": "This issue represents a bug with regards to an API",
                "digest": "01f3451efe"
            }
        }


class ClassifiedIssue(HashedIssue):
    labels: List[str] = Field(
        None, title="Issue labels", description="The labels predicted by the classifier for this issue"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "body": "This issue represents a bug with regards to an API",
                "digest": "01f3451efe",
                "labels": ["Bug", "API"]
            }
        }
