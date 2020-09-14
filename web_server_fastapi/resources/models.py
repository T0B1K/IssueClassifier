from pydantic import BaseModel, Field, validator


class Issue(BaseModel):
    body: str = Field(..., title="The body text of the issue", min_length=1)

    class Config:
        schema_extra = {
            "example": {
                "body": "Test issue"
            }
        }


class HashedIssue(Issue):
    digest: str = Field(
        None, title="The first 8 characters of the SHA-1 digest of the issue body")

    class Config:
        schema_extra = {
            "example": {
                "body": "Test issue",
                "digest": "6201223ff2"
            }
        }
