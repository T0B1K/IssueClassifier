from marshmallow import Schema, fields, validate, ValidationError, validates_schema, post_load
from .models import Issue


class IssueSchema(Schema):
    body = fields.Str(required=True, validate=validate.Length(
        min=1), error_messages={"required": "Issue body must be supplied!"})

    @post_load
    def make_issue(self, data, **kwargs):
        return Issue(**data)
