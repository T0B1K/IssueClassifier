from marshmallow import Schema, fields, post_load
from mom.models import Issue

class IssueSchema(Schema):
    digest = fields.Str()
    body = fields.Str()
    labels = fields.List(fields.Str())

    @post_load
    def make_issue(self, data, **kwargs):
        return Issue(**data)