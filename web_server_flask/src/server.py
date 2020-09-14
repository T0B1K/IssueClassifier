import json

from flask import Flask, request
from flask_restx import Api, Resource, fields
from http import HTTPStatus
from werkzeug.middleware.proxy_fix import ProxyFix

from .models import Issue
from .schemas import IssueSchema
from .server_helper import process_issues

# Flask setup
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='Issue Classifier Microsrevice',
          description='HTTP endpoint for the issue classifier microservice')

# Setting up Swagger UI
classification = api.namespace(
    'classification', description='Classification operations')
issue_model = api.model(
    'Issue', {
        'body': fields.String(required=True, description='The issue body'),
    }
)

# Initialising a Marshmallow scheme for input validation
issue_schema = IssueSchema(many=True)


@classification.route('/classify')
class IssueResource(Resource):

    @classification.expect(issue_model)
    @classification.marshal_list_with(issue_model)
    def post(self):
        request_issues = request.get_json()

        if request_issues is None:
            return ({"Error": "Either the Content-Type header wasn't specified as application/json or the request body is empty."}, HTTPStatus.BAD_REQUEST.value)

        error_messages = issue_schema.validate(request_issues)
        if not error_messages:
            return (error_messages, HTTPStatus.BAD_REQUEST.value)

        return process_issues(request_issues)
