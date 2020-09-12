from mom.models import Issue
from mom.schemas import IssueSchema
import json
import time
import os
import hashlib
from collections.abc import Iterable
from flask import Flask, request
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
import pika
import sys
sys.path.append(
    "/home/aly-mohamed/Work/Universitaet Stuttgart/Semester/6. Semester/Bachelor/FP/IssueClassifier")


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='1.0', title='Issue Classifier Microsrevice',
          description='HTTP endpoint for the issue classifier microservice')

classification_ns = api.namespace(
    'classification', description='Classification operations')

issue_model = api.model(
    'Issue', {
        'digest': fields.String(readonly=True, description='The hash digest of the issue (for unique identification)'),
        'body': fields.String(required=True, description='The issue body'),
        'labels': fields.List(fields.String)
    }
)

issue_schema = IssueSchema()


@classification_ns.route('/classify')
class Issue_Resource(Resource):

    def gen_issue_digest(self, issue):
        issue_body_encoded = (issue.body).encode('utf-8')
        issue_digest = hashlib.sha1(issue_body_encoded)
        issue.digest = issue_digest.hexdigest()

    def process_issues(self, request_issue):
        rmq_connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))
        rmq_channel = rmq_connection.channel()
        rmq_channel.exchange_declare(
            exchange='classify_issue', exchange_type='fanout')

        try:
            self.gen_issue_digest(request_issue)
            rmq_channel.basic_publish(
                exchange='classify_issue', routing_key='', body=issue_schema.dumps(request_issue))
        finally:
            rmq_connection.close()

    @classification_ns.expect(issue_model)
    @classification_ns.marshal_list_with(issue_model)
    def post(self):
        request_data = request.get_json()
        request_issue = issue_schema.load(request_data)
        self.process_issues(request_issue)


if __name__ == '__main__':
    app.run(debug=True)
