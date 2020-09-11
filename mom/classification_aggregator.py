from flask import Flask
from flask_restx import Api, Resouce, fields
import pika
import sys
import os
import time
import train

app = Flask(__name__)

api = Api(app, version='1.0', title='Issue Classifier Microsrevice', description='HTTP endpoint for the issue classifier microservice')

ns = api.namespace('classification', description='Classification operations')

issue = api.model(
    'Issue', {
        'body': fields.Integer(readonly=True, description='The issue body')
    }
)

@ns.route('/classify')
class HelloWorld(Resource):
    def post(self):
        pass