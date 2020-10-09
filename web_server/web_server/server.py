import json
from typing import List

from flask import Flask
from flask_mqtt import Mqtt
from pydantic.error_wrappers import ValidationError
from werkzeug.middleware.proxy_fix import ProxyFix

from web_server.ws_helpers.ws_models import UnclassifiedIssue
from web_server.ws_helpers.ws_parser import parse_payload
from web_server.ws_helpers.ws_pika import classify_issues

# Flask setup
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# MQTTT Setup
app.config['MQTT_BROKER_URL'] = 'localhost'
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_USERNAME'] = ''
app.config['MQTT_PASSWORD'] = ''
app.config['MQTT_KEEPALIVE'] = 5
app.config['MQTT_TLS_ENABLED'] = False
mqtt = Mqtt(app)


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    mqtt.subscribe('classification/classify')


@mqtt.on_topic("classification/classify")
def handle_issues_classification_request(client, userdata, message):
    try:
        unclassified_issues: List[UnclassifiedIssue] = parse_payload(
            message.payload)
        classify_issues(unclassified_issues)
    except ValidationError as exc:
        print(exc.json())
