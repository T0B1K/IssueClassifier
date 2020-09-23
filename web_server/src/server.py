from flask import Flask
from flask_mqtt import Mqtt
from werkzeug.middleware.proxy_fix import ProxyFix

from .schemas import IssueSchema

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

# Marshmallow Setup
issue_schema = IssueSchema(many=True)


@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    mqtt.subscribe('classification/classify')


@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    data = dict(
        topics=message.topic,
        payload=message.payload.decode()
    )
    print(data)
