#!/usr/bin/env python
import pika
import pickle

def sendToClassificationQueue(message):
    serialized = pickle.dumps(message, protocol=0) # protocol 0 is printable ASCII

    connection1 = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
    channel1 = connection1.channel()

    channel1.queue_declare(queue='classificationQueue')

    channel1.basic_publish(exchange='', routing_key='classificationQueue', body=serialized)
    print("sending vectorrized data to classificationQueue")
    connection1.close()