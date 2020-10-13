#!/usr/bin/env python
import pika, sys, os
import pickle
from classifierPrediction import predict


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='classificationQueue')

    def callback(ch, method, properties, body):
        deserializedData = pickle.loads(body)
        print( predict(deserializedData))
        
    channel.basic_consume(queue='classificationQueue', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages in the classificationQueue')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)