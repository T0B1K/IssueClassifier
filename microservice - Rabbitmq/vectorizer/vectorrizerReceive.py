#!/usr/bin/env python
import pika, sys, os
from sendToClassification import sendToClassificationQueue
from vectorizingLogic import createFeatureVector

def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='vectorrizerQueue')

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        print("vectorrizing: > {}".format(str(body)))
        sendToClassificationQueue(createFeatureVector([str(body)]))
        
    channel.basic_consume(queue='vectorrizerQueue', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages to vectorrize.')
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