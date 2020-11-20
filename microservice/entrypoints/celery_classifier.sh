#!/bin/bash

sleep 15
celery -A microservice.classifier_celery.celery worker -l INFO -P prefork -Q classify_queue -n classifier@%n