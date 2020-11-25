#!/bin/bash

sleep 15
celery -A microservice.classifier_celery.celery worker -l INFO -P prefork -Q vectorise_queue -n vectoriser@%n