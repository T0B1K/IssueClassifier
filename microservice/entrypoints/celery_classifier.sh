#!/bin/bash

celery -A microservice.classifier_celery.celery worker -l INFO -P solo -Q classify_queue -n classifier-worker@%h