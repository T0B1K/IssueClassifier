#!/bin/bash

celery -A microservice.ms_celery.celery worker -l INFO -P solo -Q classify_queue -n classifier-worker@%h