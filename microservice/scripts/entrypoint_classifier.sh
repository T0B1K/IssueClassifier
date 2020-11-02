#!/bin/bash

celery -A microservice.ms_celery.celery worker -l INFO -Q classify_queue -n classifier-worker@%h