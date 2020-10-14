#!/bin/bash

set -e

cd /microservice/
celery --app=microservice.celery_app worker worker -l INFO -Q classify_queue -n classifier-worker@%h