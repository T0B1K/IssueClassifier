#!/bin/bash

set -e

cd /microservice
celery --app=microservice.celery_app worker -l INFO -Q vectorise_queue -n vectoriser-worker@%h