#!/bin/bash

celery -A microservice.celery_app worker -l INFO -Q vectorise_queue -n vectoriser-worker@%h