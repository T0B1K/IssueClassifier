#!/bin/bash

celery -A microservice.celery_app worker -l INFO -Q classify_queue -n classifier-worker@%h