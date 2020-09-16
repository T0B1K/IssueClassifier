#!/bin/sh

# The first argument corresponds to the project directory in the Docker container
cd $1
uvicorn server:app --reload