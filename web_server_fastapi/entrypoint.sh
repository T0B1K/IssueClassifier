#!/bin/sh

# The first argument corresponds to the project directory in the Docker container
export FLASK_APP="$1/server.py"
flask run