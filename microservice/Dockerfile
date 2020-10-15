FROM python:3.8-buster

ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_VERSION=1.0.10

RUN pip install --upgrade pip
RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /microservice

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-interaction --no-ansi --no-root

COPY . .

RUN chmod -R +x ./scripts