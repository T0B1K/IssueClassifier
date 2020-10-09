FROM python:3.9-rc-buster

ARG CURRENT_ENV=production

ENV POETRY_VERSION=1.0.10 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    CURRENT_ENV=${CURRENT_ENV} \
    FLASK_APP="/web_server/web_server/server.py"

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock /web_server/

COPY entrypoint.sh /web_server/

RUN ["chmod", "+x", "/web_server/entrypoint.sh"]

RUN poetry config virtualenvs.create false && \
    poetry install $(test "${CURRENT_ENV}" == production && echo "--no-dev") --no-interaction --no-ansi

ADD web_server /web_server/web_server

CMD ["flask" , "run"]