FROM python:3.9-rc-buster

ARG POETRY_VERSION=1.0.10
ENV CODE_DIR=/code
ENV WEB_SERVER_DIR=${CODE_DIR}/web_server

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir 'poetry==$POETRY_VERSION'

WORKDIR ${CODE_DIR}

# Copy entrypoint to image & make it executable
COPY entrypoint.sh ${CODE_DIR}/
RUN ["chmod", "+x", "/entrypoint.sh"]

# Install server dependencies
COPY poetry.lock pyproject.toml ${CODE_DIR}/
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy server files
COPY web_server/ .

ENTRYPOINT /entrypoint.sh "${WEB_SERVER_DIR}}"