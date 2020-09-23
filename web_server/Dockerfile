FROM python:3.9-rc-buster

RUN pip install pipenv

ENV PROJECT_DIR=/code

COPY entrypoint.sh .

RUN ["chmod", "+x", "/entrypoint.sh"]

WORKDIR ${PROJECT_DIR}

COPY Pipfile Pipfile.lock ${PROJECT_DIR}/

RUN pipenv install --system --deploy --ignore-pipfile

COPY src/ .

ENTRYPOINT /entrypoint.sh "${PROJECT_DIR}"