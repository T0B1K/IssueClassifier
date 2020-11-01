FROM python:3.8

WORKDIR /usr/src/app/issues
COPY ./issues .

WORKDIR /usr/src/app/classifier

COPY ./classifier/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./classifier .

CMD ["python", "./train.py"]
