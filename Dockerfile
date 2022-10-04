FROM python:3.10-slim as base

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

FROM base as dev

CMD python3 run.py

FROM base as prod

COPY . .

CMD gunicorn --bind 0.0.0.0:5000 run:app