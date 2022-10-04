FROM python:3.10-slim as base

WORKDIR /app

COPY requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt && rm requirements.txt

FROM base as dev

CMD python run.py

FROM base as prod

RUN python -m pip install gunicorn

COPY . .

CMD python -m gunicorn --bind 0.0.0.0:5000 run:app