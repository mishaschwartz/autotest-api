# autotest-api

This repo contains the API for the Autotester project.

## System requirements

- [python3.8+](https://www.python.org/)
- [redis-server](https://redis.io/)

## Installation

Install the required python packages using pip

```sh
python3 -m pip install -r requirements.txt
```

Ensure that a [redis server](https://redis.io/) is started and running. 

## Configuration

The API can be configured by updating the `.env` file. Since the API is a [Flask](https://flask.palletsprojects.com/en/2.0.x/) 
application, you can also put any environment variables required to configure a Flask application (if needed). 

Please see below for a description of all options and defaults:

```shell
REDIS_URL=  # url of the redis database (this should be the same url set for the autotest backend or else the two cannot communicate)
ACCESS_LOG= # file to write access log information to (default is stdout)
ERROR_LOG= # file to write error log informatoin to (default is stderr)
SETTINGS_JOB_TIMEOUT= # the maximum runtime (in seconds) of a job that updates settings before it is interrupted (default is 600) 
```

## Start up

### In development

```bash
python3 run.py
```

### In production

Run the API as you would any other simple [Flask application](https://flask.palletsprojects.com/en/2.0.x/)

For example, if you would like to run the API using [gunicorn](https://gunicorn.org/), you could download the 
code in this repo to a directory named `autotest-api` and start it with gunicorn:

```shell
gunicorn --chdir autotest-api --bind localhost:5000 run:app
```

and configure an httpd service (such as [apache](https://httpd.apache.org/) or [nginx](https://www.nginx.com/)) 
to proxy the local server that gunicorn is running.  
