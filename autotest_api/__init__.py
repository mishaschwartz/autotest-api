"""
API for the autotester
"""

import os
import sys
import rq
import json
import io
from functools import wraps
import base64
import traceback
import dotenv
import redis
from datetime import datetime
from contextlib import contextmanager
from flask import Flask, request, jsonify, abort, make_response, send_file, Response
from werkzeug.exceptions import HTTPException
from typing import IO, Tuple, Optional, Union, Sequence, Iterable, Callable, Any, Dict

from . import form_management

DOTENVFILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
dotenv.load_dotenv(dotenv_path=DOTENVFILE)

ERROR_LOG = os.environ.get("ERROR_LOG")
ACCESS_LOG = os.environ.get("ACCESS_LOG")
SETTINGS_JOB_TIMEOUT = os.environ.get("SETTINGS_JOB_TIMEOUT", 600)
REDIS_URL = os.environ["REDIS_URL"]

app = Flask(__name__)

ID_TYPE = Union[int, str]


@contextmanager
def _open_log(log: str, mode: str = "a", fallback: IO = sys.stdout) -> IO:
    """
    Open the file at log with mode str. If log is not specified, then return
    fallback instead.
    """
    if log:
        with open(log, mode) as f:
            yield f
    else:
        yield fallback


def _redis_connection(**kwargs) -> redis.Redis:
    """
    Return a connection to the redis database at REDIS_URL.

    Passes kwargs on to the redis.Redis.from_url function except that the
    default for the decode_responses kwarg is True instead of False.
    """
    kwargs["decode_responses"] = kwargs.get("decode_responses", True)
    return redis.Redis.from_url(REDIS_URL, **kwargs)


def _rq_connection() -> redis.Redis:
    """
    Return the currently open redis connection object. If there is no
    connection currently open, one is created using the url specified in
    REDIS_URL.
    """
    conn = rq.get_current_connection()
    if conn:
        return conn
    rq.use_connection(redis=redis.Redis.from_url(REDIS_URL))
    return rq.get_current_connection()


@app.errorhandler(Exception)
def _handle_error(e) -> Tuple[Response, Optional[int]]:
    """
    Handle errors by logging the result and returing a 500 response
    """
    code = 500
    error = str(e)
    if isinstance(e, HTTPException):
        code = e.code
    with _open_log(ERROR_LOG, fallback=sys.stderr) as f:
        api_key = request.headers.get("Api-Key") or "ERROR: user not found"
        f.write(f"{datetime.now()}\n\tuser: {api_key}\n\t{traceback.format_exc()}\n")
        f.flush()
    if not app.debug:
        error = str(e).replace(api_key, "[client-api-key]")
    return jsonify(message=error), code


def _check_rate_limit(api_key: str) -> None:
    """
    Check if the user with api key api_key has exceeded their rate limit (the number
    of requests they can make per minute).

    The limit is set for each user in the redis database. If no limit is set for a given
    user, the user will not be limited in the number of requests they can make per minute.
    """
    conn = _redis_connection()
    key = f"autotest:ratelimit:{api_key}:{datetime.now().minute}"
    n_requests = conn.get(key) or 0
    user_limit = conn.get(f"autotest:ratelimit:{api_key}:limit")
    if user_limit is None:
        return
    if int(n_requests) > int(user_limit):
        abort(make_response(jsonify(message="Too many requests"), 429))
    else:
        with conn.pipeline() as pipe:
            pipe.incr(key)
            pipe.expire(key, 59)
            pipe.execute()


def _authorize_user() -> str:
    """
    Check if a user exists with the api key specified in the request header and return the
    api key if the user exists.
    """
    api_key = request.headers.get("Api-Key")
    user_name = (_redis_connection().hgetall("autotest:user_credentials") or {}).get(api_key)
    if user_name is None:
        abort(make_response(jsonify(message="Unauthorized"), 401))
    _check_rate_limit(api_key)
    return api_key


def _authorize_settings(user: str, settings_id: Optional[ID_TYPE] = None, **_kw) -> None:
    """
    Check if the settings with id settings_id belong to the given user.

    This function is a no-op if settings_id is None.
    """
    if settings_id:
        settings_ = _redis_connection().hget("autotest:settings", settings_id)
        if settings_ is None:
            abort(make_response(jsonify(message="Settings not found"), 404))
        if json.loads(settings_).get("_user") != user:
            abort(make_response(jsonify(message="Unauthorized"), 401))


def _authorize_tests(tests_id: Optional[ID_TYPE] = None, settings_id: Optional[ID_TYPE] = None, **_kw) -> None:
    """
    Check if the test with id tests_id is associated with the settings with id settings_id.

    This function is a no-op if either settings_id or tests_id is None.
    """
    if settings_id and tests_id:
        test_setting = _redis_connection().hget("autotest:tests", tests_id)
        if test_setting is None:
            abort(make_response(jsonify(message="Test not found"), 404))
        if test_setting != settings_id:
            abort(make_response(jsonify(message="Unauthorized"), 401))


def _update_settings(settings_id: ID_TYPE, user: str) -> None:
    """
    Enqueue a job to update the test settings and environment for the settings with id settings_id.

    The job will be run by the autotester backend.

    Expected params:

    - settings: a hash representing the tester settings (these will be validated against the schema)
    - file_url: a URL where test files can be downloaded from
    - files: a list of file names that can be downloaded from file_url
             NOTE: this is required because the settings need this list of files to validate against the schema
                   before the files are actually downloaded from the given URL
    """
    job_id = f"settings_{settings_id}"
    if (job := rq.job.Job.fetch(job_id, connection=_rq_connection())) and job.get_status() in ("queued", "started"):
        msg = "Settings are currently being updated. Please try again in a few minutes."
        abort(make_response(jsonify(message=msg), 409))
    test_settings = request.json.get("settings") or {}
    file_url = request.json.get("file_url")
    test_files = request.json.get("files") or []
    for filename in test_files:
        split_path = filename.split(os.path.sep)
        if ".." in split_path:
            raise Exception(".. not allowed in uploaded file path")
        if os.path.isabs(filename):
            raise Exception("uploaded files cannot include an absolute path")
    error = form_management.validate_against_schema(test_settings, schema(), test_files)
    if error:
        abort(make_response(jsonify(message=error), 422))

    queue = rq.Queue("settings", connection=_rq_connection())
    data = {"user": user, "settings_id": settings_id, "test_settings": test_settings, "file_url": file_url}
    queue.enqueue_call(
        "autotest_server.update_test_settings",
        kwargs=data,
        job_id=job_id,
        timeout=SETTINGS_JOB_TIMEOUT,
    )


def _get_jobs(test_ids: Sequence[ID_TYPE], settings_id: ID_TYPE) -> Iterable[Tuple[ID_TYPE, Optional[rq.job.Job]]]:
    """
    Yield each test_id and an associated rq Job that is responsible for running the test.
    If there is no associated job, yield the test_id and None instead.
    """
    for id_, job in zip(test_ids, rq.job.Job.fetch_many(test_ids, connection=_rq_connection())):
        if job is None or _redis_connection().hget("autotest:tests", id_) != settings_id:
            yield id_, None
        else:
            yield id_, job


def authorize(func: Callable) -> Callable:
    """
    Defines a decorator that checks whether the user that is making the request (based on the
    api key provided in the header) is authorized to access the resources (settings and test)
    accessed by the request function.

    If the user is authorized, the function is authorized and will be called, otherwise an
    error will be raised. In both cases the result is logged.
    """
    # non-secure authorization
    @wraps(func)
    def _f(*args, **kwargs):
        user = None
        log_msg = None
        try:
            user = _authorize_user()
            _authorize_settings(**kwargs, user=user)
            _authorize_tests(**kwargs)
            log_msg = f"AUTHORIZED\n\t{datetime.now()}\n\turl: {request.url}\n\tuser: {user}\n"
        except HTTPException as e:
            log_msg = (
                f"UNAUTHORIZED\n\t{datetime.now()}\n\t"
                f"url: {request.url}\n\tuser: {user}\n\tresponse: {e.get_response().response}\n"
            )
            raise e
        finally:
            if log_msg:
                with _open_log(ACCESS_LOG) as f:
                    f.write(log_msg)
                    f.flush()
        return func(*args, **kwargs, user=user)

    return _f


def _filter_private_keys(dictionary: Any) -> Any:
    """
    Return a deep copy of dictionary with all keys that start with "_" removed.
    """
    if isinstance(dictionary, dict):
        return {k: _filter_private_keys(v) for k, v in dictionary.items() if not k.startswith("_")}
    else:
        return dictionary


def get_settings(settings_id: ID_TYPE, show_hidden: bool = False) -> Dict:
    """
    Return a dictionary representation of the settings associated with settings_id.

    If show_hidden is false, filter out all hidden keys (those that start with "_").
    """
    settings_ = json.loads(_redis_connection().hget("autotest:settings", key=settings_id) or "{}")
    if settings_.get("_error"):
        raise Exception(f"Settings Error: {settings_['_error']}")
    if show_hidden:
        return settings_
    else:
        return _filter_private_keys(settings_)


@app.route("/register", methods=["POST"])
def register() -> Dict:
    """
    Register a new user and respond with the newly generated api key
    """
    # non-secure registration
    auth_type = request.json.get("auth_type")
    credentials = request.json.get("credentials")
    key = base64.b64encode(os.urandom(24)).decode("utf-8")
    data = {"auth_type": auth_type, "credentials": credentials}
    while not _redis_connection().hsetnx("autotest:user_credentials", key=key, value=json.dumps(data)):
        key = base64.b64encode(os.urandom(24)).decode("utf-8")
    return {"api_key": key}


@app.route("/reset_credentials", methods=["PUT"])
@authorize
def reset_credentials(user: str) -> Response:
    """
    Reset the auth_type and credentials for user
    """
    auth_type = request.json.get("auth_type")
    credentials = request.json.get("credentials")
    data = {"auth_type": auth_type, "credentials": credentials}
    _redis_connection().hset("autotest:user_credentials", key=user, value=json.dumps(data))
    return jsonify(success=True)


@app.route("/schema", methods=["GET"])
@authorize
def schema(**_kwargs) -> Dict:
    """
    Return the schema stored in the redis database
    """
    return json.loads(_redis_connection().get("autotest:schema") or "{}")


@app.route("/settings/<settings_id>", methods=["GET"])
@authorize
def settings(settings_id: ID_TYPE, **_kw) -> Dict:
    """
    Return the settings associated with settings_id
    """
    return get_settings(settings_id)


@app.route("/settings", methods=["POST"])
@authorize
def create_settings(user: str) -> Dict:
    """
    Create a new settings object stored in redis and return the associated id.

    See _update_settings for more details
    """
    settings_id = _redis_connection().incr("autotest:settings_id")
    _redis_connection().hset("autotest:settings", key=settings_id, value=json.dumps({"_user": user}))
    _update_settings(settings_id, user)
    return {"settings_id": settings_id}


@app.route("/settings/<settings_id>", methods=["PUT"])
@authorize
def update_settings(settings_id: ID_TYPE, user: str) -> Dict:
    """
    Update the settings associated with settings_id.

    See _update_settings for more details
    """
    _update_settings(settings_id, user)
    return {"settings_id": settings_id}


@app.route("/settings/<settings_id>/test", methods=["PUT"])
@authorize
def run_tests(settings_id: ID_TYPE, user: str) -> Dict:
    """
    Enqueue a job to run tests against the environment created for the settings with id == settings_id.

    The job will be run by the autotester backend.

    Expected params:

    - test_data: a hash representing the test data
    - categories: the categories (defined in the settings) to run these tests against
    - request_high_priority: a boolean indicating whether these tests should be run in a higher priority queue
                             (this is ignored if more than one test is sent at the same time)
    """
    test_data = request.json["test_data"]
    categories = request.json["categories"]
    high_priority = request.json.get("request_high_priority")
    queue_name = "batch" if len(test_data) > 1 else ("high" if high_priority else "low")
    queue = rq.Queue(queue_name, connection=_rq_connection())

    timeout = 0

    for settings_ in get_settings(settings_id, show_hidden=True)["testers"]:
        for data in settings_["test_data"]:
            if set(data["categories"]) & set(categories):
                timeout += data["timeout"]

    ids = []
    for data in test_data:
        url = data["file_url"]
        test_env_vars = data.get("env_vars", {})
        id_ = _redis_connection().incr("autotest:tests_id")
        _redis_connection().hset("autotest:tests", key=id_, value=settings_id)
        ids.append(id_)
        data = {
            "settings_id": settings_id,
            "test_id": id_,
            "files_url": url,
            "categories": categories,
            "user": user,
            "test_env_vars": test_env_vars,
        }
        queue.enqueue_call(
            "autotest_server.run_test",
            kwargs=data,
            job_id=str(id_),
            timeout=int(timeout * 1.5),
            failure_ttl=3600,
            result_ttl=3600,
        )  # TODO: make this configurable

    return {"test_ids": ids}


@app.route("/settings/<settings_id>/test/<tests_id>", methods=["GET"])
@authorize
def get_result(settings_id: ID_TYPE, tests_id: ID_TYPE, **_kw) -> Dict:
    """
    Return the result for the test with id tests_id
    """
    job = rq.job.Job.fetch(tests_id, connection=_rq_connection())
    job_status = job.get_status()
    result = {"status": job_status}
    if job_status == "finished":
        test_result = _redis_connection().get(f"autotest:test_result:{tests_id}")
        try:
            result.update(json.loads(test_result))
        except json.JSONDecodeError:
            result.update({"error": f"invalid json: {test_result}"})
    elif job_status == "failed":
        result.update({"error": str(job.exc_info)})
    job.delete()
    _redis_connection().delete(f"autotest:test_result:{tests_id}")
    return result


@app.route("/settings/<settings_id>/test/<tests_id>/feedback/<feedback_id>", methods=["GET"])
@authorize
def get_feedback_file(settings_id: ID_TYPE, tests_id: ID_TYPE, feedback_id: ID_TYPE, **_kw) -> Response:
    """
    Return the feedback file with id == feedback_id (in a response)
    """
    key = f"autotest:feedback_file:{tests_id}:{feedback_id}"
    data = _redis_connection(decode_responses=False).get(key)
    if data is None:
        abort(make_response(jsonify(message="File doesn't exist"), 404))
    _redis_connection().delete(key)
    return send_file(io.BytesIO(data), mimetype="application/gzip", as_attachment=True, download_name=str(feedback_id))


@app.route("/settings/<settings_id>/tests/status", methods=["GET"])
@authorize
def get_statuses(settings_id: ID_TYPE, **_kw) -> Dict:
    """
    Return a dictionary containing the statuses of all tests associated with settings_id that are still in the
    redis database.
    """
    test_ids = request.json["test_ids"]
    result = {}
    for id_, job in _get_jobs(test_ids, settings_id):
        result[id_] = job if job is None else job.get_status()
    return result


@app.route("/settings/<settings_id>/tests/cancel", methods=["DELETE"])
@authorize
def cancel_tests(settings_id: ID_TYPE, **_kw) -> Response:
    """
    Cancel tests with ids set in the "test_ids" request parameter.

    If a given job does not exist with that id or the test is not associated with
    the setting with id == settings_id then that test is skipped.
    """
    test_ids = request.json["test_ids"]
    for _, job in _get_jobs(test_ids, settings_id):
        if job is not None:
            job.cancel()
    return jsonify(success=True)
