import os
from flask import Request, abort
from werkzeug.exceptions import abort, HTTPException


class AuthMiddleware(object):

    def __init__(self, app) -> None:
        self.app = app

    def __call__(self, environ, start_response):
        try:
            request = Request(environ)
            api_key = request.headers.get('x-api-key', '')

            if  api_key != os.getenv('PRIVATE_KEY'):
                abort(401)

            return self.app(environ, start_response)
        except HTTPException as e:
            return e(environ, start_response)