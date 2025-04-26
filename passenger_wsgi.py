from fastapi_app.controller import app
from fastapi.middleware.wsgi import WSGIMiddleware

application = WSGIMiddleware(app)