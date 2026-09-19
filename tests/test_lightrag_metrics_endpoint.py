import sys
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_orig_argv = list(sys.argv)
sys.argv = ["lightrag"]
try:
    from lightrag.api.config import initialize_config
    initialize_config()
    from lightrag.api.lightrag_server import get_metrics_response
finally:
    sys.argv = _orig_argv


def test_get_metrics_response_directly():
    response = get_metrics_response()
    assert response.status_code == 200
    assert "text/plain" in response.media_type
    text = response.body.decode("utf-8")
    assert "lightrag_query_duration_seconds" in text


def test_metrics_endpoint_mounted():
    app = FastAPI()

    @app.get("/metrics")
    def metrics():
        return get_metrics_response()

    client = TestClient(app)
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "lightrag_query_duration_seconds" in res.text
