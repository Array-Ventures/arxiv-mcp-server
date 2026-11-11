"""Endpoint embeddings use no local model and never mix model generations."""
import json
import sqlite3

import httpx
import numpy as np
import pytest

from arxiv_mcp_server.tools import semantic_search as semantic


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(semantic.settings, "EMBEDDING_BASE_URL", "http://embedding.test/v1")
    monkeypatch.setattr(semantic.settings, "EMBEDDING_MODEL", "test-gemma")
    monkeypatch.setattr(semantic.settings, "EMBEDDING_API_KEY", "fake-secret-canary")
    monkeypatch.setattr(semantic.settings, "_get_storage_path_from_args", lambda: tmp_path)
    monkeypatch.setattr(semantic, "np", np)
    monkeypatch.setattr(semantic, "SentenceTransformer", None)
    return tmp_path


def response(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(semantic.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_endpoint_is_normalized_and_does_not_load_local_model(endpoint, monkeypatch):
    def handle(request):
        assert str(request.url) == "http://embedding.test/v1/embeddings"
        assert request.headers["authorization"] == "Bearer fake-secret-canary"
        assert json.loads(request.content) == {"model": "test-gemma", "input": ["paper"], "encoding_format": "float"}
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3, 4]}]})
    response(monkeypatch, handle)
    assert semantic._load_dependencies()
    assert semantic.SentenceTransformer is None
    np.testing.assert_allclose(semantic._embed_text("paper"), [0.6, 0.8])


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [[1, 2]], []])
def test_invalid_vectors_fail_without_provider_details(endpoint, monkeypatch, vector):
    response(monkeypatch, lambda _: httpx.Response(200, content=json.dumps({"data": [{"embedding": vector}]})))
    with pytest.raises(ValueError, match="^Embedding endpoint failed or returned invalid vectors.$"):
        semantic._embed_text("paper")


def test_redirect_and_error_bodies_are_not_exposed(endpoint, monkeypatch):
    response(monkeypatch, lambda _: httpx.Response(307, headers={"Location": "http://private.test"}, text="fake-secret-canary"))
    with pytest.raises(ValueError) as error:
        semantic._embed_text("paper")
    assert "canary" not in str(error.value)
    assert "private.test" not in str(error.value)


def test_configuration_change_requires_explicit_reindex(endpoint, monkeypatch):
    response(monkeypatch, lambda _: httpx.Response(200, json={"data": [{"embedding": [1, 0]}]}))
    assert semantic._upsert_index_record("2301.00001", "One", "abstract", [], [])
    monkeypatch.setattr(semantic.settings, "EMBEDDING_API_KEY", "rotated-key")
    assert len(semantic._load_vectors()) == 1
    monkeypatch.setattr(semantic.settings, "EMBEDDING_MODEL", "other-model")
    with pytest.raises(ValueError, match="reindex"):
        semantic._load_vectors()
    assert semantic.rebuild_index(clear_existing=True)["status"] == "success"
    assert semantic._load_vectors() == []


def test_legacy_index_is_not_assumed_to_use_endpoint_model(endpoint):
    with sqlite3.connect(endpoint / semantic.INDEX_DB_NAME) as conn:
        conn.execute("CREATE TABLE semantic_index (paper_id TEXT)")
        conn.execute("INSERT INTO semantic_index VALUES ('2301.00001')")
    with pytest.raises(ValueError, match="reindex"):
        semantic._connect()


def test_optional_http_bearer_guards_every_mcp_request(monkeypatch):
    from starlette.testclient import TestClient
    from arxiv_mcp_server import server

    class Manager:
        async def handle_request(self, scope, receive, send):
            from starlette.responses import PlainTextResponse
            await PlainTextResponse("ok")(scope, receive, send)

    monkeypatch.setattr(server.settings, "HTTP_BEARER_TOKEN", "test-mcp-token")
    with TestClient(server._build_http_app(Manager())) as client:
        for method in ["GET", "POST", "DELETE"]:
            assert client.request(method, "/mcp/", headers={"Authorization": "Bearer wrong"}).status_code == 401
            assert client.request(method, "/mcp/", headers={"Authorization": "Bearer test-mcp-token"}).status_code == 200
        assert client.get("/healthz").status_code == 200
