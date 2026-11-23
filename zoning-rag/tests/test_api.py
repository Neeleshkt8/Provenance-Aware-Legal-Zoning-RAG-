import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from zoning_rag import api  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    api.settings.sqlite_path = tmp_path_factory.mktemp("api") / "api.db"
    api.settings.llm_providers = ["extractive"]
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["chunks"] > 0


def test_ask_logs_and_feedback(client):
    r = client.post("/ask", json={"question": "What is floor area ratio?"}).json()
    assert not r["abstained"] and r["citations"]
    assert client.post("/feedback", json={"query_id": r["query_id"], "rating": 1}).status_code == 200
    assert client.get("/history").json()[0]["id"] == r["query_id"]


def test_ask_abstains(client):
    assert client.post("/ask", json={"question": "What is the capital of France?"}).json()["abstained"]


def test_source_and_agent(client):
    s = client.get("/source", params={"file": "05_administration.txt", "start": 4, "end": 5}).json()
    assert "72-21" in s["text"]
    a = client.post("/agent", json={"question": "What is an MX district?"}).json()
    assert a["agent"]["steps"]


def test_source_rejects_traversal(client):
    assert client.get("/source", params={"file": "../../etc/passwd", "start": 1, "end": 2}).status_code == 404
