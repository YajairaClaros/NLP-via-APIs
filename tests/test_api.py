"""Contract tests for the web service (docs/brief/qa_api_contract.yaml), using a tiny model."""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings

TOKEN = "test-token-123"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SAMPLE = json.loads((Path(__file__).resolve().parents[1] / "docs/brief/public_sample.json").read_text())["items"]


@pytest.fixture(scope="module")
def client(tiny_model_dir):
    app = create_app(Settings(api_token=TOKEN, team="team-test", model_dir=Path(tiny_model_dir), num_threads=1))
    with TestClient(app) as c:
        for _ in range(200):                      # wait for the background loader
            if c.get("/health").status_code == 200:
                break
            time.sleep(0.05)
        yield c


def body(item):
    return {"id": item["id"], "context": item["context"], "question": item["question"]}


# ---------------------------------------------------------------- health / access control
def test_health_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [
    {},                                             # no token
    {"Authorization": "Bearer wrong"},              # wrong token
    {"Authorization": f"bearer {TOKEN}"},           # scheme is case-sensitive
    {"Authorization": f"Bearer {TOKEN.upper()}"},   # token is case-sensitive
    {"Authorization": f"Bearer  {TOKEN}"},          # extra space
    {"Authorization": TOKEN},                       # no scheme
    {"Authorization": f"Basic {TOKEN}"},
])
@pytest.mark.parametrize("method,path", [("get", "/info"), ("post", "/predict")])
def test_unauthorized(client, headers, method, path):
    kwargs = {"json": body(SAMPLE[0])} if method == "post" else {}
    r = getattr(client, method)(path, headers=headers, **kwargs)
    assert r.status_code == 401
    assert r.json() == {"error": "unauthorized"}
    assert r.headers.get("www-authenticate") == "Bearer"


def test_auth_checked_before_body(client):
    r = client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 401


def test_info(client):
    r = client.get("/info", headers=AUTH)
    assert r.status_code == 200
    d = r.json()
    assert d["team"] == "team-test" and " " not in d["team"]
    assert isinstance(d["model"], str) and isinstance(d["params"], int) and d["params"] > 0
    assert isinstance(d["max_seq_length"], int) and isinstance(d["doc_stride"], int)


# ---------------------------------------------------------------- /predict
@pytest.mark.parametrize("item", SAMPLE, ids=[i["id"] for i in SAMPLE])
def test_predict_contract(client, item):
    r = client.post("/predict", headers=AUTH, json=body(item))
    assert r.status_code == 200
    d = r.json()
    assert set(d) >= {"id", "answer", "start", "score"}
    assert d["id"] == item["id"]
    assert 0.0 <= d["score"] <= 1.0
    assert (d["answer"] is None) == (d["start"] is None)
    if d["answer"] is not None:
        assert item["context"][d["start"]: d["start"] + len(d["answer"])] == d["answer"]


def test_unicode_context(client):
    ctx = "Café señor 😀 — the Lempa River is the longest river in El Salvador, flowing about 422 km. " * 20
    r = client.post("/predict", headers=AUTH, json={"id": "u-1", "context": ctx, "question": "How long is the Lempa River?"})
    d = r.json()
    assert r.status_code == 200
    if d["answer"] is not None:
        assert ctx[d["start"]: d["start"] + len(d["answer"])] == d["answer"]


@pytest.mark.parametrize("payload,needle", [
    ({"id": "x", "context": "abc"}, "question"),                                   # missing field
    ({"id": "", "context": "abc", "question": "q"}, "id"),                          # too short
    ({"id": "x" * 65, "context": "abc", "question": "q"}, "id"),                    # too long
    ({"id": "x", "context": "a" * 5001, "question": "q"}, "context"),
    ({"id": "x", "context": "abc", "question": "q" * 501}, "question"),
    ({"id": 17, "context": "abc", "question": "q"}, "id"),                          # wrong type
    ({"id": "x", "context": "abc", "question": "q", "extra": 1}, "extra"),          # additionalProperties: false
])
def test_bad_request(client, payload, needle):
    r = client.post("/predict", headers=AUTH, json=payload)
    assert r.status_code in (400, 422)
    d = r.json()
    assert d["error"] == "bad_request" and needle in d.get("detail", "")


def test_invalid_json(client):
    r = client.post("/predict", headers={**AUTH, "Content-Type": "application/json"}, content=b"{not json")
    assert r.status_code in (400, 422) and r.json()["error"] == "bad_request"


def test_array_body(client):
    r = client.post("/predict", headers=AUTH, json=[body(SAMPLE[0])])
    assert r.status_code in (400, 422) and r.json()["error"] == "bad_request"


def test_limits_accepted(client):
    r = client.post("/predict", headers=AUTH, json={"id": "x" * 64, "context": "a " * 2500, "question": "q" * 500})
    assert r.status_code == 200 and r.json()["id"] == "x" * 64


def test_no_docs_exposed(client):
    for p in ("/docs", "/openapi.json", "/redoc"):
        assert client.get(p).status_code == 404


def test_missing_token_env_fails_fast(monkeypatch):
    monkeypatch.delenv("QA_API_TOKEN", raising=False)
    from app.settings import get_settings
    with pytest.raises(RuntimeError):
        get_settings()
