"""Extractive QA web service implementing docs/brief/qa_api_contract.yaml.

    GET  /health   public     200 {"status": "ok"} | 503 {"status": "loading"}
    GET  /info     protected  200 {"team", "model", "params", "max_seq_length", "doc_stride"}
    POST /predict  protected  200 {"id", "answer", "start", "score"}

Run (one worker: the model lives in this process' memory):
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.settings import Settings, get_settings
from qa.inference import QAModel

log = logging.getLogger("qa-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROTECTED_PATHS = {"/info", "/predict"}


# ---------------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    # extra="forbid": the contract sets additionalProperties: false.
    # strict=True: no silent type coercion (an integer id is rejected, not turned into "17").
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=5000)
    question: str = Field(min_length=1, max_length=500)


def error(status: int, code: str, detail: str | None = None, headers: dict | None = None) -> JSONResponse:
    body = {"error": code}
    if detail:
        body["detail"] = detail
    return JSONResponse(status_code=status, content=body, headers=headers)


# ---------------------------------------------------------------------------- model state
class ModelState:
    """Holds the model. Loaded in a background thread so /health can answer 503 meanwhile."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.qa: QAModel | None = None
        self.error: str | None = None
        self.lock = threading.Lock()   # one inference at a time: avoids CPU thread oversubscription

    @property
    def ready(self) -> bool:
        return self.qa is not None

    def load(self) -> None:
        try:
            t0 = time.time()
            if self.settings.num_threads:
                torch.set_num_threads(self.settings.num_threads)
            qa = QAModel.load(self.settings.model_dir, device="cpu")
            qa.predict_one("What is this?", "This is a warm-up request.")   # first call is slower
            self.qa = qa
            log.info("Model loaded from %s in %.1fs (%d params, threshold=%.3f, threads=%d)",
                     self.settings.model_dir, time.time() - t0, qa.num_parameters,
                     qa.config.null_threshold, torch.get_num_threads())
        except Exception as exc:  # keep the process alive; /health keeps reporting "loading"
            self.error = repr(exc)
            log.exception("Model failed to load")


# ---------------------------------------------------------------------------- app
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()   # raises if QA_API_TOKEN is missing -> fail fast
    state = ModelState(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        threading.Thread(target=state.load, name="model-loader", daemon=True).start()
        yield

    # No /docs, /redoc or /openapi.json: the URL is public, expose only the contract.
    app = FastAPI(title="Extractive QA service", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.model = state

    # ---- access control: runs BEFORE routing and body parsing, for every protected path
    @app.middleware("http")
    async def bearer_auth(request: Request, call_next):
        if request.url.path.rstrip("/") in PROTECTED_PATHS:
            header = request.headers.get("authorization", "")
            scheme, _, token = header.partition(" ")
            # Exact, case-sensitive scheme and token; constant-time comparison of the token.
            ok = scheme == "Bearer" and hmac.compare_digest(token.encode(), settings.api_token.encode())
            if not ok:
                return error(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    # ---- error bodies always follow the contract: {"error": ..., "detail": ...}
    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        errs = exc.errors()
        detail = "invalid request body"
        if errs:
            e = errs[0]
            field = ".".join(str(x) for x in e.get("loc", ()) if x != "body")
            if e.get("type") == "missing":
                detail = f"field '{field}' is required"
            elif e.get("type") == "json_invalid":
                detail = "body is not valid JSON"
            elif field:
                detail = f"field '{field}': {e.get('msg')}"
            else:
                detail = e.get("msg", detail)
        return error(400, "bad_request", detail)

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException):
        code = "unauthorized" if exc.status_code == 401 else "internal_error" if exc.status_code >= 500 else "bad_request"
        return error(exc.status_code, code, str(exc.detail) if exc.detail else None)

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        log.exception("Unhandled error")
        return error(500, "internal_error")

    # ---- endpoints
    @app.get("/health")
    def health():
        if state.ready:
            return {"status": "ok"}
        return JSONResponse(status_code=503, content={"status": "loading"})

    @app.get("/info")
    def info():
        if not state.ready:
            return error(503, "internal_error", "model is still loading")
        qa = state.qa
        return {
            "team": settings.team,
            "model": qa.meta.get("base_model", "unknown"),
            "params": qa.num_parameters,
            "max_seq_length": qa.config.max_seq_length,
            "doc_stride": qa.config.doc_stride,
        }

    # `def` (not `async def`): FastAPI runs it in a worker thread, so the event loop
    # keeps answering /health while the model computes.
    @app.post("/predict")
    def predict(req: PredictRequest):
        if not state.ready:
            return error(503, "internal_error", "model is still loading")
        t0 = time.perf_counter()
        try:
            with state.lock:
                pred = state.qa.predict_one(req.question, req.context)
        except Exception:
            log.exception("Inference failed for id=%s", req.id)
            return error(500, "internal_error")

        answer, start = pred.answer, pred.start
        # Contract invariant. Cannot fail by construction, but never send a broken pair.
        if answer is not None and (start is None or req.context[start:start + len(answer)] != answer):
            log.error("Offset invariant broken for id=%s; returning null", req.id)
            answer, start = None, None

        ms = (time.perf_counter() - t0) * 1000
        log.info("predict id=%s windows=%d diff=%.3f answer=%r %.0fms",
                 req.id, pred.num_windows, pred.score_diff, answer, ms)
        return JSONResponse(
            content={"id": req.id, "answer": answer, "start": start,
                     "score": round(min(max(pred.score, 0.0), 1.0), 4)},
            headers={"X-Inference-Ms": f"{ms:.0f}"},
        )

    return app


def __getattr__(name: str):
    # `uvicorn app.main:app` builds the app on first access, so importing this module
    # in tests does not require QA_API_TOKEN.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
