"""Voice agent backend (v0.2 + v2 extensions).

WebSocket  /ws?mode=web            hands-free browser session (see session.py for the message protocol)
REST       /api/admin/...          settings page: persona, knowledge base upload / train / rollback, unanswered log
REST       /api/health, /api/public/settings

v2 extensions:
- Redis-backed hot session memory (REDIS_URL) with in-process fallback
- SQLite durable conversation/turn/contact/unanswered store (SQLITE_DB)
- Hybrid RAG: BM25 + dense + RRF, optional cross-encoder re-ranker, query expansion
- Capability/tool scaffold + guardrails
- Observability: structured logs, metrics, conversation export
"""
import asyncio
import csv
import io
import json
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import store
from engines import LLM, STT, TTS
from kb import KnowledgeBase, ParseError
from session import ENGINES, Session, system_prompt
from redis_session import RedisSessionStore
from db import DB
from logging_utils import setup_logging

# Import Twilio router
from twilio_router import router as twilio_router

if config.STRUCTURED_LOGS:
    setup_logging()

_log = logging.getLogger("app")
redis_store = RedisSessionStore()
vet_db = DB()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    _log.info("startup device=%s mock=%s", config.DEVICE, config.MOCK)
    ENGINES["stt"] = STT()
    ENGINES["kb"] = KnowledgeBase()
    ENGINES["tts"] = TTS()
    ENGINES["llm"] = LLM()
    await ENGINES["llm"].warmup()
    engine = store.load_settings()["stt_engine"]
    await asyncio.get_running_loop().run_in_executor(None, ENGINES["stt"].preload, engine)
    keepalive_task = asyncio.create_task(_keep_llm_loaded())
    kb = ENGINES["kb"]
    _log.info("models ready in %.1fs, knowledge base %s with %d chunks",
              time.perf_counter() - t0, kb.version, len(kb.chunks))
    _log.info("settings page: /#/settings   admin password: %s", store.admin_password())

    # v2: redis session store
    ENGINES["redis_session"] = redis_store
    await redis_store.open()

    # v2: durable store
    ENGINES["db"] = vet_db
    db_ok = await vet_db.open()
    _log.info("v2 db ok=%s", db_ok)
    yield
    # teardown
    keepalive_task.cancel()
    await redis_store.close()
    await vet_db.close()


app = FastAPI(title="Voice Agent", lifespan=lifespan)

# Include Twilio webhook routes
app.include_router(twilio_router, prefix="")


# ---------------------------------------------------------------- Twilio Media Streams WebSocket
# Twilio Media Streams connects here for real-time bidirectional audio.
# The twilio_router generates a stream_url pointing to this endpoint.

_media_stream_handler: object | None = None


def get_media_stream_handler() -> "TwilioMediaStreamHandler":
    global _media_stream_handler
    if _media_stream_handler is None:
        from livekit_worker import TwilioMediaStreamHandler
        _media_stream_handler = TwilioMediaStreamHandler()
    return _media_stream_handler


@app.websocket("/ws/twilio")
async def twilio_media_stream(ws: WebSocket, call_sid: str = Query("")):
    # call_sid is optional: Twilio connects to this route with no query string
    # (query parameters are not supported in <Stream url>). The real CallSid
    # arrives in Twilio's "start" message and the handler picks it up there.
    """
    Twilio Media Streams WebSocket endpoint.

    Twilio connects here after the /twilio/inbound webhook returns TwiML with a <Connect><Stream>.
    This handles real-time bidirectional audio: receives caller audio, processes through
    STT -> LLM -> TTS, and sends back agent audio.

    Query params:
        call_sid: The Twilio CallSid identifying this call
    """
    await ws.accept()
    _log.info("Twilio Media Stream connected: call_sid=%s", call_sid)

    handler = get_media_stream_handler()

    try:
        await handler.handle_websocket(ws, call_sid)
    except WebSocketDisconnect:
        _log.info("Twilio Media Stream disconnected: call_sid=%s", call_sid)
    except Exception:
        _log.exception("Twilio Media Stream error: call_sid=%s", call_sid)


async def _keep_llm_loaded():
    while True:
        await asyncio.sleep(600)
        await ENGINES["llm"].keep_loaded()


# ---------------------------------------------------------------- public

@app.get("/api/health")
async def health():
    s = store.load_settings()
    kb = ENGINES.get("kb")
    redis_ok = False
    redis_info = None
    if redis_store.enabled and redis_store.redis:
        try:
            redis_ok = await redis_store.redis.ping()
            redis_info = await redis_store.redis.info("server")
        except Exception as e:
            redis_ok = False
            _log.warning("redis health check failed: %s", e)
    db_ok = bool(vet_db.conn)
    metrics = {}
    if db_ok:
        try:
            metrics = await vet_db.aggregate_metrics()
        except Exception as e:
            _log.debug("metrics aggregate failed: %s", e)
    return {
        "ok": True,
        "mock": config.MOCK,
        "device": config.DEVICE,
        "stt": config.STT_MODEL,
        "embed": config.EMBED_MODEL,
        "llm": config.LLM_MODEL,
        "tts": f"kokoro/{s['voice']}",
        "stt_engine": s["stt_engine"],
        "resort": s["business_name"],
        "agent": s["agent_name"],
        "kb_version": kb.version if kb else None,
        "kb_chunks": len(kb.chunks) if kb else 0,
        "v2": {
            "redis_url_filled": bool(config.REDIS_URL),
            "redis_ok": redis_ok,
            "sqlite_ok": db_ok,
            "hybrid_rag": config.V2_RAG,
            "rerank": config.V2_RERANK,
            "metrics": metrics,
        },
    }


@app.get("/api/public/settings")
async def public_settings():
    s = store.load_settings()
    return {k: s[k] for k in ("agent_name", "business_name", "wake_required_web", "followup_seconds")}


# ---------------------------------------------------------------- live session

_active_ws: set[WebSocket] = set()


@app.websocket("/ws")
async def ws_session(ws: WebSocket, mode: str = Query("web"), key: str = Query("")):
    await ws.accept()
    if mode not in ("web", "api"):
        await ws.close(code=4400)
        return
    if mode == "api" and store.api_keys() and key not in store.api_keys():
        await ws.send_json({"type": "error", "message": "invalid API key"})
        await ws.close(code=4401)
        return
    if len(_active_ws) >= config.WS_MAX_SESSIONS:
        await ws.send_json({"type": "error", "message": "too many concurrent sessions"})
        await ws.close(code=4429)
        return
    _active_ws.add(ws)

    async def send(obj):
        await ws.send_text(json.dumps(obj))

    session = Session(send, mode=mode)
    s = store.load_settings()
    await send({"type": "ready", "session": session.id, "agent_name": s["agent_name"],
                "business_name": s["business_name"], "wake_required": session.wake})
    await session.send_state()
    tasks: set[asyncio.Task] = set()

    def spawn(coro):
        # background handlers must never die silently: log anything that crashes them
        async def _guard():
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                _log.exception("websocket handler crashed (mode=%s)", mode)
        t = asyncio.create_task(_guard())
        tasks.add(t)
        t.add_done_callback(tasks.discard)

    try:
        while True:
            try:
                # a connection that sends nothing for WS_IDLE_TIMEOUT_SEC is dead (callers
                # are kept alive by the browser heartbeat) and must not hold GPU models forever
                msg = await asyncio.wait_for(ws.receive(), timeout=config.WS_IDLE_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                _log.info("websocket idle timeout, closing (mode=%s)", mode)
                break
            if msg["type"] == "websocket.disconnect":
                break
            payload = msg.get("bytes") or msg.get("text") or ""
            if len(payload) > config.WS_MAX_MSG_BYTES:
                _log.warning("websocket frame too large (%d bytes), dropping", len(payload))
                continue
            if msg.get("bytes") is not None:
                spawn(session.on_audio(msg["bytes"]))
                continue
            try:
                data = json.loads(msg.get("text") or "{}")
            except json.JSONDecodeError:
                continue
            t = data.get("type")
            if t == "text" and data.get("text", "").strip():
                spawn(session.on_text(data["text"]))
            elif t == "progress":
                await session.on_progress(int(data.get("turn", 0)), int(data.get("index", 0)))
            elif t == "interrupt":
                await session.on_interrupt(int(data.get("turn", 0)))
            elif t == "idle":
                await session.on_idle(int(data.get("turn", 0)))
            elif t == "config":
                if "wake" in data and mode == "web":
                    session.wake = bool(data["wake"])
                if "capture_contact" in data:
                    session.capture_contact = bool(data["capture_contact"])
                await session.send_state()
            elif t == "reset":
                await session.reset()
            elif t == "ping":
                await send({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        _active_ws.discard(ws)
        await session.cancel_current()
        for t in list(tasks):
            t.cancel()


# ---------------------------------------------------------------- admin

def require_admin(authorization: str = Header("")):
    token = authorization.removeprefix("Bearer ").strip()
    if not store.check_token(token):
        raise HTTPException(401, "login required")


class Login(BaseModel):
    password: str


@app.post("/api/admin/login")
async def login(body: Login):
    if not store.check_password(body.password):
        await asyncio.sleep(1)
        raise HTTPException(401, "wrong password")
    return {"token": store.make_token()}


@app.get("/api/admin/settings", dependencies=[Depends(require_admin)])
async def get_settings():
    return {"settings": store.load_settings(), "voices": store.VOICES, "stt_engines": store.STT_ENGINES,
            "sarvam_key_set": bool(config.SARVAM_API_KEY)}


@app.put("/api/admin/settings", dependencies=[Depends(require_admin)])
async def put_settings(body: dict):
    before = store.load_settings()["stt_engine"]
    try:
        saved = store.save_settings(body)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # v2: audit settings change into SQLite when available
    if vet_db.conn:
        try:
            import json as _json
            changed_fields = {k: v for k, v in body.items() if k in saved}
            await vet_db.audit_settings("admin_api", changed_fields, saved)
        except Exception as e:
            _log.debug("settings audit skipped: %s", e)
    if saved["stt_engine"] != before:
        if saved["stt_engine"] == "sarvam" and not config.SARVAM_API_KEY:
            raise HTTPException(400, "Saved, but Sarvam needs SARVAM_API_KEY in .env (then restart).")
        # load the new model in the background (first time downloads it)
        asyncio.get_running_loop().run_in_executor(None, ENGINES["stt"].preload, saved["stt_engine"])
    return {"settings": saved}


@app.get("/api/admin/voice-sample", dependencies=[Depends(require_admin)])
async def voice_sample(voice: str, speed: float = 1.0):
    if voice not in store.VOICES:
        raise HTTPException(400, "unknown voice")
    s = store.load_settings()
    text = s["greeting"].format(agent=s["agent_name"], business=s["business_name"])
    loop = asyncio.get_running_loop()
    wav = await loop.run_in_executor(None, ENGINES["tts"].synthesize, text, voice, speed)
    import base64
    return {"wav_b64": base64.b64encode(wav).decode()}


@app.get("/api/admin/kb", dependencies=[Depends(require_admin)])
async def kb_status():
    kb = ENGINES["kb"]
    return {"active": kb.version, "chunks": len(kb.chunks), "staged": kb.staged_files(),
            "versions": list(reversed(kb.list_versions()))}


@app.post("/api/admin/kb/upload", dependencies=[Depends(require_admin)])
async def kb_upload(files: list[UploadFile] = File(...)):
    kb, out, errors = ENGINES["kb"], [], []
    for f in files:
        content = await f.read()
        if len(content) > config.MAX_UPLOAD_BYTES:
            errors.append(f"{f.filename}: file too large (max {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
            continue
        try:
            out.append(kb.stage(f.filename or "upload", content))
        except ParseError as e:
            errors.append(str(e))
    return {"staged": kb.staged_files(), "added": out, "errors": errors}


@app.delete("/api/admin/kb/staged/{name}", dependencies=[Depends(require_admin)])
async def kb_unstage(name: str):
    kb = ENGINES["kb"]
    kb.unstage(name)
    return {"staged": kb.staged_files()}


class TrainBody(BaseModel):
    note: str = ""


@app.post("/api/admin/kb/train", dependencies=[Depends(require_admin)])
async def kb_train(body: TrainBody):
    kb = ENGINES["kb"]
    loop = asyncio.get_running_loop()
    try:
        meta = await loop.run_in_executor(None, kb.train, body.note)
    except ParseError as e:
        raise HTTPException(400, str(e))
    return {"version": meta, "active": kb.version}


class Activate(BaseModel):
    version: str


@app.post("/api/admin/kb/activate", dependencies=[Depends(require_admin)])
async def kb_activate(body: Activate):
    kb = ENGINES["kb"]
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, kb.activate, body.version)
    except ParseError as e:
        raise HTTPException(404, str(e))
    return {"active": kb.version, "chunks": len(kb.chunks)}


@app.get("/api/admin/kb/chunks", dependencies=[Depends(require_admin)])
async def kb_chunks(q: str = "", offset: int = 0, limit: int = 50):
    kb = ENGINES["kb"]
    rows = kb.chunks
    if q:
        ql = q.lower()
        rows = [c for c in rows if ql in c.title.lower() or ql in c.text.lower()]
    page = rows[offset: offset + limit]
    return {"total": len(rows), "version": kb.version,
            "chunks": [{"id": c.id, "title": c.title, "text": c.text, "source": c.source} for c in page]}


class TestQ(BaseModel):
    question: str


@app.post("/api/admin/test", dependencies=[Depends(require_admin)])
async def test_question(body: TestQ):
    """Retrieval and the text answer, without speech. Shows what the agent would do."""
    s = store.load_settings()
    kb = ENGINES["kb"]
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    context = await loop.run_in_executor(None, kb.search, body.question)
    best = max((sc for _, sc in context), default=0.0)
    retrieved = [{"title": c.title, "text": c.text, "score": round(sc, 3), "source": c.source} for c, sc in context]
    if best < s["min_retrieval_score"]:
        return {"answer": s["fallback_text"], "unknown": True, "reason": "retrieval score below threshold",
                "best_score": round(best, 3), "retrieved": retrieved}
    messages = [{"role": "system", "content": system_prompt(s, context)},
                {"role": "user", "content": body.question}]
    answer = await ENGINES["llm"].complete(messages, max_tokens=config.LLM_MAX_TOKENS)
    unknown = "UNKNOWN" in answer.upper().replace(" ", "")[:20]
    return {"answer": s["fallback_text"] if unknown else answer, "unknown": unknown,
            "reason": "model said it does not know" if unknown else "",
            "best_score": round(best, 3), "retrieved": retrieved,
            "ms": round((time.perf_counter() - t0) * 1000)}


@app.get("/api/admin/unanswered", dependencies=[Depends(require_admin)])
async def unanswered():
    return {"items": store.list_unanswered()}


@app.patch("/api/admin/unanswered/{entry_id}", dependencies=[Depends(require_admin)])
async def unanswered_update(entry_id: str, body: dict):
    if not store.update_unanswered(entry_id, **body):
        raise HTTPException(404, "not found")
    return {"ok": True}


@app.get("/api/admin/unanswered.csv", dependencies=[Depends(require_admin)])
async def unanswered_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "question", "name", "phone", "note", "resolved", "mode"])
    for r in store.list_unanswered():
        w.writerow([r["time"], r["question"], r["name"], r["phone"], r["note"], r["resolved"], r["mode"]])
    return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=unanswered.csv"})


# ---------------------------------------------------------------- v2 observability

@app.get("/api/admin/metrics", dependencies=[Depends(require_admin)])
async def metrics():
    if vet_db.conn:
        try:
            return {"ok": True, "metrics": await vet_db.aggregate_metrics()}
        except Exception as e:
            raise HTTPException(500, f"metrics error: {e}")
    return {"ok": True, "metrics": {}, "note": "sqlite not enabled"}


@app.get("/api/admin/conversations", dependencies=[Depends(require_admin)])
async def conversations(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                        status: str | None = Query(None), q: str | None = Query(None),
                        mode: str | None = Query(None)):
    if not vet_db.conn:
        raise HTTPException(503, "sqlite not enabled")
    rows = await vet_db.list_conversations(limit=limit, offset=offset, status=status, q=q, mode=mode)
    return {"ok": True, "rows": rows, "limit": limit, "offset": offset}


@app.get("/api/admin/conversations/{conversation_id}/turns", dependencies=[Depends(require_admin)])
async def conversation_turns(conversation_id: str):
    if not vet_db.conn:
        raise HTTPException(503, "sqlite not enabled")
    conv = await vet_db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "conversation not found")
    turns = await vet_db.list_turns(conversation_id)
    contacts = await vet_db.list_contacts(conversation_id)
    return {"ok": True, "conversation": conv, "turns": turns, "contacts": contacts}


@app.patch("/api/admin/conversations/{conversation_id}/note", dependencies=[Depends(require_admin)])
async def conversation_note(conversation_id: str, body: dict):
    if not vet_db.conn:
        raise HTTPException(503, "sqlite not enabled")
    conv = await vet_db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "conversation not found")
    note = body.get("note", "")
    summary = body.get("summary", None)
    if not isinstance(note, str):
        note = str(note or "")
    if summary is not None and not isinstance(summary, str):
        summary = None
    ok = await vet_db.update_conversation_note(conversation_id, note, summary)
    if not ok:
        raise HTTPException(400, "note update returned no changes")
    return {"ok": True}


@app.get("/api/admin/conversations/{conversation_id}/export", dependencies=[Depends(require_admin)])
async def conversation_export(conversation_id: str, format: str = Query("json", pattern="^(json|csv)$")):
    if not vet_db.conn:
        raise HTTPException(503, "sqlite not enabled")
    conv = await vet_db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "conversation not found")
    turns = await vet_db.list_turns(conversation_id)
    if format == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["turn_no", "user_text", "agent_text", "unknown", "retrieval_best_score", "latency_ms", "sources", "created"])
        for t in turns:
            w.writerow([t["turn_no"], t["user_text"], t["agent_text"], bool(t["unknown"]),
                        t["retrieval_best_score"], t["latency_ms"], t["sources"], t["created"]])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                headers={"Content-Disposition": f"attachment; filename=conversation-{conversation_id}.csv"})
    return {"ok": True, "conversation": conv, "turns": turns}


@app.get("/api/admin/unanswered/list", dependencies=[Depends(require_admin)])
async def unanswered_list(limit: int = Query(100, ge=1, le=500), resolved: bool | None = None,
                          mode: str | None = None):
    if not vet_db.conn:
        # fallback to legacy jsonl path
        return {"ok": True, "items": store.list_unanswered(), "note": "using legacy unanswered store"}
    rows = await vet_db.list_unanswered_v2(limit=limit, resolved=resolved, mode=mode)
    return {"ok": True, "rows": rows}


@app.patch("/api/admin/unanswered/list/{entry_id}", dependencies=[Depends(require_admin)])
async def unanswered_list_update(entry_id: str, resolved: bool = Query(True)):
    if vet_db.conn:
        await vet_db.mark_unanswered_resolved(entry_id, resolved)
        return {"ok": True}
    # fallback: try legacy update (only supports note/name/phone)
    if not store.update_unanswered(entry_id, resolved=resolved):
        raise HTTPException(404, "not found")
    return {"ok": True}


# ---------------------------------------------------------------- transcript / agent log files

# only real transcript / agent log file names: no traversal, no dotfiles, no anything else
_LOGFILE_RE = re.compile(r"^(transcript|agent_log)-[A-Za-z0-9._-]{1,80}\.txt$")


def _safe_log_file(filename: str):
    from store import DATA_DIR
    if not _LOGFILE_RE.fullmatch(filename or ""):
        raise HTTPException(400, "invalid file name")
    path = DATA_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, f"File not found: {filename}")
    return path


@app.get("/api/admin/log-files", dependencies=[Depends(require_admin)])
async def log_files():
    """List available transcript and agent log files in DATA_DIR."""
    from store import DATA_DIR
    files = []
    if not DATA_DIR.exists():
        return {"ok": True, "files": []}
    pattern = re.compile(r"^(transcript|agent_log)-(.+)\.txt$")
    for p in sorted(DATA_DIR.iterdir(), key=lambda x: x.name):
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                files.append({"name": p.name, "type": m.group(1), "tag": m.group(2), "size": p.stat().st_size})
    return {"ok": True, "files": files}


@app.get("/api/admin/log-files/{filename}/content", dependencies=[Depends(require_admin)])
async def log_file_content(filename: str):
    """Return the raw content of a transcript or agent log file."""
    path = _safe_log_file(filename)
    return {"ok": True, "filename": filename, "content": path.read_text(encoding="utf-8")}


@app.get("/api/admin/log-files/{filename}/download", dependencies=[Depends(require_admin)])
async def log_file_download(filename: str):
    """Download a transcript or agent log file."""
    path = _safe_log_file(filename)
    return PlainTextResponse(path.read_text(encoding="utf-8"),
                            media_type="text/plain",
                            headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/api/admin/health/dependencies", dependencies=[Depends(require_admin)])
async def dependency_health():
    redis_ok = False
    if redis_store.enabled and redis_store.redis:
        try:
            redis_ok = await redis_store.redis.ping()
        except Exception as e:
            _log.warning("redis health check failed: %s", e)
    db_ok = bool(vet_db.conn)
    return {"ok": True, "redis": {"url_filled": bool(config.REDIS_URL), "enabled": redis_store.enabled, "ok": redis_ok},
            "sqlite": {"path_filled": bool(config.SQLITE_DB), "ok": db_ok}}


@app.get("/api/admin/settings/audit", dependencies=[Depends(require_admin)])
async def settings_audit(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    if not vet_db.conn:
        return {"ok": True, "rows": [], "note": "sqlite audit store not enabled"}
    rows = await vet_db.conn.execute_fetchall(
        "SELECT id, changed_by, fields, changed_at FROM settings_audit ORDER BY changed_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    out = []
    for r in rows:
        fields = json.loads(r["fields"]) if r["fields"] else {}
        out.append({"id": r["id"], "changed_by": r["changed_by"], "fields": fields, "changed_at": r["changed_at"]})
    return {"ok": True, "rows": out, "limit": limit, "offset": offset}


# ---------------------------------------------------------------- UI


_dist = config.BASE_DIR.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
else:
    @app.get("/")
    async def root():
        return JSONResponse({"msg": "Backend running. Build the frontend (npm run build) or run the Vite dev server."})
