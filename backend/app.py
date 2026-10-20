"""Voice agent backend (v0.2).

WebSocket  /ws?mode=web            hands-free browser session (see session.py for the message protocol)
REST       /api/admin/...          settings page: persona, knowledge base upload / train / rollback, unanswered log
REST       /api/health, /api/public/settings
"""
import asyncio
import csv
import io
import json
import time

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import store
from engines import LLM, STT, TTS
from kb import KnowledgeBase, ParseError
from session import ENGINES, Session, system_prompt

app = FastAPI(title="Voice Agent")


@app.on_event("startup")
async def load_models():
    t0 = time.perf_counter()
    print(f"[startup] device={config.DEVICE} mock={config.MOCK}")
    ENGINES["stt"] = STT()
    ENGINES["kb"] = KnowledgeBase()
    ENGINES["tts"] = TTS()
    ENGINES["llm"] = LLM()
    await ENGINES["llm"].warmup()
    engine = store.load_settings()["stt_engine"]
    await asyncio.get_running_loop().run_in_executor(None, ENGINES["stt"].preload, engine)
    asyncio.create_task(_keep_llm_loaded())
    kb = ENGINES["kb"]
    print(f"[startup] models ready in {time.perf_counter() - t0:.1f}s, knowledge base {kb.version} "
          f"with {len(kb.chunks)} chunks")
    print(f"[startup] settings page: /#/settings   admin password: {store.admin_password()}")


async def _keep_llm_loaded():
    while True:
        await asyncio.sleep(600)
        await ENGINES["llm"].keep_loaded()


# ---------------------------------------------------------------- public

@app.get("/api/health")
async def health():
    s = store.load_settings()
    kb = ENGINES.get("kb")
    return {
        "ok": True, "mock": config.MOCK, "device": config.DEVICE, "stt": config.STT_MODEL,
        "embed": config.EMBED_MODEL, "llm": config.LLM_MODEL, "tts": f"kokoro/{s['voice']}",
        "stt_engine": s["stt_engine"],
        "resort": s["business_name"], "agent": s["agent_name"],
        "kb_version": kb.version if kb else None, "kb_chunks": len(kb.chunks) if kb else 0,
    }


@app.get("/api/public/settings")
async def public_settings():
    s = store.load_settings()
    return {k: s[k] for k in ("agent_name", "business_name", "wake_required_web", "followup_seconds")}


# ---------------------------------------------------------------- live session

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

    async def send(obj):
        await ws.send_text(json.dumps(obj))

    session = Session(send, mode=mode)
    s = store.load_settings()
    await send({"type": "ready", "session": session.id, "agent_name": s["agent_name"],
                "business_name": s["business_name"], "wake_required": session.wake})
    await session.send_state()
    tasks: set[asyncio.Task] = set()

    def spawn(coro):
        t = asyncio.create_task(coro)
        tasks.add(t)
        t.add_done_callback(tasks.discard)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
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
        try:
            out.append(kb.stage(f.filename or "upload", await f.read()))
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


# ---------------------------------------------------------------- UI

_dist = config.BASE_DIR.parent / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
else:
    @app.get("/")
    async def root():
        return JSONResponse({"msg": "Backend running. Build the frontend (npm run build) or run the Vite dev server."})
