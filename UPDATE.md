# Update Log — v2 (feature/version-1)

## What changed and why

### 1. Session memory moved to Redis (with in-process fallback)
- New module: `backend/redis_session.py`
- Active session state and recent history are now stored in Redis under `session:{id}` and `session:{id}:history`, with a TTL from `V2_SESSION_TTL_SEC`.
- If `REDIS_URL` is empty or Redis is unreachable, the agent falls back to an in-process dict so the existing behavior keeps working.
- Why: process-local session state does not survive restarts and does not share state across workers.

### 2. Durable persistence via SQLite
- New module: `backend/db.py` (uses `aiosqlite`)
- Tables: `conversations`, `turns`, `contacts`, `unanswered_v2`, `settings_audit`
- Every accepted turn is written to SQLite when `SQLITE_DB` is configured. Contact captures and unanswered questions are mirrored into SQLite as well.
- Why: file-JSON/JSONL is not queryable or exportable at scale; SQLite gives pagination, metrics, and export for conversations and unanswered logs.

### 3. Hybrid RAG (BM25 + dense + RRF, optional re-ranker, query expansion)
- `backend/kb.py` now supports v2 search when `V2_RAG=1`:
  - Dense path: unchanged sentence-transformers cosine retrieval.
  - Lexical path: `rank_bm25` index built per active knowledge version.
  - Fusion: reciprocal rank fusion with `V2_RRF_K`.
  - Optional re-ranking: cross-encoder over the fused top `V2_RERANK_TOP` results when `V2_RERANK=1`.
  - Short-query expansion when `V2_QUERY_EXPANSION=1`, using a synonym map at `V2_SYNONYMS` plus KB heading terms.
- New file: `backend/knowledge/synonyms.json` (optional synonym map for short queries).
- Why: dense-only retrieval can miss exact-term questions; hybrid search improves recall and precision, especially for short keyword queries.

### 4. Agentic capability scaffold + guardrails
- New module: `backend/agent.py`
- Capability registry with built-in entries: `lookup_policy`, `capture_contact`, `capture_booking_intent`, `escalate_to_human`.
- Structured extraction helpers for contact and booking intent.
- Guardrails: unknown-gate check, citation note for numeric claims not covered by sources, out-of-domain detection.
- `session.py` now uses a capability-aware system prompt when `V2_RAG=1`, and applies the citation guardrail after streaming.
- Why: the agent should be able to express structured intents and apply basic safety/quality checks without replacing the existing streaming TTS pipeline.

### 5. Observability
- `backend/logging_utils.py`: structured JSON logging when `STRUCTURED_LOGS=1`.
- `backend/app.py`:
  - `/api/health` now returns a `v2` block: redis status, sqlite status, hybrid RAG flags, and aggregated metrics when available.
  - `/api/admin/metrics`: turn count, unknown-turn count, average latency, unanswered count.
  - `/api/admin/conversations` + `/api/admin/conversations/{id}/turns` + `/api/admin/conversations/{id}/export`: paginated conversation list and turn export (JSON or CSV).
  - `/api/admin/unanswered/list` and `/api/admin/unanswered/list/{id}`: unanswered list/resolve using SQLite when available, with legacy fallback.
  - `/api/admin/health/dependencies`: redis + sqlite dependency check.
  - `/api/admin/settings/audit`: settings change audit log from SQLite.
- Frontend:
  - `frontend/src/Settings.jsx`: new Observability tab.
  - `frontend/src/Talk.jsx`: v2 tiles added to the metrics drawer (Redis, SQLite, hybrid RAG status).
- Why: production use needs debuggable logs, health checks, metrics, and conversation export.

### 6. Configuration
- `backend/config.py`: new v2 settings: `REDIS_URL`, `SQLITE_DB`, `V2_RAG`, `V2_RERANK`, `V2_RRF_K`, `V2_TOP_DENSE`, `V2_TOP_BM25`, `V2_RERANK_TOP`, `V2_QUERY_EXPANSION`, `V2_SYNONYMS`, `V2_SESSION_TTL_SEC`, `STRUCTURED_LOGS`.
- `backend/requirements.txt`: added `redis`, `aiosqlite`, `rank-bm25` for v2.
- `.env.example`: v2 section added with commented defaults.

## Backward compatibility

- When `REDIS_URL`, `SQLITE_DB`, and `V2_RAG` are empty/disabled, the agent keeps the existing in-process/file behavior.
- Existing admin endpoints are unchanged.
- Existing knowledge base files and version layout are unchanged; v2 search reads the same `chunks.json` and `embeddings.npy`.

## Files added
- `backend/V2_SPEC.md`
- `backend/agent.py`
- `backend/db.py`
- `backend/logging_utils.py`
- `backend/redis_session.py`
- `backend/knowledge/synonyms.json`
- `UPDATE.md`

## Files modified
- `backend/app.py`
- `backend/config.py`
- `backend/kb.py`
- `backend/requirements.txt`
- `backend/session.py`
- `.env.example`
- `frontend/src/Settings.jsx`
- `frontend/src/Talk.jsx`
- `frontend/package-lock.json` (npm install regeneration)

## How to enable v2

1. Install v2 deps:
   - `pip install redis aiosqlite rank-bm25`
2. Ensure Redis is running, for example:
   - `redis-cli ping`
3. Set these in `.env`:
   - `REDIS_URL=redis://localhost:6379/0`
   - `SQLITE_DB=./data/v2.db`
   - `V2_RAG=1`
   - Optional: `V2_RERANK=1`, `V2_QUERY_EXPANSION=1`
4. Restart the backend.

## Notes
- The capability handlers currently echo their payload. Booking intent and human handoff can be wired into the existing contact/unanswered flow later.
- Full behavioral validation needs the real backend runtime with models installed; this update was built and compiled for correctness and compatibility, not end-to-end run in this environment.
