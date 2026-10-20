# Resort Voice Agent — Version 2 (production-oriented)

## Goals
- Make sessions resilient and shareable across workers via Redis session memory.
- Persist conversations, contacts, unanswered questions, and turn metrics in SQLite for analytics, export, and moderation.
- Improve RAG accuracy with hybrid search (BM25 + dense), re-ranking, metadata filtering, and short-query expansion.
- Add an agentic capability layer (tools/intents, structured extraction, guardrails) without abandoning the existing streaming TTS pipeline.
- Add observability: structured logs, health enrichment, metrics, conversation export.

## Non-goals for v2
- Not introducing a full agent framework (LangGraph/etc.) yet; keep it thin and ownable.
- Not replacing the existing Kokoro TTS / faster-whisper STT / Ollama-LLM wiring.
- Not requiring external vector DB; keep numpy embedded index for now, make it swappable.

## Architecture
- Backend: FastAPI (uvicorn). Async-first where it matters (Redis, LLM stream, KB search offload).
- Redis (redis.asyncio connection pool): active session state + recent conversation history for hot sessions; cache of kb chunk metadata.
- SQLite (aiosqlite): durable conversation log, unanswered log, contact captures, turn metrics, settings audit log. Retain JSON files as fallback/source of truth during transition.
- Knowledge base: keep sentence-transformers dense index; add BM25 index per version; fuse with RRF; optional cross-encoder re-ranker for top-N; metadata filters (source, version, section/tag); short-query expansion using KB headings + synonyms.
- Agentic: a small tools registry; booking-intent and escalation tools; structured extraction (name/phone/dates/room-type) validated before storage; guardrails (cite-only rule, unknown handling, out-of-domain refusal).

## Data model (SQLite)
- conversations: id, session_id, mode, started, ended, total_seconds, status, summary
- turns: id, conversation_id, turn_no, user_text, agent_text, unknown, retrieval_best_score, latency_ms, created
- contacts: id, conversation_id, name, phone, captured_at, status
- unanswered (mirror of existing JSONL + richer query): id, conversation_id, question, name, phone, note, resolved, created, mode
- settings_audit: id, changed_by, fields, snapshot, changed_at

## Redis keys
- session:{id} -> active session state (JSON), TTL ~ 30 min idle
- session:{id}:history -> recent message list (JSON), TTL ~ 30 min idle
- kb:meta:{version} -> chunk metadata cache
- metrics:counts -> aggregated counters (hset)

## RAG v2 design
- Per version: chunks.json (kept), embeddings.npy (kept), plus bm25 index serialized to bm25.json (tokenized corpus + doc freqs) for fast reload.
- search(query, k, filters) -> dense top-K, bm25 top-K, RRF fuse, optional re-rank top 12, apply min_retrieval_score gate.
- short query (< 6 words) expansion: append nearest KB heading terms + synonyms from a small synonym map (configurable) before dense encoding.
- metadata filters: source, version, section (parsed from chunk title prefix), tags from extra_instructions hints.

## Agentic layer
- Capability registry: each capability = name, description, accepts, returns, handler.
- Built-in v2 capabilities: capture_booking_intent (room type, dates, guests), capture_contact (name, phone), escalate_to_human (reason, queue), lookup_policy (already via RAG).
-LLM sees system prompt with available capabilities and must emit structured extraction where applicable; backend validates and stores.
-Guardrails: if retrieval score low and not small talk -> unknown handling; if answer contains non-cited numeric claims -> tone-down/citation reminder; out-of-domain detection via intent classifier heuristic (LLM self-report + keyword).

## Observability
- All backend logs as JSON via stdlib logging + custom formatter (timestamp, level, module, session_id, turn, msgs).
-/api/health enriched: redis_ok, sqlite_ok, kb_version, kb_chunks, queue depth hints.
-/api/admin/metrics: turn count, avg latency, unknown rate, unanswered count, active sessions.
-/api/admin/conversations: paginated list + export (CSV/JSON) of a conversation's turns.
-Conversation ID propagated from WebSocket session into logs and stored rows.

## Migration/naming
- New code lives alongside v1; feature-flagged via config.V2_RAG, config.REDIS_URL, config.SQLITE_DB.
- When Redis/SQLite are unavailable, fall back to existing in-process/file behavior so the agent still runs.
