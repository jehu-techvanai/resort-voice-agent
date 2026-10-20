"""v2: Redis-backed hot session state + recent conversation history, with a lightweight
in-process fallback when REDIS_URL is empty or Redis is unavailable.

Design:
- Session state lives in Redis as JSON under key session:{id} with TTL.
- Recent conversation history lives under session:{id}:history, also JSON, same TTL.
- If Redis is unavailable, we fall back to an in-process dict protected by an asyncio.Lock.
- This module does NOT own durable history; that is db.py. This module is for low-latency
  access during a live session and cross-worker state sharing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from config import REDIS_URL, V2_SESSION_TTL_SEC

_log = logging.getLogger("redis_session")

KEY_SESSION = "session:{id}"
KEY_HISTORY = "session:{id}:history"


class RedisSessionStore:
    """Single entry point for session state reads/writes.

    Owns either a redis.asyncio pool or an in-process fallback.
    """

    def __init__(self):
        self.redis = None
        self.pool = None
        self.enabled = False
        self._fallback: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def open(self) -> bool:
        if not REDIS_URL:
            _log.info("v2 Redis disabled (REDIS_URL empty); using in-process fallback")
            self.enabled = False
            return True
        try:
            import redis.asyncio as aredis
        except ImportError:
            _log.warning("redis not installed; using in-process fallback")
            self.enabled = False
            return True
        try:
            self.pool = aredis.ConnectionPool.from_url(REDIS_URL, decode_responses=True, max_connections=50)
            self.redis = aredis.Redis(connection_pool=self.pool)
            await self.redis.ping()
            self.enabled = True
            _log.info("v2 Redis enabled: %s", REDIS_URL)
            return True
        except Exception as e:
            _log.warning("v2 Redis unavailable, using in-process fallback: %s", e)
            self.enabled = False
            self.redis = None
            self.pool = None
            return True

    async def close(self) -> None:
        if self.pool:
            try:
                await self.pool.disconnect()
            except Exception as e:
                _log.warning("redis pool disconnect error: %s", e)
        self.redis = None
        self.pool = None
        self.enabled = False

    async def save_session(self, session_id: str, state: dict, history: list[dict] | None = None) -> bool:
        ttl = V2_SESSION_TTL_SEC
        payload = {"state": state, "updated": int(time.time()), "history": history or []}
        if self.enabled and self.redis:
            try:
                await self.redis.set(KEY_SESSION.format(id=session_id), json.dumps(payload), ex=ttl)
                await self.redis.set(KEY_HISTORY.format(id=session_id), json.dumps(history or []), ex=ttl)
                return True
            except Exception as e:
                _log.warning("redis save_session error: %s", e)
        # fallback
        async with self._lock:
            self._fallback[session_id] = payload
        return True

    async def load_session(self, session_id: str) -> dict | None:
        if self.enabled and self.redis:
            try:
                raw = await self.redis.get(KEY_SESSION.format(id=session_id))
                if raw:
                    return json.loads(raw)
            except Exception as e:
                _log.warning("redis load_session error: %s", e)
        async with self._lock:
            return dict(self._fallback.get(session_id, {})) if session_id in self._fallback else None

    async def load_history(self, session_id: str) -> list[dict]:
        if self.enabled and self.redis:
            try:
                raw = await self.redis.get(KEY_HISTORY.format(id=session_id))
                if raw:
                    return json.loads(raw)
            except Exception as e:
                _log.warning("redis load_history error: %s", e)
        async with self._lock:
            return list(self._fallback.get(session_id, {}).get("history", []))

    async def delete_session(self, session_id: str) -> bool:
        if self.enabled and self.redis:
            try:
                await self.redis.delete(KEY_SESSION.format(id=session_id), KEY_HISTORY.format(id=session_id))
                return True
            except Exception as e:
                _log.warning("redis delete_session error: %s", e)
        async with self._lock:
            self._fallback.pop(session_id, None)
        return True

    async def touch_session(self, session_id: str) -> bool:
        ttl = V2_SESSION_TTL_SEC
        if self.enabled and self.redis:
            try:
                await self.redis.expire(KEY_SESSION.format(id=session_id), ttl)
                await self.redis.expire(KEY_HISTORY.format(id=session_id), ttl)
                return True
            except Exception as e:
                _log.warning("redis touch_session error: %s", e)
        async with self._lock:
            if session_id in self._fallback:
                self._fallback[session_id]["updated"] = int(time.time())
        return True
