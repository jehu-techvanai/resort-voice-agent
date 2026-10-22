"""v2: durable SQLite persistence for conversations, turns, contacts, unanswered, metrics, settings audit.

Kept separate from store.py (which still owns settings.json / unanswered.jsonl / kb) so the v1
file-based paths keep working when SQLITE_DB is empty, and so Redis-owned hot session state is
distinct from durable storage.

Lifecycle: created/opened by app lifespan. All mutations go through this module.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from config import SQLITE_DB

_log = logging.getLogger("db")


class DB:
    def __init__(self, path: Path | None = None):
        self.path = path or SQLITE_DB
        self.conn: aiosqlite.Connection | None = None

    async def open(self) -> bool:
        if not self.path:
            _log.info("v2 DB disabled (SQLITE_DB empty)")
            return False
        try:
            self.conn = await aiosqlite.connect(str(self.path))
            self.conn.row_factory = aiosqlite.Row
            # WAL lets dashboard reads happen while a live voice turn is writing;
            # busy_timeout avoids immediate 'database is locked' under that mix.
            await self.conn.execute("PRAGMA journal_mode=WAL")
            await self.conn.execute("PRAGMA busy_timeout=5000")
            await self.conn.execute("PRAGMA synchronous=NORMAL")
            await self._migrate()
            _log.info("v2 DB opened: %s", self.path)
            return True
        except Exception as e:
            _log.exception("v2 DB failed to open: %s", e)
            self.conn = None
            return False

    async def close(self) -> None:
        if self.conn:
            try:
                await self.conn.close()
            except Exception as e:
                _log.warning("v2 DB close error: %s", e)
            self.conn = None

    async def _migrate(self) -> None:
        assert self.conn
        await self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations(
                id TEXT PRIMARY KEY,
                session_id TEXT,
                mode TEXT,
                started TEXT,
                ended TEXT,
                total_seconds REAL,
                status TEXT,
                summary TEXT
            );
            CREATE TABLE IF NOT EXISTS turns(
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                turn_no INTEGER,
                user_text TEXT,
                agent_text TEXT,
                unknown INTEGER,
                retrieval_best_score REAL,
                latency_ms INTEGER,
                sources TEXT,
                created TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );
            CREATE TABLE IF NOT EXISTS contacts(
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                name TEXT,
                phone TEXT,
                captured_at TEXT,
                status TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            );
            CREATE TABLE IF NOT EXISTS unanswered_v2(
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                question TEXT,
                name TEXT,
                phone TEXT,
                note TEXT,
                resolved INTEGER,
                created TEXT,
                mode TEXT,
                session_id TEXT
            );
            CREATE TABLE IF NOT EXISTS settings_audit(
                id TEXT PRIMARY KEY,
                changed_by TEXT,
                fields TEXT,
                snapshot TEXT,
                changed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS turns_conv ON turns(conversation_id);
            CREATE INDEX IF NOT EXISTS unanswered_v2_conv ON unanswered_v2(conversation_id);
            CREATE INDEX IF NOT EXISTS unanswered_v2_resolved ON unanswered_v2(resolved);
            """
        )
        await self.conn.commit()

    # ---------------------------------------------------------------- conversations
    async def start_conversation(self, session_id: str, mode: str) -> dict:
        assert self.conn
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "id": uuid.uuid4().hex[:12],
            "session_id": session_id,
            "mode": mode,
            "started": now,
            "ended": "",
            "total_seconds": 0.0,
            "status": "active",
            "summary": "",
        }
        await self.conn.execute(
            "INSERT INTO conversations(id,session_id,mode,started,status,summary) VALUES(?,?,?,?,?,?)",
            (row["id"], row["session_id"], row["mode"], now, "active", ""),
        )
        await self.conn.commit()
        return row

    async def end_conversation(self, conversation_id: str, total_seconds: float, summary: str = "") -> bool:
        assert self.conn
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        await self.conn.execute(
            "UPDATE conversations SET ended=?, total_seconds=?, status='done', summary=? WHERE id=?",
            (now, total_seconds, summary, conversation_id),
        )
        await self.conn.commit()
        return True

    async def list_conversations(self, limit: int = 50, offset: int = 0, status: str | None = None,
                                   q: str | None = None, mode: str | None = None) -> list[dict]:
        assert self.conn
        qsql = "SELECT id,session_id,mode,started,ended,total_seconds,status,summary FROM conversations"
        params: list[Any] = []
        where = []
        if status:
            where.append("status=?")
            params.append(status)
        if mode:
            where.append("mode=?")
            params.append(mode)
        if q:
            where.append("(id LIKE ? OR session_id LIKE ? OR summary LIKE ? OR mode LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like, like, like])
        if where:
            qsql += " WHERE " + " AND ".join(where) + " ORDER BY started DESC"
        else:
            qsql += " ORDER BY started DESC"
        qsql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = await self.conn.execute_fetchall(qsql, params)
        return [dict(r) for r in rows]

    async def get_conversation(self, conversation_id: str) -> dict | None:
        assert self.conn
        r = await self.conn.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,))
        row = await r.fetchone()
        return dict(row) if row else None

    # ---------------------------------------------------------------- turns
    async def add_turn(
        self,
        conversation_id: str,
        turn_no: int,
        user_text: str,
        agent_text: str,
        unknown: bool,
        retrieval_best_score: float,
        latency_ms: int,
        sources: list[dict] | None,
    ) -> dict:
        assert self.conn
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "id": uuid.uuid4().hex[:12],
            "conversation_id": conversation_id,
            "turn_no": turn_no,
            "user_text": user_text.strip()[:500],
            "agent_text": agent_text.strip()[:1000],
            "unknown": 1 if unknown else 0,
            "retrieval_best_score": round(retrieval_best_score, 3),
            "latency_ms": int(latency_ms),
            "sources": json.dumps(sources or []),
            "created": now,
        }
        await self.conn.execute(
            "INSERT INTO turns(id,conversation_id,turn_no,user_text,agent_text,unknown,retrieval_best_score,latency_ms,sources,created) VALUES"
            "(?,?,?,?,?,?,?,?,?,?)",
            tuple(row.values()),
        )
        await self.conn.commit()
        return row

    async def list_turns(self, conversation_id: str) -> list[dict]:
        assert self.conn
        rows = await self.conn.execute_fetchall(
            "SELECT * FROM turns WHERE conversation_id=? ORDER BY turn_no", (conversation_id,)
        )
        out = []
        for r in rows:
            d = dict(r)
            d["sources"] = json.loads(d["sources"]) if d["sources"] else []
            out.append(d)
        return out

    # ---------------------------------------------------------------- contacts
    async def add_contact(self, conversation_id: str, name: str, phone: str, status: str = "captured") -> dict:
        assert self.conn
        row = {
            "id": uuid.uuid4().hex[:12],
            "conversation_id": conversation_id,
            "name": name.strip()[:60],
            "phone": phone.strip()[:20],
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
        }
        await self.conn.execute(
            "INSERT INTO contacts(id,conversation_id,name,phone,captured_at,status) VALUES(?,?,?,?,?,?)",
            tuple(row.values()),
        )
        await self.conn.commit()
        return row

    async def update_conversation_note(self, conversation_id: str, note: str, summary: str | None = None) -> bool:
        assert self.conn
        fields = []
        vals = []
        if note != "":
            fields.append("summary=?")
            vals.append(note[:500])
        if summary is not None:
            if not fields:
                fields.append("summary=?")
            vals.append(summary[:500])
        if not fields:
            return False
        vals.append(conversation_id)
        await self.conn.execute(
            f"UPDATE conversations SET {', '.join(fields)} WHERE id=?", vals,
        )
        await self.conn.commit()
        return True

    async def list_contacts(self, conversation_id: str) -> list[dict]:
        assert self.conn
        rows = await self.conn.execute_fetchall(
            "SELECT id,name,phone,captured_at,status FROM contacts WHERE conversation_id=? ORDER BY captured_at",
            (conversation_id,),
        )
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- unanswered (v2 mirror)
    async def log_unanswered_v2(
        self,
        conversation_id: str,
        session_id: str,
        mode: str,
        question: str,
        name: str = "",
        phone: str = "",
        note: str = "",
    ) -> dict:
        assert self.conn
        row = {
            "id": uuid.uuid4().hex[:12],
            "conversation_id": conversation_id,
            "session_id": session_id,
            "mode": mode,
            "question": question.strip()[:500],
            "name": name.strip()[:60],
            "phone": phone.strip()[:20],
            "note": note.strip()[:200],
            "resolved": 0,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        await self.conn.execute(
            "INSERT INTO unanswered_v2(id,conversation_id,session_id,mode,question,name,phone,note,resolved,created) VALUES"
            "(?,?,?,?,?,?,?,?,?,?)",
            tuple(row.values()),
        )
        await self.conn.commit()
        return row

    async def list_unanswered_v2(self, limit: int = 100, resolved: bool | None = None, mode: str | None = None) -> list[dict]:
        assert self.conn
        q = "SELECT * FROM unanswered_v2 WHERE 1=1"
        params: list[Any] = []
        if resolved is not None:
            q += " AND resolved=?"
            params.append(1 if resolved else 0)
        if mode:
            q += " AND mode=?"
            params.append(mode)
        q += " ORDER BY created DESC LIMIT ?"
        params.append(limit)
        rows = await self.conn.execute_fetchall(q, params)
        return [dict(r) for r in rows]

    async def mark_unanswered_resolved(self, entry_id: str, resolved: bool = True) -> bool:
        assert self.conn
        await self.conn.execute(
            "UPDATE unanswered_v2 SET resolved=? WHERE id=?", (1 if resolved else 0, entry_id)
        )
        await self.conn.commit()
        return True

    # ---------------------------------------------------------------- settings audit
    async def audit_settings(self, changed_by: str, fields: dict, snapshot: dict) -> dict:
        assert self.conn
        row = {
            "id": uuid.uuid4().hex[:12],
            "changed_by": changed_by,
            "fields": json.dumps(fields),
            "snapshot": json.dumps(snapshot),
            "changed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        await self.conn.execute(
            "INSERT INTO settings_audit(id,changed_by,fields,snapshot,changed_at) VALUES(?,?,?,?,?)",
            tuple(row.values()),
        )
        await self.conn.commit()
        return row

    # ---------------------------------------------------------------- metrics helpers
    async def count_turns(self, conversation_id: str) -> int:
        assert self.conn
        r = await self.conn.execute("SELECT COUNT(*) FROM turns WHERE conversation_id=?", (conversation_id,))
        row = await r.fetchone()
        return row[0] if row else 0

    async def aggregate_metrics(self) -> dict:
        assert self.conn
        t = await self.conn.execute_fetchone("SELECT COUNT(*) AS c FROM turns")
        u = await self.conn.execute_fetchone("SELECT COUNT(*) AS c FROM turns WHERE unknown=1")
        c = await self.conn.execute_fetchall("SELECT conversation_id, AVG(latency_ms) AS avg FROM turns GROUP BY conversation_id")
        latencies = [r["avg"] for r in c if r["avg"] is not None]
        # p50/p95 latency percentiles (per Deepgram-style production gates: averages hide
        # the tail spikes that actually make callers abandon the call)
        lat_rows = await self.conn.execute_fetchall("SELECT latency_ms FROM turns WHERE latency_ms > 0")
        all_lat = sorted(r["latency_ms"] for r in lat_rows)

        def pct(p: float) -> int:
            if not all_lat:
                return 0
            return all_lat[min(len(all_lat) - 1, max(0, round(p / 100 * (len(all_lat) - 1))))]

        return {
            "turns": t["c"] if t else 0,
            "unknown_turns": u["c"] if u else 0,
            "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 1) if latencies else 0,
            "latency_p50_ms": pct(50),
            "latency_p95_ms": pct(95),
            "latency_max_ms": all_lat[-1] if all_lat else 0,
            "conversations_with_turns": len(latencies),
            "unanswered_v2_count": (await self.conn.execute_fetchone("SELECT COUNT(*) AS c FROM unanswered_v2"))["c"],
        }
