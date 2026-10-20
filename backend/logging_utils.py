"""v2: structured JSON logging with session/turn context."""
import json
import logging
import sys
import time
from typing import Any

if True:  # keep import simple
    from config import STRUCTURED_LOGS


class ContextFilter(logging.Filter):
    """Attaches a simple context dict to log records if set on the thread/task."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = getattr(record, "ctx", None)
        if ctx:
            for k, v in ctx.items():
                setattr(record, f"ctx_{k}", v)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if getattr(record, "exc_info", None):
            base["error"] = self.formatException(record.exc_info)
        ctx = {}
        for k in ("session_id", "turn", "utt", "mode"):
            v = getattr(record, f"ctx_{k}", None)
            if v is not None:
                ctx[k] = v
        if ctx:
            base["ctx"] = ctx
        return json.dumps(base, ensure_ascii=False)


def setup_logging() -> None:
    root = logging.getLogger()
    if STRUCTURED_LOGS:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.handlers = [handler]
    else:
        # keep a readable fallback
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.handlers = [handler]
    root.setLevel(logging.INFO)


def log(ctx: dict[str, Any] | None = None, **extra) -> logging.Logger:
    """Small helper to grab a logger with context attached via extra."""
    return logging.getLogger("v2")
