"""v2: lightweight agentic capability layer.

Keeps the existing streaming pipeline intact but adds:
- a capability registry that the system prompt can advertise to the LLM
- structured extraction validators (contact, booking intent)
- guardrails: citation enforcement, unknown handling, out-of-domain detection
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from config import MOCK

_log = logging.getLogger("agent")


# ---------------------------------------------------------------- capabilities (tools)

@dataclass
class Capability:
    name: str
    description: str
    accepts: str          # what the LLM may pass, in plain language
    returns: str          # what the caller gets back
    handler: Callable[[dict[str, Any]], dict[str, Any]]


CAPABILITIES: list[Capability] = []


def register(cap: Capability) -> Capability:
    CAPABILITIES.append(cap)
    return cap


# ---------------------------------------------------------------- built-in capabilities

register(
    Capability(
        name="lookup_policy",
        description="Answer a factual question from the resort/agent knowledge base. "
        "Use this when the customer asks about rooms, prices, facilities, policies, "
        "check-in, transfers, spa, dining, activities, weddings, etc.",
        accepts="a short natural-language question",
        returns="a concise factual answer, or a note that the information was not found",
        handler=lambda payload: payload,
    )
)


register(
    Capability(
        name="capture_contact",
        description="Collect the customer's name and phone number when the agent could not "
        "answer a question and wants to log it for a callback. Only use after the agent says "
        "it cannot answer and asks for contact details.",
        accepts="name (string) and phone (digits only)",
        returns="the captured name and phone, or a note that the caller declined",
        handler=lambda payload: payload,
    )
)


register(
    Capability(
        name="capture_booking_intent",
        description="When a caller wants to book or reserve, capture the room type, check-in "
        "date, number of nights, number of guests and any special requests. Do NOT confirm "
        "availability or prices beyond what the knowledge base states; say the team will confirm.",
        accepts="room_type, checkin_date, nights, guests, special_requests (all optional strings)",
        returns="the captured intent as structured fields",
        handler=lambda payload: payload,
    )
)


register(
    Capability(
        name="escalate_to_human",
        description="If the caller is upset, asks for a person, or the request is outside the "
        "agent's scope (complaints, refunds already processed, legal questions, anything the "
        "agent should not answer), hand off to a human with a reason and queue.",
        accepts="reason (short) and queue (optional)",
        returns="a handoff acknowledgement",
        handler=lambda payload: payload,
    )
)


# ---------------------------------------------------------------- structured extraction

_PHONE_RE = re.compile(r"[\+]?[\d\s\-()]{6,15}")
_DATE_RE = re.compile(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b", re.I)


def extract_contact(text: str) -> dict[str, str]:
    name = ""
    phone = ""
    # crude name extraction: "i am X", "my name is X", "this is X"
    m = re.search(r"(?:i(?:'m| am)\s+|name is\s+|this is\s+|call me\s+)([A-Za-z][A-Za-z.' -]{1,40})", text, re.I)
    if m:
        name = m.group(1).strip().rstrip(" ,.!?")
    digits = _PHONE_RE.search(text)
    if digits:
        digits = re.sub(r"[^\d]", "", digits.group(0))
        if 6 <= len(digits) <= 15:
            phone = digits
    return {"name": name[:60], "phone": phone}


def extract_booking_intent(text: str, context: list[tuple[Any, float]] | None = None) -> dict[str, Any]:
    """Best-effort extraction of booking fields from free text. Validated lightly."""
    out = {
        "room_type": "",
        "checkin_date": "",
        "nights": "",
        "guests": "",
        "special_requests": "",
        "confidence": "low",
    }
    low = text.lower()
    room_hits = []
    for hint in ("pool villa", "sea view deluxe", "garden view", "family suite", "villa", "suite", "room"):
        if hint in low:
            room_hits.append(hint)
    if room_hits:
        out["room_type"] = max(set(room_hits), key=lambda h: low.count(h))
        out["confidence"] = "medium"
    dates = _DATE_RE.findall(text)
    if dates:
        out["checkin_date"] = dates[0]
        out["confidence"] = "medium"
    m = re.search(r"(\d+)\s*(?:night|day|stay)", low)
    if not m:
        m = re.search(r"(\d+)\s*(?:people|guests|adults|adults and|adults,)", low)
    if m:
        out["nights" if "night" in (m.group(0) or "") else "guests"] = m.group(1)
        out["confidence"] = "medium"
    m = re.search(r"(?:for|with)\s+(\d+)\s*(?:people|guests|adults|adults and|adults,)", low)
    if m:
        out["guests"] = m.group(1)
        out["confidence"] = "medium"
    return out


# ---------------------------------------------------------------- guardrails

class Guardrails:
    """Applies post-generation checks before a reply is spoken."""

    def __init__(self, settings: dict, retrieval_best: float | None = None):
        self.settings = settings
        self.retrieval_best = retrieval_best

    def unknown_gate(self, answer: str) -> bool:
        """If retrieval was poor and the model did not clearly know, treat as unknown."""
        if self.retrieval_best is None:
            return False
        if self.retrieval_best < self.settings.get("min_retrieval_score", 0.4):
            upper = answer.upper().replace(" ", "")
            if "UNKNOWN" in upper or "DON'T KNOW" in upper or "I DON'T KNOW" in upper or "I DO NOT KNOW" in upper:
                return True
        return False

    def citation_note(self, answer: str, sources: list[dict]) -> str:
        """Return a note to append when the answer states a number/policy not covered by sources."""
        if not sources:
            return ""
        # if the answer includes rupees/prices and no source mentions price-like content, flag
        has_currency = bool(re.search(r"rupees?|rs\.?\s*\d", answer, re.I))
        src_has_price = any(re.search(r"rupees?|rs\.?\s*\d|\d{1,3}(,\d{3})+", c.get("text", ""), re.I) for c in sources)
        if has_currency and not src_has_price:
            return " (I should check that figure against our current rates.)"
        return ""

    def out_of_domain(self, answer: str, user_text: str) -> bool:
        """Crude out-of-domain detection: model refuses or deflects in a way that suggests scope exit."""
        low = answer.lower()
        user_low = user_text.lower()
        if "i can't help with that" in low or "that's not something i can" in low:
            return True
        # legal/medical/complaint keywords in user text that the agent should redirect
        leave_keywords = ["lawyer", "sue", "lawsuit", "medical", "doctor", "prescription", "refund my money now", "complaint"]
        if any(k in user_low for k in leave_keywords) and "escalat" in low:
            return True
        return False


def build_capability_text() -> str:
    if not CAPABILITIES:
        return ""
    lines = ["You have access to these capabilities. Use them when the conversation calls for it; "
             "otherwise answer normally from the knowledge base info below."]
    for c in CAPABILITIES:
        lines.append(f"- {c.name}: {c.description}")
        lines.append(f"  Accepts: {c.accepts}")
        lines.append(f"  Returns: {c.returns}")
    return "\n".join(lines)


# ---------------------------------------------------------------- v2 system prompt wrapper

def system_prompt_v2(settings: dict, context: list, user_text: str, greeted: bool = False,
                     prev_answer: str | None = None) -> str:
    """Extend the existing system_prompt with capability list and guardrail hints."""
    from session import system_prompt as base_prompt
    base = base_prompt(settings, context, greeted=greeted, prev_answer=prev_answer)
    caps = build_capability_text()
    extra = []
    if caps:
        extra.append(caps)
    extra.append(
        "If you use a capability, respond with your normal spoken reply and, in your head, "
        "remember the structured details so the backend can record them. Do not say the word "
        "'capability' out loud."
    )
    if prev_answer:
        brief = prev_answer[:250] + ("..." if len(prev_answer) > 250 else "")
    else:
        brief = ""
    extra.append(
        "Repetition guard: "
        + ("your last spoken reply was: \"" + brief + "\". " if brief else "")
        + "Compose each reply fresh. Never reuse whole sentences or whole phrases from your "
        "earlier replies — the customer notices a parrot immediately. Answer the newest "
        "message in new words."
    )
    extra.append(
        "Identity: questions like 'who are you' or 'what is your name' are normal and "
        "answerable — reply briefly with your name and role. Scope: if the question is not "
        "something a receptionist at this business could answer, reply with exactly UNKNOWN "
        "rather than guessing."
    )
    extra.append(
        "NEVER repeat your name or the business name in every reply. Only introduce yourself on "
        "the very first greeting. After that, answer directly without saying who you are or where "
        "you work every time."
    )
    if extra:
        base = base.rstrip() + "\n\n" + "\n".join(extra)
    return base
