"""One live conversation (a browser tab, an API client, later a phone call).

Input arrives as whole utterances (16 kHz PCM cut by a voice-activity detector on the client) or typed text.
For each utterance: speech to text -> noise / echo filters -> calling-name check (web only) -> English check
-> retrieval -> LLM streamed sentence by sentence -> TTS per sentence.

Barge-in: while the agent is speaking the client keeps listening and turns the agent's volume down when it
hears speech. If the server accepts that speech as a real request, it cancels the current reply, keeps in
the history only the sentences the caller actually heard, and starts the new turn. If the server rejects it
(noise, echo of the agent's own voice, no calling name) the client restores the volume and nothing changes.
"""
import asyncio
import base64
import difflib
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

import config
import store
from engines import SentenceSplitter, clean_for_speech, pcm16_to_float
from agent import Guardrails, extract_contact, extract_booking_intent, system_prompt_v2

ENGINES = {}   # filled by app.py at startup: stt, tts, llm, kb, redis_session, db

_log = logging.getLogger("session")

ENGLISH_ONLY_REPLY = "Sorry, I can only help in English at the moment. Could you please say that again in English?"
NOT_HEARD_REPLY = "Sorry, I didn't catch that. Could you say it again?"

# Phrases Whisper tends to invent on noise or silence
HALLUCINATIONS = {
    "thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching", "you", "bye",
    "okay", "ok", "so", "oh", "uh", "um", "hmm", "mm", "yeah", "the", "i", "please subscribe", "subtitles by",
    "praise",
}
WAKE_PREFIXES = {"hey", "hi", "hello", "ok", "okay", "oh", "excuse", "me", "so"}
SMALLTALK = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening)|thanks?|thank you|ok(ay)?|yes|no|yeah|"
                       r"bye|goodbye|great|cool|fine|sure|alright|who are you|what is your name|how are you)\b", re.I)

# Greetings that should wake the agent even without the calling name (user doesn't know the name yet).
# Includes closing 인사를 so "bye" can also wake the session for a proper goodbye.
GREETINGS = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening|day)|yo|what'?s up|howdy|welcome|bye|goodbye)\b", re.I)
DECLINE = re.compile(r"\b(no( thanks?| thank you)?|not now|don'?t|do not|never ?mind|skip|no need)\b", re.I)
QUESTION_START = re.compile(r"^(?:and |also |so |but |okay,? |ok,? |well,? |right,? )?\s*(what|how|is|are|do|does|can|could|where|when|which|who|why|will|would|tell)\b", re.I)

# Words and short phrases that signal the speaker is still talking about the resort.
# Used to let context-carrying follow-ups wake the agent even without a question word.
RESORT_FOLLOWUP_HINTS = re.compile(
    r"\b(room|rooms|villa|villas|suite|suites|garden|sea|pool|beach|view|deluxe|famil"
    r"y|price|prices|cost|costs|cheap|expensive|rupees|rs\.?|budget|rate|rates|spend"
    r"|stay|staying|book|booking|reserve|reservation|availab|available|availability"
    r"|check.?in|check.?out|deposit|cancel|cancellation|transfer|airport|spa|ayurv"
    r"eda|doctor|massage|dinner|lunch|breakfast|restaurant|bar|pool|gym|yoga|kit"
    r"chen|club|kids|family|couple|single|adult|child|guest|guests|person|people"
    r"|wedding|event|wedding|party|group|weather|location|where|distance|km|hour"
    r"|minute|hour|time|opening|open|close|closed|phone|email|website|website"
    r"|wifi|wi.?fi|parking|accessible|wheelchair|pet|pets|smoking|non.?smoking"
    r"|per night|per person|for two|for one|for three|for four|for a|for an"
    r"|low season|peak season|december|january|summer|winter|monsoon|rain"
    r"|which one|what's included|what is included|what about|how about|and the"
    r"|tell me|explain|describe|more|detail|details|include|includes|extra"
    r"|additional|nearby|surrounding|area|nearby|close to|far| kilometres\b)"
    r"|\b(sea view|pool villa|garden view|family suite|deluxe room|beach front"
    r"|lighthouse beach|kovalam|trivandrum|kovalam|kerala|kerala|india|kovalon"
    r"|palmlagoon|palm lagoon)\b",
    re.I,
)


def _context_carries_resort_topic(text: str) -> bool:
    """Return True when *text* is short but clearly continues a resort conversation.

    Catches follow-ups like 'And the deluxe rooms?', 'What about peak season?',
    'How much for a pool villa?', 'Tell me about the family suite' that do not start
    with a question word but still carry enough resort context to deserve a reply.
    """
    t = _norm(text)
    if len(t) < 4:
        return False
    # must carry at least one resort topic hint
    if not RESORT_FOLLOWUP_HINTS.search(text):
        return False
    # reject pure interjections / back-channel that happen to contain a keyword
    # e.g. 'oh', 'ok', 'yeah', 'right', 'cool' with a stray 'room' from echo
    pure_backchannel = bool(
        re.match(r"^(oh|ok|okay|yeah|yep|yes|no|right|cool|nice|great|good|thanks?|thank you|please|sure|alright)\b",
                  text.strip(), re.I)
    )
    if pure_backchannel and len(text.strip().split()) <= 3:
        return False
    return True
NUM_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6", "seven": "7", "eight": "8", "nine": "9"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _trim_repetition(text: str, prev: str) -> str | None:
    """If `text` opens with a run that closely matches `prev`, drop the repeated run.

    Returns the trimmed text, or None when there is nothing significant to trim.
    Works at word level so the mapping between normalised and original text stays exact.
    """
    if not prev or not text:
        return None
    a_words = _norm(text).split()
    b_words = _norm(prev).split()
    if not a_words or not b_words:
        return None
    # longest common prefix, in words
    i = 0
    while i < len(a_words) and i < len(b_words) and a_words[i] == b_words[i]:
        i += 1
    # require at least ~6 words of repeat before trimming
    if i < 6:
        return None
    # do not trim when the new reply is fully contained in the old one (nothing new)
    if i >= len(a_words):
        return None
    # drop the repeated leading words from the original text
    a_words_orig = text.lower().split()
    if i >= len(a_words_orig):
        return None
    rest = " ".join(a_words_orig[i:]).strip()
    if not rest:
        return None
    # prefer to start at a sentence boundary within the remainder
    m = re.search(r"(?<=[.!?])\s+", rest)
    if m:
        rest = rest[m.end():].lstrip()
    if len(rest) < 20:
        return None
    return rest


def match_wake(text: str, name: str) -> tuple[bool, str]:
    """Return (matched, text with the calling name removed). Accepts the name at the start
    ("Maya, ...", "Hey Maya ...") or at the very end ("... , Maya?"). Fuzzy, because Whisper may write Mya / Maia."""
    target = _norm(name).split()[0] if _norm(name) else ""
    if not target:
        return True, text
    words = text.split()
    clean = [_norm(w) for w in words]

    def is_name(i: int) -> int:
        """0 = no match, 1 = word i matches, 2 = words i and i+1 joined match ("my a")."""
        w = clean[i]
        if w and (w == target or _sim(w, target) >= 0.72):
            return 1
        if i + 1 < len(clean) and len(w) <= 3 and _sim(w + clean[i + 1], target) >= 0.8:
            return 2
        return 0

    i = 0
    while i < len(clean) and i < 3 and clean[i] in WAKE_PREFIXES:
        i += 1
    for j in range(i, min(i + 2, len(clean))):
        n = is_name(j)
        if n:
            rest = " ".join(words[j + n:]).lstrip(" ,.!?-")
            return True, rest
    if len(clean) >= 2:
        for j in (len(clean) - 1, len(clean) - 2):
            if j > 0 and is_name(j) == 1:
                rest = " ".join(words[:j]).rstrip(" ,.!?-")
                tail = words[-1][-1] if words[-1][-1] in "?!." else ""
                return True, rest + tail
    # Accept common greetings even without the calling name — the caller may not know the agent's name yet.
    if GREETINGS.match(text.strip()):
        return True, text
    # Also accept direct questions as a valid wake-up — the caller may just want an answer.
    if QUESTION_START.match(text.strip()):
        return True, text
    # Accept context-carrying follow-ups that don't start with a question word.
    # Examples: "And the deluxe rooms?", "What about peak season?", "How about the pool villa?",
    # "Tell me about the family suite." These carry enough resort context to deserve a reply.
    if _context_carries_resort_topic(text.strip()):
        return True, text
    return False, text


def words_to_digits(text: str) -> str:
    out, toks = [], re.findall(r"[a-z]+|\d+", text.lower())
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("double", "triple") and i + 1 < len(toks) and (toks[i + 1] in NUM_WORDS or toks[i + 1].isdigit()):
            d = NUM_WORDS.get(toks[i + 1], toks[i + 1][:1])
            out.append(d * (2 if t == "double" else 3))
            i += 2
            continue
        if t.isdigit():
            out.append(t)
        elif t in NUM_WORDS and t not in ("o", "oh") or (t in ("o", "oh") and out):
            out.append(NUM_WORDS[t])
        i += 1
    return "".join(out)


def extract_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    if len(digits) < 6:
        digits = words_to_digits(text)
    return digits if 6 <= len(digits) <= 15 else ""


def speak_digits(digits: str) -> str:
    return " ".join(digits)


_HINTS = {"version": None, "words": []}


def stt_hints(settings: dict) -> list[str]:
    """Names the speech-to-text should expect: calling name, business name, knowledge-base headings."""
    kb = ENGINES.get("kb")
    if kb and _HINTS["version"] != kb.version:
        seen, words = set(), []
        for c in kb.chunks:
            t = re.sub(r"\s*\(part \d+\)$", "", c.title).split(":")[-1].strip()
            if 2 < len(t) <= 48 and not t.endswith("?") and t.lower() not in seen:
                seen.add(t.lower())
                words.append(t)
        _HINTS.update(version=kb.version, words=words[:40])
    return [settings["agent_name"], settings["business_name"]] + _HINTS["words"]


def system_prompt(settings: dict, context: list, greeted: bool = False,
                 prev_answer: str | None = None) -> str:
    facts = "\n\n".join(f"[{c.title}]\n{c.text}" for c, _ in context) or "(nothing relevant found)"
    extra = settings.get("extra_instructions", "").strip()
    intro_rule = (
        "Introduce yourself by name and business on this first greeting. "
        "Say something warm and welcoming."
        if not greeted
        else
        "You greeted the customer earlier in this conversation. Do NOT re-introduce yourself "
        "or repeat the business name; answer directly and naturally."
    )
    repeat_rule = ""
    if prev_answer:
        brief = prev_answer[:200] + ("..." if len(prev_answer) > 200 else "")
        repeat_rule = (
            "Use the conversation history above instead of repeating yourself. If the customer's "
            "message refers back to what you already said (your last reply began: " + brief + "), "
            "confirm briefly or add one new detail, and never copy the whole answer word for word. "
            "If the customer is thanking you or acknowledging your last reply, keep your reply brief "
            "and do not re-state the information you already gave — a short acknowledgment or one-line "
            "follow-up is enough. If they ask something new, answer only the new part."
        )
    return f"""You are {settings['agent_name']}, the receptionist at {settings['business_name']}, a {settings.get('business_type') or 'business'}. You are speaking to a customer on the phone and your words will be read aloud.

How to talk:
- Sound like a warm, real human receptionist: short natural sentences, one idea each. Ask one short follow-up question when it keeps the conversation moving.
- Converse with what the customer actually asked. If they ask about price ranges, talk about the price ranges they mean, not a generic line about rooms being available.
- Vary how your replies open and flow; never start two replies in a row in the same way, and never reuse a sentence you already said in this conversation.
- For thanks or a short acknowledgement, give one very brief friendly line in your own words, and do not restart what you were explaining. Say goodbye naturally when the customer is ending the call, in your own warm words.
- If asked who you are or what your name is, answer briefly with your name and that you are the receptionist here — this is never UNKNOWN.
- You remember the conversation so far: build on it naturally, the way a receptionist would, instead of treating each message as the first one.

Rules:
- Speak English only. If the customer uses another language, politely ask them to speak in English.
- Answer only from INFO below. If INFO does not contain the answer, reply with exactly the single word UNKNOWN and nothing else. Never guess or invent prices, dates, policies or availability.
- Keep each reply to one to three short sentences. No lists, no markdown, no emojis, no symbols like the rupee sign; say "rupees".
- You cannot confirm bookings or check live availability. If the customer wants to book, ask for their name, dates and number of guests, and say the team will confirm.
- {intro_rule}
{('- ' + repeat_rule) if repeat_rule else ''}
{('- ' + extra) if extra else ''}
INFO:
{facts}"""


def contact_prompt() -> str:
    return ("From the customer's message, extract their name and phone number (phone as digits "
            "only). Also decide move_on: true when the customer is clearly not giving contact "
            "details now — for example they declined, changed the subject, or asked a new "
            'question. Reply with JSON only, like '
            '{"name": "Ravi", "phone": "9847012345", "move_on": false}. '
            "Use empty strings for anything not given.")


class Session:
    def __init__(self, send, mode: str = "web", wake: bool | None = None, capture_contact: bool | None = None):
        self.id = uuid.uuid4().hex[:10]
        self.send_json = send
        self.mode = mode
        s = store.load_settings()
        self.wake = s["wake_required_web"] if wake is None and mode == "web" else bool(wake)
        self.capture_contact = (mode == "web") if capture_contact is None else bool(capture_contact)
        self.history: list[dict] = []
        self.awake_until = 0.0
        self.speaking_turn = None          # turn whose audio the client is still playing
        self.pending_contact = None        # {"entry": id, "name": str, "phone": str, "attempts": int}
        self.turn_no = 0
        self.utt_no = 0
        self.task: asyncio.Task | None = None
        self.turn_sentences: dict[int, list[str]] = {}
        self.heard_upto: dict[int, int] = {}
        self.recent_agent_text: list[str] = []
        self.decide_lock = asyncio.Lock()
        # consecutive unknown turns; contact capture only arms from the second one on
        self._unknown_streak = 0
        # Last assistant reply, so the system prompt can tell the LLM not to repeat it.
        self._last_answer: str = ""
        # Per-conversation transcript and agent log file paths.
        self.transcript_path: Path | None = None
        self.agent_log_path: Path | None = None
        # v2: durable conversation row (filled on first accepted turn when db is available)
        self.conversation_id: str | None = None
        self.started_at: float = 0.0
        # Redis session store (may be None)
        self.redis_store = ENGINES.get("redis_session")
        self.db = ENGINES.get("db")
        # Greeting state: once the agent has greeted the caller, don't re-greet on every "hello".
        self._greeted: bool = False

    # ------------------------------------------------------------ helpers
    @property
    def settings(self):
        return store.load_settings()

    def awake(self) -> bool:
        return (not self.wake) or self.speaking_turn is not None or time.time() < self.awake_until

    async def send(self, obj):
        try:
            await self.send_json(obj)
        except Exception:
            pass

    async def send_state(self):
        left = max(0, int((self.awake_until - time.time()) * 1000))
        await self.send({"type": "state", "awake": self.awake(), "awake_ms": left if self.wake else None,
                         "wake_required": self.wake, "pending_contact": bool(self.pending_contact)})

    def extend_awake(self):
        self.awake_until = time.time() + self.settings["followup_seconds"]

    def is_echo(self, text: str) -> bool:
        t = _norm(text)
        if len(t) < 8:
            return False
        for said in self.recent_agent_text[-6:]:
            s = _norm(said)
            if t in s or _sim(t, s) > 0.6:
                return True
        return False

    # ------------------------------------------------------------ inputs
    async def on_audio(self, pcm: bytes):
        self.utt_no += 1
        utt = self.utt_no
        t0 = time.perf_counter()
        audio = pcm16_to_float(pcm)
        dur = audio.size / 16000
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        if dur < config.MIN_UTTERANCE_SEC:
            return await self.send({"type": "ignored", "utt": utt, "reason": "too short", "text": ""})
        if rms < config.MIN_RMS:
            return await self.send({"type": "ignored", "utt": utt, "reason": "too quiet", "text": ""})
        loop = asyncio.get_running_loop()
        s = self.settings
        try:
            tr = await loop.run_in_executor(None, ENGINES["stt"].transcribe_pcm, audio, s["stt_engine"], stt_hints(s))
        except Exception as e:
            return await self.send({"type": "error", "utt": utt, "message": f"Speech to text failed: {e}"})
        stt_ms = round((time.perf_counter() - t0) * 1000)
        await self.send({"type": "heard", "utt": utt, "text": tr.text, "lang": tr.language, "stt_ms": stt_ms,
                         "engine": s["stt_engine"]})

        text = tr.text.strip()
        reason = None
        if not re.sub(r"[\W_]+", "", text):
            reason = "no speech"
        elif tr.no_speech_prob > config.NO_SPEECH_PROB and tr.avg_logprob < -0.8:
            reason = "no speech"
        elif _norm(text) in HALLUCINATIONS and dur < 2.5 and not self.pending_contact:
            # Short backchannels ("thank you", "okay") can be genuine while a conversation
            # is open. Only drop them outright while the agent is busy speaking; otherwise
            # let the dynamic wake check judge them with the conversation context, instead
            # of a hardcoded word list.
            if self.speaking_turn is not None:
                reason = "noise"
        elif self.is_echo(text):
            reason = "echo of agent"
        if reason:
            return await self.send({"type": "ignored", "utt": utt, "reason": reason, "text": text})

        async with self.decide_lock:
                woke_now = False
                kind = "other"
                if self.wake and not self.awake():
                    # The LLM decides first, using the session history — this is what lets a
                    # garbled or name-less follow-up wake the agent. The name/greeting matcher
                    # is only an emergency fallback inside _dynamic_wake_check.
                    ok, text, dynamic, kind = await self._dynamic_wake_check(text)
                    if not ok:
                        return await self.send({"type": "ignored", "utt": utt, "reason": "no calling name", "text": text})
                    woke_now = True
                    if dynamic:
                        _log.debug("dynamic wake accepted (kind=%s): %r", kind, text)
                elif self.wake:
                    # Inside the follow-up window: keep the conversation flowing, accept what
                    # got past the noise filters and strip a calling name if one was said.
                    # Whether it is a greeting/goodbye/thanks is routed in run_turn.
                    ok, rest = match_wake(text, self.settings["agent_name"])
                    if ok and rest:
                        text = rest
                non_english = tr.language != "en" and tr.language_prob >= config.NON_ENGLISH_THRESHOLD
                await self.start_turn(text, utt=utt, t0=t0, stt_ms=stt_ms, non_english=non_english, kind=kind)

    async def on_text(self, text: str):
        self.utt_no += 1
        async with self.decide_lock:
            await self.start_turn(text.strip(), utt=self.utt_no, t0=time.perf_counter(), stt_ms=0)

    async def on_progress(self, turn: int, index: int):
        self.heard_upto[turn] = max(self.heard_upto.get(turn, -1), index)

    async def on_interrupt(self, turn: int):
        """The caller started talking over the agent: stop the reply now, before the new words are transcribed."""
        if turn and turn != self.turn_no:
            return
        await self.cancel_current()
        self.speaking_turn = None
        self.extend_awake()
        await self.send_state()

    async def on_idle(self, turn: int):
        if turn == self.speaking_turn or turn == self.turn_no:
            self.speaking_turn = None
            self.extend_awake()
            await self.send_state()

    async def reset(self):
        await self.cancel_current()
        self.history.clear()
        self.pending_contact = None
        self.awake_until = 0
        self.speaking_turn = None
        self._greeted = False
        self._last_answer = ""
        self.transcript_path = None
        self.agent_log_path = None
        await self._sync_session_state()
        await self.send_state()

    # ------------------------------------------------------------ transcript / agent log

    def _ensure_log_files(self):
        """Create per-conversation transcript.txt and agent_log.txt in DATA_DIR."""
        if self.transcript_path is not None:
            return
        from store import DATA_DIR
        tag = f"{self.conversation_id or self.id}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.transcript_path = DATA_DIR / f"transcript-{tag}.txt"
        self.agent_log_path = DATA_DIR / f"agent_log-{tag}.txt"
        header = f"# Conversation transcript — {time.strftime('%Y-%m-%d %H:%M:%S')} | session={self.id}\n"
        self.transcript_path.write_text(header, encoding="utf-8")
        self.agent_log_path.write_text(header, encoding="utf-8")

    def _log_transcript(self, role: str, text: str):
        self._ensure_log_files()
        line = f"{'=' * 40}\n[{time.strftime('%H:%M:%S')}] {role.upper()}: {text}\n{'=' * 40}\n"
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(line)

    def _log_agent(self, role: str, text: str, extra: str = ""):
        self._ensure_log_files()
        line = f"[{time.strftime('%H:%M:%S')}] {role}: {text}"
        if extra:
            line += f"  ||  {extra}"
        line += "\n"
        with open(self.agent_log_path, "a", encoding="utf-8") as f:
            f.write(line)

    # ------------------------------------------------------------ v2 session sync

    async def _sync_session_state(self):
        if not self.redis_store:
            return
        try:
            await self.redis_store.save_session(self.id, self._state_dict(), self.history)
        except Exception as e:
            _log = __import__("logging").getLogger("session")
            _log.debug("v2 session sync skipped: %s", e)

    def _recent_exchange(self, limit: int = 6) -> str:
        """Render the last few conversation turns so LLM decisions can use the session history."""
        s = self.settings
        lines = []
        for m in self.history[-limit:]:
            who = "Customer" if m["role"] == "user" else "You"
            lines.append(f"{who}: {m['content']}")
        return "\n".join(lines)

    async def _dynamic_wake_check(self, text: str) -> tuple[bool, str, bool, str]:
        """Decide dynamically (LLM, using the session history) whether *text* deserves a reply.

        Returns (ok, rest, dynamic, kind):
        - ok=True means wake the session; rest is the text to continue with. When the transcript
          has an obvious speech-to-text slip (e.g. 'praise ranges' garbled during a discussion
          about prices), rest is a lightly corrected wording so retrieval and the answer track
          what the customer actually meant.
        - kind labels the utterance ('greeting' | 'goodbye' | 'thanks' | 'question' | 'other')
          so the turn can be routed without keyword lists.
        - dynamic=True means the LLM made the decision.

        The name/greeting matcher below the LLM is only an emergency fallback for when the
        LLM is unreachable — the LLM decides in normal operation.
        """
        llm = ENGINES.get("llm")
        if not llm:
            return self._wake_fallback(text)
        s = self.settings
        convo = self._recent_exchange()
        prompt = (
            f"You are {s['agent_name']}, the receptionist at {s['business_name']}, a "
            f"{s.get('business_type') or 'business'}. Speech-to-text produced this transcript "
            f"of what the customer said:\n\"" + text + "\"\n"
        )
        if convo:
            prompt += "\nRecent part of your conversation with this customer:\n" + convo + "\n"
        else:
            prompt += ("\nThere is no recent conversation with this customer yet, so be strict: "
                       "only accept it if it clearly names you or the business, or is a full "
                       "question a receptionist would answer.\n")
        prompt += (
            "\nDecide:\n"
            "- wake: whether the customer would expect a reply. true for a real question, "
            "request, greeting, correction, thanks or acknowledgement, or a short follow-up "
            "to what you were just discussing — even when the transcript came out garbled or "
            "the customer forgot to say your name. false for background noise, another "
            "conversation, phantom transcriptions or meaningless words.\n"
            "- kind: the closest label — \"greeting\" (hello or who-are-you opener), "
            "\"goodbye\" (ending the conversation), \"thanks\" (thank you or a short "
            "acknowledgement), \"question\" (asking anything), \"other\".\n"
            "- clean: if wake=true and the transcript contains an obvious transcription slip, "
            "a corrected wording; otherwise the transcript unchanged.\n"
            'Reply with JSON only, like {"wake": true, "kind": "question", '
            '"clean": "price ranges"} or {"wake": false, "kind": "", "clean": ""}.'
        )
        try:
            resp = await asyncio.wait_for(
                llm.complete([{"role": "user", "content": prompt}], max_tokens=28),
                timeout=4.0,
            )
            m = re.search(r"\{.*\}", resp, re.S)
            if not m:
                return self._wake_fallback(text)
            data = json.loads(m.group(0))
            clean = str(data.get("clean") or "").strip() or text
            kind = str(data.get("kind") or "").strip().lower()
            if kind not in ("greeting", "goodbye", "thanks", "question", "other"):
                kind = "other"
            return bool(data.get("wake")), clean, True, kind
        except Exception as e:
            _log.debug("wake check fell back to the name/greeting matcher: %s", e)
            return self._wake_fallback(text)

    def _wake_fallback(self, text: str) -> tuple[bool, str, bool, str]:
        """Emergency wake decision when the LLM is unavailable (production safety)."""
        ok, rest = match_wake(text, self.settings["agent_name"])
        return ok, rest, False, "other"

    async def _route_short(self, text: str) -> str:
        """Classify a short message with the LLM (plus conversation context) so routing
        into goodbye / formal greeting / brief-thanks never relies on keyword lists."""
        llm = ENGINES.get("llm")
        if not llm:
            return self._route_fallback(text)
        s = self.settings
        convo = self._recent_exchange()
        prompt = (
            f"You are {s['agent_name']}, the receptionist at {s['business_name']}. "
            f"The customer says: \"" + text + "\"\n"
        )
        if convo:
            prompt += "\nRecent part of your conversation with this customer:\n" + convo + "\n"
        prompt += (
            "\nChoose the kind of this message:\n"
            "- greeting: a hello or 'who are you' style opener\n"
            "- goodbye: they are ending the conversation\n"
            "- thanks: thanking you or a short acknowledgement\n"
            "- other: anything else (questions, requests, business matters)\n"
            'Reply with JSON only, like {"kind": "other"}.'
        )
        try:
            resp = await asyncio.wait_for(
                llm.complete([{"role": "user", "content": prompt}], max_tokens=16),
                timeout=3.0,
            )
            m = re.search(r"\{.*\}", resp, re.S)
            if m:
                data = json.loads(m.group(0))
                kind = str(data.get("kind") or "").strip().lower()
                if kind in ("greeting", "goodbye", "thanks"):
                    return kind
            return "other"
        except Exception as e:
            _log.debug("short-message router fell back to the matcher: %s", e)
            return self._route_fallback(text)

    def _route_fallback(self, text: str) -> str:
        """Emergency routing when the LLM is unavailable (production safety)."""
        t = text.strip().lower()
        if t in ("bye", "goodbye", "bye bye", "good bye"):
            return "goodbye"
        if SMALLTALK.match(text.strip()) and len(text.split()) <= 3:
            return "greeting"
        if SMALLTALK.match(text.strip()):
            return "thanks"
        return "other"

    async def _standalone_query(self, text: str) -> str:
        """Rewrite a short follow-up into a self-contained retrieval query using the
        conversation history: "how much is it?" becomes e.g. "price for the sea view
        deluxe room". Returns the original text when there is nothing to resolve."""
        llm = ENGINES.get("llm")
        convo = self._recent_exchange()
        s = self.settings
        if not llm or not convo:
            return text
        prompt = (
            f"You are the receptionist at {s['business_name']}. Below is the recent part "
            f"of your conversation with this customer, followed by their latest message.\n"
            + convo + "\n\n"
            "Latest message: \"" + text + "\"\n\n"
            "Rewrite the latest message as one short self-contained search query that can be "
            "looked up in the knowledge base on its own: replace words like 'it', 'that one', "
            "'also', 'what about', 'tell me more' with the actual topic names they refer to in "
            "the conversation. If the latest message is already self-contained, return it "
            "unchanged. Reply with the query text only — no quotes and no explanation."
        )
        try:
            resp = await asyncio.wait_for(
                llm.complete([{"role": "user", "content": prompt}], max_tokens=50),
                timeout=3.0,
            )
            out = resp.strip().strip('"').strip()
            out = out.rstrip(".").strip()
            # Ignore nonsense or overly long outputs; the raw message is the fallback.
            if 2 < len(out) <= 160 and not out.startswith(("http", "{")):
                return out
            return text
        except Exception as e:
            _log.debug("standalone query rewrite skipped: %s", e)
            return text

    def _state_dict(self) -> dict:
        return {
            "turn_no": self.turn_no,
            "utt_no": self.utt_no,
            "awake_until": self.awake_until,
            "speaking_turn": self.speaking_turn,
            "pending_contact": self.pending_contact,
            "mode": self.mode,
            "wake": self.wake,
            "capture_contact": self.capture_contact,
        }

    # ------------------------------------------------------------ turns
    async def cancel_current(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
        # keep only what the caller actually heard of an interrupted reply
        prev = self.turn_no
        if prev in self.turn_sentences and self.history and self.history[-1]["role"] == "assistant":
            heard = self.turn_sentences[prev][: self.heard_upto.get(prev, -1) + 1]
            full = self.turn_sentences[prev]
            if len(heard) < len(full):
                self.history[-1]["content"] = (" ".join(heard) + " ...") if heard else "(interrupted)"

    async def start_turn(self, text: str, utt: int, t0: float, stt_ms: int, non_english=False, kind="other"):
        await self.cancel_current()
        self.turn_no += 1
        turn = self.turn_no
        self.extend_awake()
        await self.send({"type": "accepted", "utt": utt, "turn": turn, "text": text})
        # Log user utterance to transcript and agent log
        self._log_transcript("user", text)
        self._log_agent("user", text, f"utt={utt} stt_ms={stt_ms} non_english={non_english}")
        # v2: begin durable conversation on first accepted turn
        if not self.conversation_id and self.db:
            try:
                conv = await self.db.start_conversation(self.id, self.mode)
                self.conversation_id = conv["id"]
                self.started_at = time.time()
            except Exception as e:
                _log = __import__("logging").getLogger("session")
                _log.debug("v2 start_conversation skipped: %s", e)
        self.task = asyncio.create_task(self.run_turn(turn, text, t0, stt_ms, non_english, kind))

    async def speak(self, turn: int, sentences: list[str], t0: float, timings: dict):
        """Send fixed sentences (no LLM)."""
        s = self.settings
        loop = asyncio.get_running_loop()
        self.turn_sentences[turn] = []
        for i, sent in enumerate(sentences):
            self.turn_sentences[turn].append(sent)
            await self.send({"type": "sentence", "turn": turn, "index": i, "text": sent})
            wav = await loop.run_in_executor(None, ENGINES["tts"].synthesize, sent, s["voice"], s["speed"])
            if i == 0:
                timings["first_audio_ms"] = round((time.perf_counter() - t0) * 1000)
            self.speaking_turn = turn
            await self.send({"type": "audio", "turn": turn, "index": i, "wav_b64": base64.b64encode(wav).decode()})
        self.recent_agent_text += sentences

    async def run_turn(self, turn, user_text, t0, stt_ms, non_english, kind="other"):
        s = self.settings
        timings = {"stt_ms": stt_ms}
        sources, unknown = [], False
        try:
            # Short messages are routed by the LLM with the conversation context (greeting /
            # goodbye / thanks get their own handling instead of keyword matching). Longer
            # texts cannot be a hello or a farewell, so they skip the router entirely.
            if kind not in ("greeting", "goodbye", "thanks") and len(user_text.split()) <= 5:
                kind = await self._route_short(user_text)

            # End-of-session goodbye: the configured goodbye text, chosen by the router.
            if kind == "goodbye":
                goodbye = s.get("goodbye_text", "Goodbye! Have a nice day.")
                await self.speak(turn, [goodbye], t0, timings)
                self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": goodbye}]
                self._greeted = True
                return await self.finish(turn, t0, timings, sources)

            # First greeting: deliver the formal greeting once per session.
            if not self._greeted and kind == "greeting":
                g = s["greeting"].format(agent=s["agent_name"], business=s["business_name"])
                await self.speak(turn, [g], t0, timings)
                self.history += [{"role": "assistant", "content": g}]
                self._greeted = True
                return await self.finish(turn, t0, timings, sources)
            if non_english:
                await self.speak(turn, [ENGLISH_ONLY_REPLY], t0, timings)
                return await self.finish(turn, t0, timings, sources)
            if not user_text:
                await self.speak(turn, [NOT_HEARD_REPLY], t0, timings)
                return await self.finish(turn, t0, timings, sources)
            if self.pending_contact:
                handled = await self.contact_step(turn, user_text, t0, timings)
                if handled:
                    return await self.finish(turn, t0, timings, sources)

            # retrieval
            tr0 = time.perf_counter()
            kb = ENGINES["kb"]
            loop = asyncio.get_running_loop()
            context = await loop.run_in_executor(None, kb.search, user_text)
            prev_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), "")
            if prev_user and len(user_text.split()) < 8:
                extra = await loop.run_in_executor(None, kb.search, f"{prev_user} {user_text}", 2)
                have = {c.id for c, _ in context}
                context += [(c, sc) for c, sc in extra if c.id not in have][:1]
            best = max((sc for _, sc in context), default=0.0)
            # Follow-up handling: "and the price?", "how much is it?", "what about that one?"
            # only make sense together with earlier turns. When the plain query retrieves
            # nothing relevant (and there is conversation history to resolve against), an LLM
            # rewrites the message into a standalone question using the session history and
            # the search runs again. Self-contained questions never pay this extra latency.
            timings["rewrite_ms"] = 0
            if (best < s["min_retrieval_score"] and self.history
                    and kind not in ("greeting", "goodbye", "thanks")
                    and len(user_text.split()) <= 8):
                rw0 = time.perf_counter()
                standalone = await self._standalone_query(user_text)
                if standalone and standalone.strip().lower() != user_text.strip().lower():
                    extra2 = await loop.run_in_executor(None, kb.search, standalone)
                    have2 = {c.id for c, _ in context}
                    context += [(c, sc) for c, sc in extra2 if c.id not in have2][:2]
                timings["rewrite_ms"] = round((time.perf_counter() - rw0) * 1000)
            timings["retrieval_ms"] = round((time.perf_counter() - tr0) * 1000)
            sources = [{"title": c.title, "score": round(sc, 3)} for c, sc in context]
            best = max((sc for _, sc in context), default=0.0)
            small_talk = kind in ("greeting", "goodbye", "thanks")

            # Guardrail: detect out-of-domain questions before asking the LLM.
            # When in scope (or when the check is unavailable), the LLM itself decides:
            # identity/meta questions ("who are you") are answered directly, and only a
            # real UNKNOWN triggers the fallback — never a good retrieval score alone.
            ood_reply = None
            if not small_talk and best < s["min_retrieval_score"]:
                ood_reply = await self._check_out_of_domain(user_text, context)
            if ood_reply:
                await self.speak(turn, [ood_reply], t0, timings)
                self.history += [{"role": "user", "content": user_text},
                                 {"role": "assistant", "content": ood_reply}]
                # A scope redirect is a complete reply: do not also run the unknown
                # fallback (which would double-speak and ask for contact details).
                self._unknown_streak = 0
            else:
                unknown = await self.llm_reply(turn, user_text, context, t0, timings, sources=sources)
                if not unknown:
                    self._unknown_streak = 0
            if unknown:
                await self.handle_unknown(turn, user_text, t0, timings)
            else:
                # A full reply happened, so a later "hello" must not re-introduce the agent.
                self._greeted = True
            await self.finish(turn, t0, timings, sources, unknown)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.send({"type": "error", "turn": turn, "message": f"{type(e).__name__}: {e}"})
            self.speaking_turn = None

    async def llm_reply(self, turn, user_text, context, t0, timings, sources: list | None = None) -> bool:
        """Stream the LLM reply into TTS. Returns True if the model said it does not know."""
        s = self.settings
        # Pass greeting state and the last answer so the LLM avoids repetition.
        if config.V2_RAG:
            prompt_text = system_prompt_v2(s, context, user_text, greeted=self._greeted,
                                           prev_answer=self._last_answer)
        else:
            prompt_text = system_prompt(s, context, greeted=self._greeted,
                                         prev_answer=self._last_answer)
        messages = [{"role": "system", "content": prompt_text}]
        messages += self.history[-config.HISTORY_TURNS * 2:]
        messages.append({"role": "user", "content": user_text})
        self.history.append({"role": "user", "content": user_text})
        self.turn_sentences[turn] = []

        queue: asyncio.Queue = asyncio.Queue()
        reply_parts: list[str] = []
        state = {"unknown": False}

        async def produce():
            splitter, head, released, first = SentenceSplitter(), "", False, True
            try:
                async for delta in ENGINES["llm"].stream(messages):
                    if first:
                        timings["llm_first_token_ms"] = round((time.perf_counter() - t0) * 1000)
                        first = False
                    reply_parts.append(delta)
                    if not released:
                        head += delta
                        if "UNKNOWN" in head.upper().replace(" ", ""):
                            state["unknown"] = True
                            return
                        if len(head) < 14:
                            continue
                        released, delta = True, head
                    for sent in splitter.feed(delta):
                        await queue.put(sent)
                if not released and head:
                    if "UNKNOWN" in head.upper():
                        state["unknown"] = True
                        return
                    splitter.feed(head)
                for sent in splitter.flush():
                    await queue.put(sent)
            except Exception as e:
                await queue.put(e)
            finally:
                timings["llm_done_ms"] = round((time.perf_counter() - t0) * 1000)
                await queue.put(None)

        producer = asyncio.create_task(produce())
        loop = asyncio.get_running_loop()
        idx, spoken = 0, []
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                sent = clean_for_speech(item.replace("UNKNOWN", ""))
                if not sent:
                    continue
                spoken.append(sent)
                self.turn_sentences[turn].append(sent)
                await self.send({"type": "sentence", "turn": turn, "index": idx, "text": sent})
                wav = await loop.run_in_executor(None, ENGINES["tts"].synthesize, sent, s["voice"], s["speed"])
                if idx == 0:
                    timings["first_audio_ms"] = round((time.perf_counter() - t0) * 1000)
                self.speaking_turn = turn
                await self.send({"type": "audio", "turn": turn, "index": idx,
                                 "wav_b64": base64.b64encode(wav).decode()})
                idx += 1
        finally:
            if not producer.done():
                producer.cancel()
            if spoken:
                full = " ".join(spoken)
                # v2 guardrail: if the new reply heavily repeats the previous one, trim the
                # stored text so the next turn's repeat-avoidance sees a clean prior reply.
                # The audio for this turn was already streamed, so we only adjust what is
                # recorded in history and _last_answer — not what the client already heard.
                if self._last_answer and len(spoken) >= 1:
                    prev_norm = _norm(self._last_answer)
                    cur_norm = _norm(full)
                    if prev_norm and cur_norm and prev_norm != cur_norm:
                        cut = _trim_repetition(full, self._last_answer)
                        if cut:
                            full = cut
                # v2 guardrail: append a dynamic citation note if the answer claims
                # something the sources do not back. Send it as its own sentence so the
                # client renders it once, and keep turn_sentences in sync.
                if config.V2_RAG and sources:
                    guard = Guardrails(s, retrieval_best=None)
                    note = guard.citation_note(full, sources)
                    if note:
                        note_text = note.strip()
                        # Only add the note if it is not already (roughly) present in what
                        # the model streamed — avoids duplicating a caveat the model itself
                        # already said.
                        already = any(_sim(note_text, s) > 0.55 for s in spoken)
                        if not already:
                            full = full.rstrip() + note
                            self.turn_sentences[turn].append(note_text)
                            idx += 1
                            await self.send({"type": "sentence", "turn": turn, "index": idx, "text": note_text})
                            wav = await loop.run_in_executor(None, ENGINES["tts"].synthesize, note_text, s["voice"], s["speed"])
                            await self.send({"type": "audio", "turn": turn, "index": idx,
                                             "wav_b64": base64.b64encode(wav).decode()})
                self.history.append({"role": "assistant", "content": full})
                self.recent_agent_text += spoken
                self._last_answer = full   # remember for next turn's repeat-avoidance
            elif state["unknown"]:
                self.history.pop()   # user line is re-added by handle_unknown
        return state["unknown"] and not spoken

    async def handle_unknown(self, turn, user_text, t0, timings):
        s = self.settings
        # v2: also log to SQLite unanswered store when available
        extra_note = ""
        if self.db:
            try:
                await self.db.log_unanswered_v2(self.conversation_id or "", self.id, self.mode, user_text)
            except Exception as e:
                _log = __import__("logging").getLogger("session")
                _log.debug("v2 log_unanswered_v2 skipped: %s", e)
        entry = store.log_unanswered(user_text, self.id, self.mode)
        self._unknown_streak += 1
        # Ask for contact details only after several unanswerable questions in a row;
        # a single miss mid-conversation must not hijack the next question.
        ask_contact = self.capture_contact and self._unknown_streak >= 2
        lines = [s["fallback_text"]]
        if ask_contact:
            lines.append("May I have your name and phone number?")
            self.pending_contact = {"entry": entry["id"], "name": "", "phone": "", "attempts": 0}
        if not self.history or self.history[-1].get("content") != user_text:
            self.history.append({"role": "user", "content": user_text})
        await self.speak(turn, lines, t0, timings)
        self.history.append({"role": "assistant", "content": " ".join(lines)})

    async def contact_step(self, turn, text, t0, timings) -> bool:
        """Collect name and phone after an unanswered question. Returns False if the caller moved on to a new question."""
        pc = self.pending_contact
        name, phone = "", extract_phone(text)
        move_on = False
        try:
            raw = await ENGINES["llm"].complete(
                [{"role": "system", "content": contact_prompt()}, {"role": "user", "content": text}], max_tokens=40)
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            name = str(data.get("name", "")).strip()[:60]
            phone = phone or re.sub(r"\D", "", str(data.get("phone", "")))
            # The LLM judges (with the message as a whole) whether the caller moved on.
            move_on = bool(data.get("move_on"))
        except Exception:
            move_on = bool(DECLINE.match(text.strip()))   # emergency fallback, LLM unreachable
        if len(phone) < 6:
            phone = ""
        if name.lower() in ("", "none", "null", "unknown"):
            name = ""
        if not name and not phone:
            if move_on:
                store.update_unanswered(pc["entry"], note="caller declined to leave contact details")
                self.pending_contact = None
                await self.speak(turn, ["No problem. I've logged your question and the team will look into it."], t0, timings)
                return True
            pc["attempts"] += 1
            if pc["attempts"] >= 1:
                # nothing given on this message: stop chasing contact details and let the
                # turn flow to the normal answer path instead of swallowing real questions
                store.update_unanswered(pc["entry"], note="no contact details given")
                self.pending_contact = None
                return False
            await self.speak(turn, ["Sorry, could you tell me your name and phone number?"], t0, timings)
            return True
        pc["name"] = name or pc["name"]
        pc["phone"] = phone or pc["phone"]
        store.update_unanswered(pc["entry"], name=pc["name"], phone=pc["phone"])
        if pc["name"] and pc["phone"]:
            self.pending_contact = None
            line = (f"Thank you, {pc['name']}. I have noted {speak_digits(pc['phone'])}. "
                    "Someone will call you back soon.")
            await self.speak(turn, [line], t0, timings)
            self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": line}]
            return True
        pc["attempts"] += 1
        if pc["attempts"] >= 3:
            self.pending_contact = None
            await self.speak(turn, ["Thank you. I've noted that and someone will get back to you."], t0, timings)
            return True
        ask = "And your phone number, please?" if not pc["phone"] else "And may I have your name?"
        await self.speak(turn, [ask], t0, timings)
        return True

    async def _check_out_of_domain(self, user_text: str, context: list) -> str | None:
        """LLM-based scope guardrail.

        Returns a short polite redirect when the utterance is clearly outside
        {business_name}'s domain, or None when it is — or might be — about our business,
        so the normal answer / fallback flow can handle it.

        The decision is made by the LLM from the business identity in settings, the topics
        our knowledge base covers, and the recent conversation. No hardcoded keyword lists:
        the guardrail keeps working as the knowledge base and settings change.
        """
        s = self.settings
        llm = ENGINES.get("llm")
        if not llm:
            return None

        kb = ENGINES.get("kb")
        topics: list[str] = []
        if kb:
            seen: set[str] = set()
            for c in kb.chunks:
                t = re.sub(r"\s*\(part \d+\)$", "", c.title).strip()
                if ":" in t:
                    t = t.split(":")[-1].strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower())
                    topics.append(t)

        prompt = (
            f"You are {s['agent_name']}, the receptionist at {s['business_name']}, a "
            f"{s.get('business_type') or 'hospitality'} business. You only answer questions "
            f"about this business and its services.\n"
        )
        if topics:
            prompt += "\nTopics you can talk about: " + "; ".join(topics[:25]) + ".\n"
        convo = self._recent_exchange()
        if convo:
            prompt += "\nRecent part of your conversation with this customer:\n" + convo + "\n"
        prompt += (
            "\nThe customer just said:\n\"" + user_text + "\"\n"
            "in_scope=true when a receptionist here would treat this as a question meant for "
            "us — anything about our rooms, prices, facilities, food, activities, events, "
            "location, policies or bookings, or a short follow-up to what you were just "
            "discussing, even if it does not name the business.\n"
            "in_scope=false for general knowledge, news, other businesses or places, personal "
            "or professional advice, or anything unrelated to us.\n"
            "When in_scope=false, write a short warm reply of one sentence (20 words max) that "
            "politely steers the customer back to what you can help with; do not answer the "
            "question itself.\n"
            'Reply with JSON only, like {"in_scope": true, "reply": ""} or '
            '{"in_scope": false, "reply": "I can only help with questions about our resort."}. '
            f"Name {s['business_name']} in the reply only if it fits naturally."
        )
        try:
            resp = await asyncio.wait_for(
                llm.complete([{"role": "user", "content": prompt}], max_tokens=40),
                timeout=4.0,
            )
            m = re.search(r"\{.*\}", resp, re.S)
            if not m:
                return None
            data = json.loads(m.group(0))
            if data.get("in_scope"):
                return None
            answer = str(data.get("reply") or "").strip()
            if not answer:
                return None
            return clean_for_speech(answer)
        except Exception as e:
            # Fail open: if the scope check is unavailable, the normal answer / UNKNOWN
            # fallback flow still handles the turn safely.
            _log.debug("scope guardrail fell back to normal flow: %s", e)
            return None

    async def finish(self, turn, t0, timings, sources, unknown=False):
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000)
        if self.speaking_turn != turn:
            self.speaking_turn = None
        self.extend_awake()
        del self.history[: max(0, len(self.history) - config.HISTORY_TURNS * 2)]
        # v2: durable turn log
        if self.conversation_id and self.db:
            try:
                last_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), "")
                last_assistant = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), "")
                retrieval_best = None
                for s in (sources or []):
                    if isinstance(s, dict) and "score" in s and retrieval_best is None:
                        retrieval_best = float(s["score"])
                await self.db.add_turn(
                    self.conversation_id,
                    turn,
                    last_user,
                    last_assistant,
                    unknown,
                    retrieval_best or 0.0,
                    timings.get("total_ms", 0),
                    sources,
                )
            except Exception as e:
                _log = __import__("logging").getLogger("session")
                _log.debug("v2 add_turn skipped: %s", e)
        await self._sync_session_state()
        await self.send({"type": "done", "turn": turn, "timings": timings, "sources": sources, "unknown": unknown})
        # Log agent response to transcript and agent log
        last_assistant = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), "")
        if last_assistant:
            self._log_transcript("agent", last_assistant)
            extra = f"turn={turn} total_ms={timings.get('total_ms', 0)} unknown={unknown}"
            if sources:
                extra += f" sources={len(sources)}"
            self._log_agent("agent", last_assistant, extra)
        await self.send_state()
