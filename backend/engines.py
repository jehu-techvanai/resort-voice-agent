"""Model wrappers: speech to text, LLM streaming, sentence splitting and text to speech.

Each heavy model is loaded once at startup and guarded by a lock, because one machine (and at most one
small GPU) is shared by all of them. In MOCK mode lightweight stand-ins are used so the pipeline and UI
can be tested without downloads.
"""
import asyncio
import io
import json
import math
import re
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field

import httpx
import numpy as np

import config

SAMPLE_RATE_IN = 16000


# ---------------------------------------------------------------- speech to text

@dataclass
class Transcript:
    text: str
    language: str
    language_prob: float
    duration: float = 0.0
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0


def decode_audio_ffmpeg(audio_bytes: bytes, sr: int = SAMPLE_RATE_IN) -> np.ndarray:
    """Decode browser audio (webm/opus, mp4, ogg, wav) to 16 kHz mono float32 with the ffmpeg CLI.
    Avoids faster-whisper's PyAV decoder, whose API changes between PyAV versions."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
         "-f", "s16le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=audio_bytes, capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode the recording: {proc.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


WHISPER_MODELS = {
    "whisper-small": "small",
    "whisper-medium": "medium",
    "whisper-large-v3-turbo": "large-v3-turbo",
}
STT_ENGINES = list(WHISPER_MODELS) + ["sarvam"]


def to_wav16k(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE_IN)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class STT:
    """Speech to text with switchable engines: local Whisper (small / medium / large-v3-turbo) or Sarvam (cloud).

    Accent handling:
    - Whisper: language is detected first; anything plausibly English is then decoded as English, with a prompt
      holding the agent name, business name and knowledge-base headings (resort, room and place names).
    - Sarvam saaras:v4 is trained on Indian speech; the same names go in as key terms.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.load_lock = threading.Lock()
        self.models = {}
        self.mock_queue: list[str] = []   # tests push fake transcripts here
        self.http = httpx.Client(timeout=httpx.Timeout(20.0, connect=5.0))

    # ---- loading
    def _whisper(self, engine: str):
        name = WHISPER_MODELS.get(engine, config.STT_MODEL)
        with self.load_lock:
            if name not in self.models:
                from faster_whisper import WhisperModel
                t0 = time.perf_counter()
                try:
                    m = WhisperModel(name, device=config.DEVICE, compute_type=config.STT_COMPUTE)
                except Exception:
                    if name != "large-v3-turbo":
                        raise
                    m = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device=config.DEVICE,
                                     compute_type=config.STT_COMPUTE)
                self.models[name] = m
                print(f"[stt] loaded whisper {name} in {time.perf_counter() - t0:.1f}s")
            return self.models[name]

    def preload(self, engine: str):
        if config.MOCK or engine == "sarvam":
            return
        m = self._whisper(engine)
        noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE_IN) * 0.01).astype(np.float32)
        with self.lock:
            list(m.transcribe(noise, beam_size=1, language="en", without_timestamps=True)[0])

    # ---- transcription
    def transcribe(self, audio_bytes: bytes, engine: str = "whisper-small", hints=None) -> Transcript:
        """Encoded audio (webm etc.)."""
        if config.MOCK:
            return self._mock(len(audio_bytes) / 4000)
        return self.transcribe_pcm(decode_audio_ffmpeg(audio_bytes), engine, hints)

    def transcribe_pcm(self, audio: np.ndarray, engine: str = "whisper-small", hints=None) -> Transcript:
        """16 kHz mono float32 samples, as sent by the browser VAD."""
        dur = audio.size / SAMPLE_RATE_IN
        if config.MOCK:
            return self._mock(dur)
        if audio.size < 1600:
            return Transcript("", "en", 1.0, dur, 1.0, -5.0)
        hints = [h for h in (hints or []) if h][:50]
        if engine == "sarvam":
            return self._sarvam(audio, dur, hints)
        return self._whisper_transcribe(audio, dur, engine, hints)

    def _whisper_transcribe(self, audio, dur, engine, hints) -> Transcript:
        m = self._whisper(engine)
        prompt = ("Customer call. " + ", ".join(hints) + ".")[:400] if hints else None
        with self.lock:
            lang, prob, en_prob = "en", 1.0, 1.0
            try:
                lang, prob, allp = m.detect_language(audio)
                en_prob = dict(allp).get("en", 0.0)
            except Exception:
                pass   # older faster-whisper: assume English
            clearly_other = lang != "en" and prob >= config.NON_ENGLISH_THRESHOLD and en_prob < 0.05
            segments, info = m.transcribe(
                audio, beam_size=1, vad_filter=True, condition_on_previous_text=False, without_timestamps=True,
                language=None if clearly_other else "en",
                initial_prompt=None if clearly_other else prompt,
            )
            segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if prompt and text and text.rstrip(".") in prompt:   # prompt echoed back on silence
            text = ""
        nsp = max((s.no_speech_prob for s in segs), default=1.0)
        alp = float(np.mean([s.avg_logprob for s in segs])) if segs else -5.0
        return Transcript(text, lang if clearly_other else "en", float(prob), dur, float(nsp), alp)

    def _sarvam(self, audio, dur, hints) -> Transcript:
        if not config.SARVAM_API_KEY:
            raise RuntimeError("Sarvam selected but SARVAM_API_KEY is not set in .env")
        data = {"model": config.SARVAM_MODEL, "language_code": config.SARVAM_LANGUAGE}
        if hints and config.SARVAM_MODEL.endswith("v4"):
            data["keyterms"] = json.dumps([h[:64] for h in hints])
        r = self.http.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": config.SARVAM_API_KEY},
            files={"file": ("speech.wav", to_wav16k(audio), "audio/wav")},
            data=data,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Sarvam STT error {r.status_code}: {r.text[:200]}")
        j = r.json()
        code = (j.get("language_code") or "en-IN").split("-")[0].lower()
        prob = j.get("language_probability")
        text = (j.get("transcript") or "").strip()
        return Transcript(text, code, float(prob) if prob is not None else 1.0, dur,
                          0.0 if text else 1.0, 0.0 if text else -5.0)

    def _mock(self, dur: float) -> Transcript:
        if self.mock_queue:
            item = self.mock_queue.pop(0)
            lang, _, text = item.partition("|") if "|" in item[:4] else ("en", "", item)
            return Transcript(text, lang, 0.95, max(dur, 1.0), 0.05, -0.3)
        if dur < 0.05:
            return Transcript("", "en", 1.0, dur, 1.0, -5.0)
        import os
        text = os.environ.get("MOCK_TRANSCRIPT", "How much is a sea view room for two nights?")
        return Transcript(text, "en", 0.99, dur, 0.05, -0.3)


# ---------------------------------------------------------------- LLM

class LLM:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(config.LLM_TIMEOUT, connect=5.0))
        self.mock_reply: str | None = None   # tests can override

    async def stream(self, messages, max_tokens: int | None = None):
        """Yield text deltas from an OpenAI-compatible /chat/completions stream."""
        if config.MOCK:
            reply = self.mock_reply or _mock_answer(messages)
            for word in reply.split(" "):
                await asyncio.sleep(0.02)
                yield word + " "
            return
        body = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": True,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
        }
        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
        async with self.client.stream("POST", f"{config.LLM_BASE_URL}/chat/completions", json=body, headers=headers) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if delta:
                    yield delta

    async def complete(self, messages, max_tokens: int = 120) -> str:
        parts = [d async for d in self.stream(messages, max_tokens=max_tokens)]
        return "".join(parts).strip()

    async def keep_loaded(self):
        """Ollama unloads a model after 5 idle minutes, so the next question waits for a reload.
        keep_alive=-1 pins it in memory until Ollama restarts. Ignored for non-Ollama servers."""
        if config.MOCK or ":11434" not in config.LLM_BASE_URL:
            return
        root = config.LLM_BASE_URL.rsplit("/v1", 1)[0]
        try:
            await self.client.post(f"{root}/api/generate", json={"model": config.LLM_MODEL, "keep_alive": -1})
        except Exception as e:
            print(f"[warmup] could not pin model in Ollama: {e}")

    async def warmup(self):
        if config.MOCK:
            return
        await self.keep_loaded()
        try:
            await self.complete([{"role": "user", "content": "Say OK."}], max_tokens=5)
        except Exception as e:  # server may still be starting; first call will just be slower
            print(f"[warmup] LLM not reachable yet: {e}")


def _mock_answer(messages) -> str:
    sys = messages[0]["content"] if messages else ""
    user = messages[-1]["content"].lower() if messages else ""
    if "extract" in sys.lower():
        name = re.search(r"(?:i am|i'm|name is|this is)\s+([a-z]+)", user)
        digits = re.sub(r"\D", "", user)
        return json.dumps({"name": name.group(1).title() if name else "", "phone": digits if len(digits) >= 6 else ""})
    if "spaceship" in user or "parking on mars" in user:
        return "UNKNOWN"
    return ("A Sea View Deluxe Room is 9,500 rupees a night in low season and 12,500 rupees in peak season, "
            "for two adults with breakfast. Taxes are extra. Would you like me to note down your dates?")


_SENTENCE_END = re.compile(r"([.!?])\s+")
_ABBREV = {"a.m.", "p.m.", "mr.", "mrs.", "ms.", "dr.", "rs.", "no.", "approx.", "e.g.", "i.e.", "etc.", "st."}


class SentenceSplitter:
    """Collect streamed tokens and release whole sentences, so TTS can start before the LLM finishes."""

    def __init__(self, min_chars: int = 20, first_clause: bool = True):
        self.buf = ""
        self.min_chars = min_chars
        self.first_clause = first_clause   # speak the first clause early so the caller hears something sooner

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        if self.first_clause and not _SENTENCE_END.search(self.buf):
            m = re.search(r",\s", self.buf[30:])
            if m and len(self.buf) > 70:
                cut = 30 + m.start() + 1
                head = self.buf[:cut].strip()
                if not re.search(r"\d,$", head):   # never split 9,500
                    self.buf = self.buf[cut:].lstrip()
                    self.first_clause = False
                    out.append(head)
        while True:
            m = _SENTENCE_END.search(self.buf)
            if not m:
                break
            cut = m.end(1)
            candidate = self.buf[:cut].strip()
            last_word = candidate.rsplit(" ", 1)[-1].lower()
            if m.group(1) == "." and last_word in _ABBREV:
                nxt = _SENTENCE_END.search(self.buf, m.end())
                if not nxt:
                    break
                cut = nxt.end(1)
                candidate = self.buf[:cut].strip()
            if len(candidate) < self.min_chars and len(self.buf) < 200:
                nxt = _SENTENCE_END.search(self.buf, m.end())
                if not nxt:
                    break
                cut = nxt.end(1)
                candidate = self.buf[:cut].strip()
            out.append(candidate)
            self.buf = self.buf[cut:].lstrip()
            self.first_clause = False
        return out

    def flush(self) -> list[str]:
        rest = self.buf.strip()
        self.buf = ""
        return [rest] if rest else []


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_#`>\[\]]+", "", text)
    text = text.replace("₹", "rupees ").replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- text to speech

SAMPLE_RATE_OUT = 24000


def to_wav(audio: np.ndarray, sr: int = SAMPLE_RATE_OUT) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


class TTS:
    """Kokoro-82M. One pipeline per accent (a = American, b = British), picked from the voice prefix."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pipelines = {}
        if not config.MOCK:
            self.synthesize("Hello.", config.TTS_VOICE)  # warm up

    def _pipeline(self, lang: str):
        if lang not in self.pipelines:
            from kokoro import KPipeline
            self.pipelines[lang] = KPipeline(lang_code=lang, device=config.DEVICE, repo_id="hexgrad/Kokoro-82M")
        return self.pipelines[lang]

    def synthesize(self, text: str, voice: str | None = None, speed: float | None = None) -> bytes:
        voice = voice or config.TTS_VOICE
        speed = speed or config.TTS_SPEED
        if config.MOCK:
            dur = max(0.4, len(text) / 15.0)
            t = np.arange(int(SAMPLE_RATE_OUT * dur)) / SAMPLE_RATE_OUT
            env = np.minimum(1, np.minimum(t, dur - t) * 20)
            time.sleep(0.03)
            return to_wav(0.15 * np.sin(2 * math.pi * 330 * t) * env)
        lang = voice[0] if voice[0] in "ab" else "a"
        with self.lock:
            pipe = self._pipeline(lang)
            parts = [a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
                     for _, _, a in pipe(text, voice=voice, speed=speed)]
        audio = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        return to_wav(audio)
