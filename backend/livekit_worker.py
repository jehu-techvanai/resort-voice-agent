"""
livekit_worker.py — LiveKit voice agent worker for inbound/outbound calls.

Uses LiveKit Agents SDK with Gemini Live for real-time voice conversations.
This worker handles:
  - Inbound calls from Twilio (via LiveKit SIP)
  - Direct LiveKit room connections
  - Outbound call dispatch

Pipeline: Phone → Twilio → LiveKit → Gemini Live → STT/LLM/TTS → LiveKit → Twilio → Phone
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents.voice import room_io
from livekit import rtc, api

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engines import STT, TTS, LLM, SentenceSplitter, clean_for_speech, pcm16_to_float
from session import Session, system_prompt, match_wake, extract_phone
from config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_PHONE_NUMBER,
    LIVEKIT_URL,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_SIP_URI,
)

log = logging.getLogger("livekit-worker")


# ============================================================================
# Configuration
# ============================================================================

AGENT_NAME = os.getenv("AGENT_NAME", "Aira")
BUSINESS_NAME = os.getenv("RESORT_NAME", "Palm Lagoon Resort")
VOICE_ID = os.getenv("TTS_VOICE", "af_heart")
VOICE_SPEED = float(os.getenv("TTS_SPEED", "1.0"))


# ============================================================================
# Helper functions
# ============================================================================

def extract_phone_from_room_name(room_name: str) -> Optional[str]:
    """
    Extract phone number from LiveKit room name.
    Room naming convention: sip:{phone} or {phone}@sip
    """
    if not room_name:
        return None

    # Extract phone from SIP URI or room name
    match = re.search(r'\+?\d{10,15}', room_name)
    if match:
        return match.group(0)

    return None


def get_call_logger(phone: str) -> logging.Logger:
    """Create a logger for a specific call."""
    logger = logging.getLogger(f"call.{phone}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(handler)
    return logger


# ============================================================================
# LiveKit Agent
# ============================================================================

class VoiceAgent(Agent):
    """LiveKit voice agent for handling phone calls."""

    def __init__(self):
        super().__init__()
        self.call_logger: Optional[logging.Logger] = None
        self.call_sid: Optional[str] = None
        self.caller_phone: Optional[str] = None
        self.session: Optional[Session] = None
        self.tts_stream: Optional[asyncio.Queue] = None

    async def setup(self, ctx: RunContext):
        """Called when the agent is being started."""
        log.info("VoiceAgent setup started")

        # Extract caller info from room
        room_name = ctx.room.name
        self.caller_phone = extract_phone_from_room_name(room_name)
        self.call_sid = ctx.room.name

        log.info("Call from: %s, room: %s", self.caller_phone, room_name)

        # Create call logger
        if self.caller_phone:
            self.call_logger = get_call_logger(self.caller_phone)
            self.call_logger.info("Call started - caller: %s", self.caller_phone)

        # Initialize session
        async def send_callback(obj):
            """Send messages to the client."""
            if hasattr(ctx, 'send_to_client'):
                await ctx.send_to_client(obj)

        self.session = Session(send_callback, mode="phone")
        self.session.wake = True  # Phone calls don't need wake word

        log.info("VoiceAgent setup complete")

    async def detect(self, ctx: JobContext):
        """Detect if this job should be handled."""
        room_name = ctx.room.name

        # Handle SIP calls (from Twilio)
        if room_name.startswith("sip:"):
            return True

        # Handle direct LiveKit room connections
        if "call" in room_name.lower() or "phone" in room_name.lower():
            return True

        return False

    async def warmup(self, proc: JobProcess):
        """Pre-warm models."""
        log.info("Warming up models...")

        # Initialize STT
        stt = STT()
        stt_engine = os.getenv("STT_ENGINE", "whisper-small")
        await asyncio.get_event_loop().run_in_executor(None, stt.preload, stt_engine)

        # Initialize TTS
        tts = TTS()
        tts.synthesize("Hello.", VOICE_ID, VOICE_SPEED)

        # Warm up LLM
        llm = LLM()
        await llm.warmup()

        log.info("Models warmed up")

        # Store in userdata for access in other methods
        proc.userdata["stt"] = stt
        proc.userdata["tts"] = tts
        proc.userdata["llm"] = llm

    async def on_connect(self, session: AgentSession, ctx: JobContext):
        """Called when agent connects to the room."""
        log.info("Agent connected to room: %s", ctx.room.name)

        # Connect to room with audio
        await session.connect(audio=True, video=False)

        # Log the connection
        if self.call_logger:
            self.call_logger.info("Agent connected to room")

    async def on_participant_joined(self, session: AgentSession, ctx: JobContext, participant: rtc.Participant):
        """Called when a participant joins the room."""
        log.info("Participant joined: %s", participant.identity)

        if self.call_logger:
            self.call_logger.info("Participant joined: %s", participant.identity)

        # If this is the SIP caller, start the conversation
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            await self.start_conversation(session, ctx)

    async def start_conversation(self, session: AgentSession, ctx: JobContext):
        """Start the voice conversation."""
        log.info("Starting conversation with %s", self.caller_phone)

        if self.call_logger:
            self.call_logger.info("Conversation started")

        # Get settings
        from store import load_settings
        settings = load_settings()

        # Generate greeting
        greeting = settings.get("greeting", "Hello!").format(
            agent=settings.get("agent_name", AGENT_NAME),
            business=settings.get("business_name", BUSINESS_NAME),
        )

        # Speak greeting
        await self.speak_text(session, greeting)

        if self.call_logger:
            self.call_logger.info("Greeting played: %s", greeting)

    async def speak_text(self, session: AgentSession, text: str):
        """Speak text using TTS."""
        log.debug("Speaking: %s", text)

        # Synthesize audio
        tts = session.proc.userdata.get("tts") or TTS()
        wav_bytes = await asyncio.get_event_loop().run_in_executor(
            None, tts.synthesize, text, VOICE_ID, VOICE_SPEED
        )

        # Play audio in the room
        try:
            source = session.room.create_audio_source(
                name=f"agent-{session.id}",
                artifact_type=rtc.ArtifactType.AUDIO,
            )

            stream = session.publish_source(source)

            # Convert WAV to frames and play
            import numpy as np
            from io import BytesIO
            import wave

            with BytesIO(wav_bytes) as f:
                with wave.open(f, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    frames = wav_file.readframes(wav_file.getnframes())

            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            # Create audio track
            track = session.room.create_local_audio_track(
                source=source,
                sample_rate=sample_rate,
            )

            # Play the audio
            buffer_size = 20  # ms
            samples_per_frame = int(sample_rate * buffer_size / 1000)

            for i in range(0, len(audio_data), samples_per_frame):
                frame = audio_data[i:i + samples_per_frame]
                if len(frame) < samples_per_frame:
                    break

                # Convert to bytes
                pcm = (frame * 32767).astype(np.int16).tobytes()

                # Write to track
                await track.write(pcm)

            # Clean up
            await track.close()

        except Exception as e:
            log.error("Error playing audio: %s", e)

    async def on_audio_frame(self, session: AgentSession, ctx: JobContext, track: rtc.Track, frame: rtc.AudioFrame):
        """Handle incoming audio frames from the caller."""
        # In production, this would:
        # 1. Buffer audio
        # 2. Detect speech end (VAD)
        # 3. Send to STT
        # 4. Process through LLM
        # 5. Speak response

        # For now, just log
        if self.call_logger:
            self.call_logger.debug(
                "Audio frame received: %d samples, %d Hz",
                len(frame.samples),
                frame.sample_rate,
            )

    async def on_sip_call_answered(self, session: AgentSession, ctx: JobContext, sip_participant: rtc.SipParticipant):
        """Called when SIP call is answered."""
        log.info("SIP call answered by %s", sip_participant.identity)

        if self.call_logger:
            self.call_logger.info("SIP call answered")

        await self.start_conversation(session, ctx)


# ============================================================================
# Twilio Media Stream Handler
# ============================================================================

# VAD / noise-gate constants for the Twilio Media Streams handler.
# Twilio sends 20 ms of 16-bit PCM per media frame by default (320 bytes at 16 kHz).
# These values are tunable via env so a production deploy can re-tune without code changes.

_VAD_SAMPLE_RATE = 16000
_VAD_FRAME_BYTES = 320  # 20 ms * 16 kHz * 2 bytes/frame
_VAD_MIN_UTTERANCE_SEC = float(os.getenv("TWILIO_VAD_MIN_UTTERANCE_SEC", "0.25"))   # ignore bursts shorter than this
_VAD_MIN_RMS = float(os.getenv("TWILIO_VAD_MIN_RMS", "0.010"))                        # silence/noise floor below this
_VAD_SPeech_RMS = float(os.getenv("TWILIO_VAD_SPEECH_RMS", "0.025"))                  # confident-speech energy above this
_BARGEIN_WINDOW_SEC = float(os.getenv("TWILIO_BARGEIN_WINDOW_SEC", "0.25"))           # how long caller must speak over agent before interrupt
_BARGEIN_RMS = float(os.getenv("TWILIO_BARGEIN_RMS", "0.020"))                        # energy needed to count as barge-in


def _audio_rms(pcm16: bytes) -> float:
    """Return RMS of 16-bit PCM bytes, normalized to [-1, 1]."""
    if not pcm16:
        return 0.0
    try:
        arr = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    except Exception:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


class TwilioMediaStreamHandler:
    """
    Handle Twilio Media Streams WebSocket connections.

    Real-time bidirectional audio with:
      - VAD / noise gate: only transcribe real speech, ignore silence and short bursts
      - Barge-in: if the caller speaks while the agent is playing audio, interrupt
        the agent, stop the outgoing audio, and process what the caller said

    Production tuning: set TWILIO_VAD_* and TWILIO_BARGEIN_* env vars (see above).
    """

    def __init__(self):
        self.connections: dict[str, dict] = {}
        self.stt = STT()
        self.tts = TTS()
        self.llm = LLM()
        # serialize STT transcribe calls per-call so two turns on the same call don't
        # race the Whisper model ( WhisperModel is not thread-safe for concurrent use).
        self._stt_locks: dict[str, asyncio.Lock] = {}

    # ---------------------------------------------------------------- connection bookkeeping

    def _ensure_lock(self, call_sid: str) -> asyncio.Lock:
        if call_sid not in self._stt_locks:
            self._stt_locks[call_sid] = asyncio.Lock()
        return self._stt_locks[call_sid]

    # ---------------------------------------------------------------- websocket entry point

    async def handle_websocket(self, websocket, call_sid: str):
        """Handle a Twilio Media Stream WebSocket connection."""
        log.info("Twilio Media Stream connected: call_sid=%s", call_sid)

        connection = {
            "websocket": websocket,
            "call_sid": call_sid,
            "stream_sid": "",
            "buffer": bytearray(),            # incoming caller audio, pending VAD decision
            "speech_energy": [],              # recent frame RMS values (for barge-in)
            "is_listening": False,            # True while we are accumulating a candidate utterance
            "speech_start_ms": 0,             # when the current candidate utterance started
            "total_speech_bytes": 0,          # bytes of speech seen in current candidate
            "minimum_bytes": int(_VAD_MIN_UTTERANCE_SEC * _VAD_SAMPLE_RATE * 2),
            "is_agent_speaking": False,       # True while we are streaming agent audio out
            "agent_outstanding": [],          # queued-but-unsent media frames (for interrupt)
            "agent_cancel_requested": False,  # set by barge-in; process loop drains it
            "last_activity_ms": 0,            # last time we saw audio energy above the floor
            "consecutive_silence_frames": 0,  # for ending an utterance
            "ended": False,
        }
        self.connections[call_sid] = connection

        try:
            async for message in websocket.iter_text():
                if connection["ended"]:
                    break
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                event = data.get("event")
                log.debug("Media Stream event: %s", event)

                if event == "media":
                    await self.handle_media(connection, data)
                elif event == "start":
                    await self.handle_start(connection, data)
                elif event == "stop":
                    await self.handle_stop(connection, data)
                elif event == "dtmf":
                    await self.handle_dtmf(connection, data)

                # after every event, drain any pending agent audio unless we got
                # interrupted. This keeps the outbound stream alive while the LLM
                # is thinking.
                if connection["is_agent_speaking"]:
                    await self._drain_agent_stream(connection)

        except Exception as e:
            log.error("Media Stream error: %s", e, exc_info=True)
        finally:
            # tell Twilio we are done playing anything
            await self._cancel_agent_audio(connection)
            if call_sid in self.connections:
                del self.connections[call_sid]
            if call_sid in self._stt_locks:
                del self._stt_locks[call_sid]
            log.info("Media Stream disconnected: call_sid=%s", call_sid)

    # ---------------------------------------------------------------- event handlers

    async def handle_start(self, connection: dict, data: dict):
        """Handle stream start event."""
        stream = data.get("stream", {})
        connection["call_sid"] = stream.get("callSid", connection["call_sid"])
        connection["stream_sid"] = stream.get("streamSid", "")

        log.info(
            "Stream started: call_sid=%s, stream_id=%s",
            connection["call_sid"],
            connection["stream_sid"],
        )

        # Send initial heartbeat so Twilio knows the connection is live
        await self.send_media(connection, {
            "event": "media",
            "streamSid": connection["stream_sid"],
            "media": {"payload": ""},
        })

    async def handle_media(self, connection: dict, data: dict):
        """Handle incoming audio media from the caller.

        This is where VAD / noise gating and barge-in live:
          1. decode the frame
          2. if the agent is speaking, check for barge-in
          3. feed the frame into the VAD state machine
        """
        media = data.get("media", {})
        payload = media.get("payload", "")
        if not payload:
            return

        try:
            audio_bytes = base64.b64decode(payload)
        except Exception:
            return

        # ---- barge-in check (caller speaking while agent is playing) ----
        if connection["is_agent_speaking"]:
            rms = _audio_rms(audio_bytes)
            now = time.time()
            connection["speech_energy"].append((now, rms))
            # keep only the recent window
            cutoff = now - _BARGEIN_WINDOW_SEC
            connection["speech_energy"] = [
                (t, e) for t, e in connection["speech_energy"] if t >= cutoff
            ]
            loud_frames = sum(1 for _, e in connection["speech_energy"] if e >= _BARGEIN_RMS)
            if loud_frames >= 2:  # at least two 20 ms frames above threshold (~40 ms)
                log.info("Barge-in detected on call %s", connection["call_sid"])
                connection["is_agent_speaking"] = False
                connection["agent_cancel_requested"] = True
                connection["speech_energy"] = []
                # treat the current frame as the start of a new utterance
                connection["is_listening"] = True
                connection["speech_start_ms"] = now * 1000
                connection["total_speech_bytes"] = 0
                connection["consecutive_silence_frames"] = 0
                connection["last_activity_ms"] = now * 1000
                await self._cancel_agent_audio(connection)
                # fall through to VAD processing of this frame below

        # ---- VAD / noise gate state machine ----
        rms = _audio_rms(audio_bytes)
        now_ms = time.time() * 1000
        connection["last_activity_ms"] = now_ms

        # prune old energy samples
        cutoff = now_ms - _BARGEIN_WINDOW_SEC * 1000
        connection["speech_energy"] = [
            (t, e) for t, e in connection["speech_energy"] if t >= cutoff
        ]

        if rms >= _VAD_MIN_RMS:
            # there is sound here
            connection["consecutive_silence_frames"] = 0
            if not connection["is_listening"]:
                # potential start of speech — wait for a minimum burst before committing
                connection["is_listening"] = True
                connection["speech_start_ms"] = now_ms
                connection["total_speech_bytes"] = 0
                connection["speech_energy"] = []

            connection["total_speech_bytes"] += len(audio_bytes)
            connection["speech_energy"].append((now_ms, rms))

            # buffer the audio so we can transcribe the whole utterance at once
            connection["buffer"].extend(audio_bytes)

            # if we have enough audio and the recent energy is clearly speech, process now
            enough = (
                connection["total_speech_bytes"] >= connection["minimum_bytes"]
                and _recent_energy_is_speech(connection)
            )
            if enough:
                await self._flush_utterance(connection)
        else:
            # silence / noise floor
            connection["consecutive_silence_frames"] += 1
            if connection["is_listening"]:
                # give speech a little time to resume (debounce the end)
                if (
                    connection["total_speech_bytes"] >= connection["minimum_bytes"]
                    and connection["consecutive_silence_frames"] >= 6   # ~120 ms of silence
                ):
                    await self._flush_utterance(connection)
                elif connection["consecutive_silence_frames"] >= 60:  # ~1.2 s silence → abandon
                    connection["is_listening"] = False
                    connection["total_speech_bytes"] = 0
                    connection["buffer"] = bytearray()
                    connection["speech_energy"] = []

    async def handle_stop(self, connection: dict, data: dict):
        """Handle stream stop event."""
        log.info("Stream stopped: call_sid=%s", connection["call_sid"])
        connection["ended"] = True
        connection["is_listening"] = False
        connection["buffer"] = bytearray()
        await self._cancel_agent_audio(connection)

    async def handle_dtmf(self, connection: dict, data: dict):
        """Handle DTMF tones."""
        dtmf = data.get("dtmf", {})
        digit = dtmf.get("digit", "")
        log.info("DTMF received: %s", digit)

        if digit == "#":
            # end of DTMF input — flush whatever we have buffered
            if connection["buffer"]:
                await self._flush_utterance(connection)
        elif digit == "*":
            # cancel / repeat: stop agent audio and discard current candidate
            connection["is_listening"] = False
            connection["total_speech_bytes"] = 0
            connection["buffer"] = bytearray()
            connection["speech_energy"] = []
            await self._cancel_agent_audio(connection)

    # ---------------------------------------------------------------- utterance pipeline

    async def _flush_utterance(self, connection: dict):
        """Transcribe the buffered utterance, reply through LLM+TTS, then reset."""
        connection["is_listening"] = False
        buf = bytes(connection["buffer"])
        connection["buffer"] = bytearray()
        connection["total_speech_bytes"] = 0
        connection["speech_energy"] = []
        connection["consecutive_silence_frames"] = 0

        if len(buf) < 1600:
            return

        await self.process_audio(connection, connection["stream_sid"], buf)

    async def process_audio(self, connection: dict, stream_sid: str = "", audio_bytes: bytes = b""):
        """Process audio through STT→LLM→TTS pipeline.

        Called with a full utterance's worth of PCM (base64-decoded by handle_media).
        When called from _flush_utterance, audio_bytes is the buffered utterance.
        When called from the old path (direct), audio_bytes may be empty and we read
        from connection["buffer"] — that path is deprecated but kept for safety.
        """
        if not audio_bytes:
            audio_bytes = bytes(connection["buffer"])
            if len(audio_bytes) < 1600:
                return

        log.debug("Processing audio: %d bytes", len(audio_bytes))

        lock = self._ensure_lock(connection["call_sid"])
        async with lock:
            try:
                audio = pcm16_to_float(audio_bytes)

                # STT
                from store import load_settings
                settings = load_settings()
                stt_engine = settings.get("stt_engine", "whisper-small")
                transcript = self.stt.transcribe_pcm(audio, stt_engine)

                text = transcript.text.strip()
                if not text:
                    return

                log.info("Transcription: %s", text)

                # LLM
                messages = [
                    {"role": "system", "content": system_prompt(settings, [])},
                    {"role": "user", "content": text},
                ]
                response_text = await self.llm.complete(messages, max_tokens=150)

                log.info("LLM response: %s", response_text)

                # TTS — stream the response back in small chunks so it can be
                # interrupted by the next barge-in.
                if not response_text:
                    return

                wav_bytes = self.tts.synthesize(response_text, VOICE_ID, VOICE_SPEED)
                await self._play_audio(connection, stream_sid, wav_bytes)

            except Exception as e:
                log.error("Audio processing error: %s", e, exc_info=True)

    # ---------------------------------------------------------------- outbound audio (interruptible)

    async def _play_audio(self, connection: dict, stream_sid: str, wav_bytes: bytes):
        """Convert a WAV to 16-bit PCM frames and stream them to Twilio in small
        chunks. While sending, if barge-in arrives we stop early."""
        if not wav_bytes:
            return

        from io import BytesIO
        import wave

        try:
            with BytesIO(wav_bytes) as f:
                with wave.open(f, "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            log.error("Failed to decode TTS WAV: %s", e)
            return

        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0

        # send in 100 ms chunks so a barge-in can cut the reply short
        chunk_ms = 100
        samples_per_chunk = int(sample_rate * chunk_ms / 1000)
        connection["is_agent_speaking"] = True
        connection["agent_cancel_requested"] = False

        try:
            for i in range(0, len(audio), samples_per_chunk):
                if connection["agent_cancel_requested"]:
                    log.debug("Agent audio interrupted by barge-in")
                    break

                chunk = audio[i : i + samples_per_chunk]
                if len(chunk) < samples_per_chunk / 2:
                    break

                pcm = (chunk * 32767).astype("<i2").tobytes()
                b64 = base64.b64encode(pcm).decode()

                await self.send_media(connection, {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": b64},
                })
        except Exception as e:
            log.error("Error streaming agent audio: %s", e)
        finally:
            connection["is_agent_speaking"] = False
            connection["agent_cancel_requested"] = False

    async def _drain_agent_stream(self, connection: dict):
        """No-op placeholder: agent audio is now streamed directly in _play_audio.
        Kept so the main loop's call site stays meaningful if we later add a queue."""
        pass

    async def _cancel_agent_audio(self, connection: dict):
        """Stop any in-progress agent audio and clear the speaking flag."""
        connection["is_agent_speaking"] = False
        connection["agent_cancel_requested"] = True

    # ---------------------------------------------------------------- helpers

    async def send_media(self, connection: dict, data: dict):
        """Send a media frame to Twilio."""
        try:
            await connection["websocket"].send_json(data)
        except Exception as e:
            log.error("Failed to send media: %s", e)


def _recent_energy_is_speech(connection: dict) -> bool:
    """Return True if the recent frames look like speech rather than noise/silence."""
    samples = connection.get("speech_energy", [])
    if not samples:
        return False
    recent = samples[-6:]  # last ~120 ms
    if not recent:
        return False
    avg = sum(e for _, e in recent) / len(recent)
    # need a clear margin above the noise floor
    return avg >= _VAD_SPeech_RMS and avg > _VAD_MIN_RMS * 1.5



# ============================================================================
# Entry Point
# ============================================================================

async def main():
    """Main entry point for the LiveKit worker."""
    log.info("Starting LiveKit voice agent worker...")

    # Check configuration
    if not LIVEKIT_URL:
        log.error("LIVEKIT_URL not configured")
        return

    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        log.error("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be configured")
        return

    # Create and run the agent
    agent = VoiceAgent()

    options = WorkerOptions(
        agent=agent,
        websocket_url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET,
        # max_retries=3,
        # job_timeout=300,
    )

    log.info("Starting worker with LiveKit at: %s", LIVEKIT_URL)

    await cli.run_app(options)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    asyncio.run(main())
