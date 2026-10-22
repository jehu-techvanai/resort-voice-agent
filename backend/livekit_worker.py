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
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
from livekit.plugins import noise_cancellation
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

class TwilioMediaStreamHandler:
    """
    Handle Twilio Media Streams WebSocket connections.

    This receives real-time audio from Twilio and sends back audio responses.
    """

    def __init__(self):
        self.connections: dict[str, dict] = {}
        self.stt = STT()
        self.tts = TTS()
        self.llm = LLM()

    async def handle_websocket(self, websocket, call_sid: str):
        """Handle a Twilio Media Stream WebSocket connection."""
        log.info("Twilio Media Stream connected: call_sid=%s", call_sid)

        # Store connection
        connection = {
            "websocket": websocket,
            "call_sid": call_sid,
            "buffer": bytearray(),
            "is_speaking": False,
            "current_track": None,
        }
        self.connections[call_sid] = connection

        try:
            async for message in websocket.iter_text():
                data = json.loads(message)

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

        except Exception as e:
            log.error("Media Stream error: %s", e)
        finally:
            if call_sid in self.connections:
                del self.connections[call_sid]
            log.info("Media Stream disconnected: call_sid=%s", call_sid)

    async def handle_start(self, connection: dict, data: dict):
        """Handle stream start event."""
        stream = data.get("stream", {})
        connection["call_sid"] = stream.get("callSid", connection["call_sid"])

        log.info(
            "Stream started: call_sid=%s, stream_id=%s",
            connection["call_sid"],
            stream.get("streamSid"),
        )

        # Send initial heartbeat
        await self.send_media(connection, {
            "event": "media",
            "streamSid": data.get("stream", {}).get("streamSid", ""),
            "media": {"payload": ""},
        })

    async def handle_media(self, connection: dict, data: dict):
        """Handle incoming audio media."""
        media = data.get("media", {})
        payload = media.get("payload", "")

        if not payload:
            return

        # Decode base64 audio
        import base64
        audio_bytes = base64.b64decode(payload)

        # Add to buffer
        connection["buffer"].extend(audio_bytes)

        # Check if we should process (simple threshold for demo)
        if len(connection["buffer"]) > 16000:  # ~1 second at 16kHz
            stream_sid = data.get("stream", {}).get("streamSid", "")
            await self.process_audio(connection, stream_sid)
            connection["buffer"] = bytearray()

    async def process_audio(self, connection: dict, stream_sid: str = ""):
        """Process buffered audio through STT→LLM→TTS pipeline."""
        buffer = bytes(connection["buffer"])

        if len(buffer) < 1600:
            return

        log.debug("Processing audio: %d bytes", len(buffer))

        try:
            # Convert to float32
            audio = pcm16_to_float(buffer)

            # STT
            from store import load_settings
            settings = load_settings()
            stt_engine = settings.get("stt_engine", "whisper-small")
            transcript = self.stt.transcribe_pcm(audio, stt_engine)

            if not transcript.text.strip():
                return

            log.info("Transcription: %s", transcript.text)

            # LLM
            messages = [
                {"role": "system", "content": system_prompt(settings, [])},
                {"role": "user", "content": transcript.text},
            ]
            response_text = await self.llm.complete(messages, max_tokens=150)

            log.info("LLM response: %s", response_text)

            # TTS
            wav_bytes = self.tts.synthesize(response_text, VOICE_ID, VOICE_SPEED)

            # Send audio back to Twilio
            import base64
            audio_b64 = base64.b64encode(wav_bytes).decode()

            await self.send_media(connection, {
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": audio_b64,
                },
            })

        except Exception as e:
            log.error("Audio processing error: %s", e)

    async def handle_stop(self, connection: dict, data: dict):
        """Handle stream stop event."""
        log.info("Stream stopped: call_sid=%s", connection["call_sid"])

        # Clean up
        connection["buffer"] = bytearray()

    async def handle_dtmf(self, connection: dict, data: dict):
        """Handle DTMF tones."""
        dtmf = data.get("dtmf", {})
        digit = dtmf.get("digit", "")

        log.info("DTMF received: %s", digit)

        # Handle DTMF input
        if digit == "#":
            # End of input, process
            await self.process_audio(connection)
        elif digit == "*":
            # Cancel/repeat
            pass

    async def send_media(self, connection: dict, data: dict):
        """Send media response to Twilio."""
        try:
            await connection["websocket"].send_json(data)
        except Exception as e:
            log.error("Failed to send media: %s", e)


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
