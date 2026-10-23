"""
twilio_router.py — Twilio inbound voice webhook for the voice agent.

Twilio calls this endpoint when someone calls your Twilio phone number.
We return TwiML that:
  1. Gathers the caller's speech (or DTMF if speech fails)
  2. Sends audio to our backend via WebSocket for STT→LLM→TTS processing
  3. Plays back the agent's response

Two modes:
  - stream_mode: Real-time streaming via WebSocket (lower latency, full duplex)
  - gather_mode: Simple gather→process→playback (fallback, higher latency)

For production, use stream_mode with Twilio Media Streams.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import (
    Gather,
    VoiceResponse as TwiMLResponse,
    Say,
    Connect,
    Stream,
)

log = logging.getLogger("twilio-router")

router = APIRouter()

# Call session state (in production, use Redis)
_call_sessions: dict[str, dict] = {}


def generate_twiml(
    stream_url: str,
    stream_token: str,
    caller_phone: str,
    called_phone: str,
    use_media_stream: bool = True,
) -> str:
    """
    Generate TwiML to handle an inbound call.

    Args:
        stream_url: WebSocket URL for media streaming
        stream_token: Authentication token for the stream
        caller_phone: Caller's phone number
        called_phone: Called phone number
        use_media_stream: If True, use Twilio Media Streams for real-time audio
    """
    response = TwiMLResponse()

    if use_media_stream:
        # Media Streams mode: real-time bidirectional audio
        # NOTE: <Stream url> must NOT contain query string parameters (they are
        # unsupported and make Twilio refuse to connect). The CallSid arrives
        # in the WebSocket "start" event (stream.callSid) instead.
        connect = Connect()
        connect.append(Stream(url=stream_url))
        response.append(connect)

        # Fallback: if stream disconnects, gather speech as backup
        gather = Gather(
            input="speech",
            speech_timeout="5",
            action="/twilio/gather",
            method="POST",
        )
        gather.say(
            "Sorry, I'm having trouble with the connection. Please try again later.",
            voice="alice",
        )
        response.append(gather)
    else:
        # Simple gather mode (fallback)
        gather = Gather(
            input="speech",
            speech_timeout="5",
            action="/twilio/process-speech",
            method="POST",
        )
        gather.say(
            "Hello, I'm listening. Please speak after the tone.",
            voice="alice",
        )
        response.append(gather)

    return str(response)


def generate_gather_twiml(
    speech_result: str,
    call_sid: str,
    continue_conversation: bool = True,
) -> str:
    """
    Generate TwiML after gathering speech from the caller.
    """
    response = TwiMLResponse()

    if continue_conversation:
        # Send the speech to our backend and stream response back
        # In production, this would connect to WebSocket
        response.redirect(f"/twilio/continue-call?call_sid={call_sid}")
    else:
        response.say(
            "Thank you for calling. Have a great day!",
            voice="alice",
        )
        response.hangup()

    return str(response)


@router.post("/twilio/inbound")
async def handle_inbound_call(
    request: Request,
    twilio_signature: Optional[str] = Header(None, alias="X-Twilio-Signature"),
):
    """
    Twilio webhook for inbound calls.

    Expects Twilio's standard voice webhook parameters:
    - CallSid: Unique identifier for this call
    - From: Caller's phone number
    - To: Your Twilio phone number
    - CallStatus: queued|ringing|in-progress|completed|busy|no-answer|failed|canceled
    - Direction: inbound or outbound
    """
    form_data = await request.form()

    call_sid = form_data.get("CallSid", "")
    caller_phone = form_data.get("From", "")
    called_phone = form_data.get("To", "")
    call_status = form_data.get("CallStatus", "")
    direction = form_data.get("Direction", "")

    log.info(
        "Inbound call: call_sid=%s, from=%s, to=%s, status=%s, direction=%s",
        call_sid,
        caller_phone,
        called_phone,
        call_status,
        direction,
    )

    # Validate Twilio signature in production
    # if not validate_twilio_signature(...):
    #     raise HTTPException(403, "Invalid signature")

    # Check if Twilio credentials are configured
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    if not twilio_sid:
        log.warning("Twilio not configured, returning simple greeting")
        response = TwiMLResponse()
        response.say(
            "Thank you for calling. Our team is unavailable at the moment.",
            voice="alice",
        )
        response.hangup()
        return Response(content=str(response), media_type="text/xml")

    # Create or retrieve call session
    if call_sid not in _call_sessions:
        _call_sessions[call_sid] = {
            "call_sid": call_sid,
            "caller_phone": caller_phone,
            "called_phone": called_phone,
            "started_at": datetime.utcnow().isoformat(),
            "status": "in_progress",
            "transcript": [],
        }

    # Generate WebSocket URL for media streaming
    # NOTE: no query string allowed in <Stream url> (Twilio rejects it and the
    # caller hears the "cannot reach your server" error). The call_sid is
    # delivered to the WebSocket handler in the Twilio "start" event instead.
    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    stream_url = f"wss://{base_url.replace('http://', '').replace('https://', '')}/ws/twilio"
    stream_token = str(uuid.uuid4())

    use_media_stream = os.getenv("TWILIO_MEDIA_STREAMS", "true").lower() == "true"

    twiml = generate_twiml(
        stream_url=stream_url,
        stream_token=stream_token,
        caller_phone=caller_phone,
        called_phone=called_phone,
        use_media_stream=use_media_stream,
    )

    log.info("TwiML response generated for call %s", call_sid)
    return Response(content=twiml, media_type="text/xml")


@router.post("/twilio/gather")
async def handle_gather_result(request: Request):
    """
    Handle results from Twilio Gather verb.
    Called when speech is recognized or DTMF is entered.
    """
    form_data = await request.form()

    call_sid = form_data.get("CallSid", "")
    speech_result = form_data.get("SpeechResult", "")
    digits = form_data.get("Digits", "")
    call_status = form_data.get("CallStatus", "")

    log.info(
        "Gather result: call_sid=%s, speech=%s, digits=%s, status=%s",
        call_sid,
        speech_result,
        digits,
        call_status,
    )

    # Update session
    if call_sid in _call_sessions:
        session = _call_sessions[call_sid]
        if speech_result:
            session["transcript"].append({
                "role": "user",
                "text": speech_result,
                "timestamp": datetime.utcnow().isoformat(),
            })

    # Generate response TwiML
    twiml = generate_gather_twiml(
        speech_result=speech_result or digits,
        call_sid=call_sid,
        continue_conversation=bool(speech_result or digits),
    )

    return Response(content=twiml, media_type="text/xml")


@router.post("/twilio/process-speech")
async def process_speech(request: Request):
    """
    Process gathered speech and generate a response.
    This is a simpler fallback mode without Media Streams.
    """
    form_data = await request.form()

    call_sid = form_data.get("CallSid", "")
    speech_result = form_data.get("SpeechResult", "")
    caller_phone = form_data.get("From", "")

    log.info(
        "Processing speech: call_sid=%s, speech=%s, from=%s",
        call_sid,
        speech_result,
        caller_phone,
    )

    # In production, this would:
    # 1. Send speech to STT if needed (already done by Twilio)
    # 2. Process through LLM
    # 3. Generate TTS response
    # 4. Return TwiML with the response

    response = TwiMLResponse()

    if speech_result:
        # For now, echo back - in production, use LLM
        response.say(
            f"I heard you say: {speech_result}. Let me check on that for you.",
            voice="alice",
        )
        # In production, this would be the actual LLM response
        response.say(
            "I'm looking into that now. Please hold.",
            voice="alice",
        )
    else:
        response.say(
            "I didn't catch that. Could you please repeat?",
            voice="alice",
        )

    response.hangup()
    return Response(content=str(response), media_type="text/xml")


@router.get("/twilio/continue-call")
async def continue_call(
    call_sid: str = Query(...),
    response_url: Optional[str] = Query(None),
):
    """
    Continue an ongoing conversation.
    In production, this would stream the response back via WebSocket
    or return TwiML with the next say/play instruction.
    """
    log.info("Continuing call: call_sid=%s", call_sid)

    response = TwiMLResponse()

    # Get conversation history from session
    session = _call_sessions.get(call_sid, {})
    transcript = session.get("transcript", [])

    if transcript:
        last_user_msg = transcript[-1].get("text", "") if transcript else ""

        # In production, this would be LLM-generated response
        response.say(
            f"Thank you for your message about '{last_user_msg}'. "
            "Let me connect you with the right team member.",
            voice="alice",
        )
    else:
        response.say(
            "Hello! How can I help you today?",
            voice="alice",
        )

    response.hangup()
    return Response(content=str(response), media_type="text/xml")


@router.post("/twilio/status")
async def handle_call_status(request: Request):
    """
    Twilio call status callback.
    Updates the call status in our system.
    """
    form_data = await request.form()

    call_sid = form_data.get("CallSid", "")
    call_status = form_data.get("CallStatus", "")
    final_report = form_data.get("FinalReport", "")

    log.info(
        "Call status update: call_sid=%s, status=%s, report=%s",
        call_sid,
        call_status,
        final_report,
    )

    # Update session status
    if call_sid in _call_sessions:
        _call_sessions[call_sid]["status"] = call_status
        _call_sessions[call_sid]["ended_at"] = datetime.utcnow().isoformat()

        # Log the completed call
        if call_status in ("completed", "busy", "no-answer", "failed"):
            log.info(
                "Call completed: call_sid=%s, status=%s, duration=%s",
                call_sid,
                call_status,
                final_report,
            )
            # In production, save to database here

    return Response(content="OK", media_type="text/plain")


def validate_twilio_signature(
    url: str,
    params: dict,
    signature: str,
    auth_token: str,
) -> bool:
    """
    Validate Twilio request signature to prevent spoofing.
    """
    from twilio.request_validator import RequestValidator

    validator = RequestValidator(auth_token)
    return validator.validate(url, params, signature)
