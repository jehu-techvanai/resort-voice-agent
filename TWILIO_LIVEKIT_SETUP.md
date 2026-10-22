# Twilio + LiveKit Voice Integration Setup Guide

This guide explains how to configure your voice agent to handle inbound phone calls via Twilio and/or LiveKit.

## Quick Start

### Option 1: Twilio (PSTN Phone Calls)

1. **Get Twilio credentials:**
   - Sign up at [Twilio Console](https://console.twilio.com)
   - Buy a phone number
   - Copy your Account SID and Auth Token

2. **Configure `.env`:**
   ```bash
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=your_auth_token_here
   TWILIO_PHONE_NUMBER=+1234567890
   BASE_URL=https://your-domain.com  # Use ngrok for local dev
   ```

3. **Set up Twilio webhook:**
   - In Twilio Console → Phone Numbers → Manage → Active Numbers
   - Click your phone number
   - Configure "Voice & Fax" → "A call comes in"
   - Set URL to: `https://your-domain.com/twilio/inbound`
   - Method: HTTP POST

4. **Restart backend:**
   ```bash
   ./run.sh
   ```

5. **Test:** Call your Twilio number. The agent will answer and converse.

### Option 2: LiveKit (VoIP / SIP Calls)

1. **Get LiveKit credentials:**
   - Create a project at [LiveKit Cloud](https://cloud.livekit.io)
   - Go to Settings → API Keys
   - Create an API key with Voice Agent permissions

2. **Configure `.env`:**
   ```bash
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=your_api_key_here
   LIVEKIT_API_SECRET=your_api_secret_here
   ```

3. **Run the LiveKit worker:**
   ```bash
   python backend/livekit_worker.py
   ```

4. **For PSTN calls via LiveKit SIP:**
   - Set up a SIP trunk in LiveKit Cloud
   - Add to `.env`:
     ```bash
     LIVEKIT_SIP_URI=sip:your-trunk.sip.livekit.cloud
     SIP_OUTBOUND_TRUNK_ID=ST_xxxxx
     ```

### Local Development with ngrok

For testing locally without a public server:

1. **Install ngrok:** https://ngrok.com/download

2. **Start ngrok:**
   ```bash
   ngrok http 8000
   ```

3. **Use the ngrok URL** (e.g., `https://abc123.ngrok-free.app`) in:
   - Twilio webhook configuration
   - `.env` BASE_URL setting

## How It Works

### Twilio Integration Flow

```
Caller → Twilio Phone Number → Twilio → Your Backend (/twilio/inbound)
                                                      ↓
                                              Generate TwiML
                                                      ↓
                                    ┌─────────────────┴─────────────────┐
                                    ↓                                   ↓
                           Media Streams (recommended)          Gather Fallback
                                    ↓                                   ↓
                           WebSocket /ws/twilio                 /twilio/process-speech
                                    ↓                                   ↓
                           STT → LLM → TTS                     STT → LLM → TTS
                                    ↓                                   ↓
                           Audio back to Twilio                 Audio back to Twilio
                                    ↓                                   ↓
                                 Caller hears response            Caller hears response
```

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/twilio/inbound` | POST | Twilio webhook for incoming calls |
| `/twilio/gather` | POST | Handle speech recognition results |
| `/twilio/process-speech` | POST | Process speech and generate response |
| `/twilio/continue-call` | GET | Continue an ongoing conversation |

### LiveKit Worker Features

- Real-time audio processing via LiveKit Agents SDK
- Gemini Live native audio support (if configured)
- SIP participant handling for PSTN calls
- WebSocket media stream handling for Twilio integration

## Configuration Reference

### Required Settings

```bash
# Twilio
TWILIO_ACCOUNT_SID=        # Your Twilio Account SID
TWILIO_AUTH_TOKEN=         # Your Twilio Auth Token
TWILIO_PHONE_NUMBER=       # Your Twilio phone number

# LiveKit (if using)
LIVEKIT_URL=               # LiveKit server URL (wss://...)
LIVEKIT_API_KEY=           # LiveKit API key
LIVEKIT_API_SECRET=        # LiveKit API secret
```

### Optional Settings

```bash
# Application
BASE_URL=http://localhost:8000     # Public URL for webhooks
TWILIO_MEDIA_STREAMS=true          # Use real-time streaming (default: true)

# LiveKit SIP (for PSTN)
LIVEKIT_SIP_URI=                   # SIP URI for trunk
SIP_OUTBOUND_TRUNK_ID=             # SIP trunk ID
```

## Testing

Run the test suite:

```bash
cd backend
python -m pytest test_twilio_integration.py -v
```

Expected output: 7 passed, 1 skipped (LiveKit test skipped if not installed)

## Troubleshooting

### Twilio webhook returns 403/invalid signature

Ensure your `BASE_URL` matches exactly what Twilio is calling (including https://).

### No audio / one-way audio

- Check that `TWILIO_MEDIA_STREAMS=true` is set
- For local testing, ngrok must be running with HTTPS
- Check firewall rules allow WebSocket connections

### LiveKit worker won't connect

- Verify `LIVEKIT_URL` starts with `wss://`
- Check API key and secret are correct
- Ensure network allows outbound WebSocket connections

### Call disconnects immediately

- Check Twilio webhook response is valid XML
- Verify the `/twilio/inbound` endpoint returns 200 OK
- Check backend logs for errors

## Architecture Notes

### Twilio Media Streams

Twilio Media Streams provides real-time bidirectional audio:
- Lower latency than Gather→Process→Play
- Full duplex (can interrupt/barge-in)
- Requires WebSocket support at your endpoint

### Fallback Mode

When Media Streams isn't available or fails:
- Twilio Gather collects speech
- Sends to `/twilio/process-speech`
- Returns TwiML with response
- Higher latency but more compatible

### Voice Selection

The agent uses the TTS voice configured in your settings:
- `TTS_VOICE=af_heart` (default, American English)
- `TTS_VOICE=bf_emma` (British English)
- See Kokoro documentation for all available voices

## Production Deployment

### Security

1. **Validate Twilio signatures** (in production):
   ```python
   from twilio.request_validator import RequestValidator
   validator = RequestValidator(TWILIO_AUTH_TOKEN)
   is_valid = validator.validate(url, params, signature)
   ```

2. **Use HTTPS** for all webhook endpoints

3. **Rate limit** the `/twilio/inbound` endpoint

4. **Monitor** call logs and performance metrics

### Scaling

- The backend can handle multiple concurrent calls (limited by `WS_MAX_SESSIONS`)
- For high volume, run multiple backend instances behind a load balancer
- Use Redis for session state sharing across instances

### Monitoring

- Check `/api/admin/metrics` for latency percentiles
- Review conversation logs in `/api/admin/conversations`
- Monitor unanswered questions in `/api/admin/unanswered`
