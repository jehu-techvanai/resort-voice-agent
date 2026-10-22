# Voice Agent (v2)

A hands-free voice receptionist built from small open models: Whisper (speech to text), bge-small (retrieval), a 3B LLM through Ollama, and Kokoro (voice). It answers from a knowledge base you upload, and logs questions it cannot answer for a callback.

## What's new in v2

### Conversation intelligence — the LLM decides, not keyword lists
- **Natural conversation memory.** The LLM was never seeing its own replies (a bug in v0.3 dropped them from session history), so conversations felt robotic. Fixed: every reply is remembered, wake checks, routing and retrieval all read the last turns.
- **Wake decisions by the LLM.** Follow-ups like "OK, price ranges" after a price discussion are recognized without the calling name; the same call also cleans obvious speech-to-text slips ("praise ranges" → "price ranges") before retrieval. The old name/greeting matcher only runs if the LLM is unreachable, so a dead Ollama never kills the line.
- **Greetings, goodbyes and thank-yous** are classified by the LLM from the conversation, so brief replies get a friendly one-liner instead of restarting the topic.
- **Scope guardrail.** The long hardcoded keyword list is replaced by an LLM check built from the business name, business type and the actual knowledge-base topics. Out-of-scope questions get a short warm redirect; it keeps working automatically as you rename the business or edit the KB.
- **Follow-up questions.** Short follow-ups ("and the price?", "what about that one?") that retrieve nothing are rewritten by the LLM into a standalone question using the session history, then searched again. Direct questions pay no extra latency.

### Production hardening
- **No silent crashes.** WebSocket handlers log full tracebacks instead of failing invisibly; the LLM keep-alive task is cancelled on shutdown.
- **Session limits:** `WS_MAX_SESSIONS` (8), `WS_IDLE_TIMEOUT_SEC` (600, with a browser heartbeat), and a per-frame byte cap, so dead or stuck connections can no longer hold the GPU models forever.
- **Endpoint safety:** log-file downloads validate the filename (no path traversal), knowledge uploads are capped at `MAX_UPLOAD_BYTES` (10 MB per file), and the deprecated `Query(regex=...)` API was replaced.
- **SQLite in production shape:** WAL journal mode and a 5s busy timeout, so dashboard reads no longer block a live voice turn.
- **Latency percentiles.** `/api/admin/metrics` and the Observability tab now show p50/p95/max latency per turn, not just averages — tail spikes are what make callers hang up.

### 2050 interface
- The UI was redesigned as a dark sci-fi HUD: glass panels on a deep-space grid, a reactor-style voice orb with a distinct glow per state (asleep / awake / hearing / thinking / speaking), streaming chat bubbles with neon retrieval chips, a fixed glass top bar with active-state navigation, and mono typography for all timings. Settings, Conversations and Log files share the new theme.

## What's new in v0.3

- **Speech to text is switchable** in Settings, Persona:
  - Whisper small, medium or large-v3-turbo (local).
  - Sarvam saaras:v4 (cloud, built for Indian speech; set `SARVAM_API_KEY` in `.env` and restart).
- **Better accent handling with Whisper.** Anything that could be English is decoded as English. The model is primed with the calling name, the business name and the knowledge-base headings (room types, place names). Sarvam gets the same names as key terms.
- **The English-only check is stricter** before it rejects speech (0.8 confidence instead of 0.6), so Indian-accented English is less likely to be taken as Hindi.
- **Less lag:**
  - The LLM is pinned in Ollama's memory (it used to unload after 5 idle minutes).
  - Whisper is warmed up at startup.
  - The voice-detector files load while the page opens.
  - The first part of each answer is spoken as soon as its first clause is ready.
- **Listening starts by itself** when the page opens, once the microphone has been allowed. Browsers still need one tap anywhere before they play sound; the page says so.
- **Barge-in:**
  - If you keep talking for half a second over the agent, it stops at once and the reply is cancelled on the server.
  - While the agent talks, the voice detector needs clearer speech, so the agent triggers on its own voice less often.
  - "Echo cancellation boost" plays the agent's voice through the browser's WebRTC path, which Chrome's echo canceller works on. This should help on laptop speakers; how much depends on the machine.
- **Auto-reconnect** if the connection to the server drops.

## v0.2 features

- **Always listening.** A voice detector runs in the browser (Silero VAD) and only sends speech to the server, never continuous audio.
- **Calling name (web page only).** Say "Maya, ..." to start. You then get a follow-up window (20 seconds by default) where you can speak without the name. You can also say just "Maya" and it greets you.
- **Noise filtering.** The agent ignores very short sounds, quiet sounds, Whisper's typical noise phrases ("thank you", "you"), its own voice coming back through the speakers, and speech without the calling name. Ignored speech is shown in grey, so you can see what was filtered and why.
- **Barge-in.** If you speak while the agent is talking, its volume drops at once. If the server accepts what you said, the reply stops and your new question is answered, and the history keeps only what you actually heard. If it was noise, the volume comes back.
- **Unknown questions.** If retrieval finds nothing relevant, or the model says it does not know, the agent says the fallback line and asks for your name and phone number. The question goes to Settings, Unanswered. If you decline, or move on to another question, the question is still logged.
- **Settings page** (`/#/settings`, password protected):
  - Persona: calling name, business, greeting, fallback line, extra instructions, voice, speed.
  - Knowledge base: upload, Train, version history with rollback, and a chunk browser.
  - Test: shows the answer, retrieved chunks and scores, to help tune the threshold.
  - Unanswered: the log, with CSV export.
- **Mobile.** The layout works on phones. The microphone needs https, so use `./share.sh`.

## Start

```
./run.sh                 # http://localhost:8000
```

The settings password is printed in the terminal at startup. To choose your own, set `ADMIN_PASSWORD` in `.env`.

Open the page, press **Start listening**, and say "Maya, what are the rates for pool villas?".

## Upgrading from v0.1

1. Unzip the new version over the old folder. This keeps `.venv`, `.env` and `data/`.
2. Run `./upgrade.sh`.
3. Run `./run.sh`.

## Using it on a phone or another computer

```
./share.sh               # prints an https://...trycloudflare.com link
```

Anyone with the link can use the agent while the tunnel runs, so set `ADMIN_PASSWORD` first. Press Ctrl+C to stop the tunnel.

## Knowledge base

- **Supported files:** DOCX, PDF (text, not scanned), MD and TXT.
- **How files are split into chunks:**
  - FAQ files, where a question is followed by its answer, become one chunk per question.
  - Other documents are split at their headings, and long sections are split into overlapping pieces.
  - In DOCX files, table rows are kept as "Header: value" text.
- **Train** builds a new version from all files listed under the upload box. If you want a clean replacement, remove any old files from that list first.
- **Versions** are kept in `data/kb/`. Click "Use this" on any version to roll back.
- **After uploading a different business's documents,** also change Business name, Type of business and Calling name in Persona.

**Tuning the retrieval threshold.** The default is 0.40. It is a starting guess, not measured on real data.
- In the Test tab, ask 10 questions the documents answer and 10 they do not.
- If good questions come back as unknown, lower the threshold.
- If unrelated questions get answers, raise it.
- The model also has its own "UNKNOWN" check as a second gate. Small models are not fully reliable at this, so test with your own questions.

## Machines

**Mac (Apple Silicon):**
```
./setup-mac.sh
./run.sh
```
Ollama uses the Apple GPU. Whisper, embeddings and Kokoro run on the CPU.

**Ubuntu, CPU only:**
```
./setup-cpu.sh
./run.sh
```
The CPU needs AVX2. Check with `grep -o avx2 /proc/cpuinfo`. Virtual machines with the generic QEMU CPU type do not have it.

**Ubuntu with an NVIDIA GPU:**
```
./setup.sh
./run.sh
```

**Cloud:** 1 GPU with 16 GB or more, 4 or more vCPUs and 16 to 32 GB RAM, for example:
- Azure NC4as T4 v3
- AWS g6.xlarge
- Paperspace A4000

## Settings in .env

See `.env.example`. The main ones are:
- `LLM_MODEL`, for example `llama3.2:3b`, `qwen2.5:3b-instruct` or `gemma3:4b`
- `STT_MODEL`, which must stay multilingual (not `*.en`) so the English check works
- `ADMIN_PASSWORD`
- `PORT`
- `MIN_UTTERANCE_SEC`, `MIN_RMS` and `NO_SPEECH_PROB`, the noise filters
- v2 connection limits: `WS_MAX_SESSIONS`, `WS_IDLE_TIMEOUT_SEC`, `WS_MAX_MSG_BYTES`, `MAX_UPLOAD_BYTES`
- v2 stores: `REDIS_URL`, `SQLITE_DB`, hybrid RAG (`V2_RAG`, `V2_RERANK`, `V2_QUERY_EXPANSION`) — see `UPDATE.md` for the full list

## Live session protocol (WebSocket `/ws?mode=web`)

**Client to server:**
- A binary frame holding one utterance: 16 kHz mono PCM16.
- `{"type":"text","text":...}`
- `{"type":"progress","turn":n,"index":i}`, sent when sentence i starts playing.
- `{"type":"idle","turn":n}`, sent when the reply has finished playing.
- `{"type":"config","wake":bool}`
- `{"type":"reset"}`

**Server to client:**
- `ready`, `state`
- `heard`, `ignored` (with a reason), `accepted`
- `sentence`, `audio` (a WAV file as base64)
- `done` (with timings, retrieved sources, and an unknown flag)
- `error`

The API for other programs (phase 3) will use the same protocol with `mode=api` and an API key, and no calling name.

## Phone integration (Twilio + LiveKit)

### Twilio inbound calls

1. Buy a phone number at [Twilio Console](https://console.twilio.com)
2. Set the voice webhook URL to: `https://your-domain.com/twilio/inbound`
3. Add credentials to `.env`:
   ```
   TWILIO_ACCOUNT_SID=ACxxxxx
   TWILIO_AUTH_TOKEN=xxxxx
   TWILIO_PHONE_NUMBER=+1234567890
   ```
4. Restart the backend: `./run.sh`

The agent will answer calls and converse using the same STT→LLM→TTS pipeline.

### LiveKit voice calls

1. Create a project at [LiveKit Cloud](https://cloud.livekit.io)
2. Get API credentials (API Key + Secret)
3. Add to `.env`:
   ```
   LIVEKIT_URL=wss://your-project.livekit.cloud
   LIVEKIT_API_KEY=xxxxx
   LIVEKIT_API_SECRET=xxxxx
   ```
4. Run the LiveKit worker:
   ```
   python backend/livekit_worker.py
   ```

For PSTN calls via LiveKit SIP, also configure:
   ```
   LIVEKIT_SIP_URI=sip:your-trunk.sip.livekit.cloud
   SIP_OUTBOUND_TRUNK_ID=ST_xxxxx
   ```

### Local testing with ngrok

For local development, expose your backend with ngrok:
```bash
ngrok http 8000
```
Use the ngrok URL in your Twilio webhook configuration.

## Known limits

- Other people talking nearby are still speech. Outside the follow-up window, the calling name is what filters them out. Inside it, nearby speech can be taken as a question.
- Without headphones the agent can hear itself. Browser echo cancellation and the echo check reduce this but don't remove it.
- Listening stops when a phone's screen locks.
- Speech to text runs after each utterance ends, not while you speak.
- One process with one copy of each model, so simultaneous sessions take turns.
- Twilio Media Streams require HTTPS (use ngrok for local testing).
