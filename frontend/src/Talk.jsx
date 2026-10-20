import { useCallback, useEffect, useRef, useState } from 'react'
import { ChunkPlayer, floatTo16BitPCM } from './audio.js'

const STAGES = [
  ['stt_ms', 'Speech to text'],
  ['retrieval_ms', 'Retrieval'],
  ['llm_first_token_ms', 'LLM first token'],
  ['first_audio_ms', 'First audio ready'],
  ['total_ms', 'Full reply done'],
]

function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${location.host}/ws?mode=web`
}

function loadPref(key, fallback) {
  try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v) } catch { return fallback }
}
function savePref(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* storage unavailable */ }
}

export default function Talk({ health }) {
  const [running, setRunning] = useState(false)
  const [starting, setStarting] = useState(false)
  const [phase, setPhase] = useState('off') // off | asleep | awake | hearing | thinking | speaking
  const [awakeLeft, setAwakeLeft] = useState(0)
  const [wakeRequired, setWakeRequired] = useState(() => loadPref('wakeRequired', true))
  const [allowBargeIn, setAllowBargeIn] = useState(() => loadPref('bargeIn', true))
  const [showIgnored, setShowIgnored] = useState(() => loadPref('showIgnored', true))
  const [autoStart, setAutoStart] = useState(() => loadPref('autoStart', true))
  const [fastBarge, setFastBarge] = useState(() => loadPref('fastBarge', true))
  const [echoBoost, setEchoBoost] = useState(() => loadPref('echoBoost', true))
  const [needTap, setNeedTap] = useState(false)
  const [agentName, setAgentName] = useState(health?.agent || 'Maya')
  const [log, setLog] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState('')
  const [typed, setTyped] = useState('')
  const [pendingContact, setPendingContact] = useState(false)

  const wsRef = useRef(null)
  const vadRef = useRef(null)
  const playerRef = useRef(null)
  const turnRef = useRef(0)
  const awakeUntilRef = useRef(0)
  const wakeRef = useRef(wakeRequired)
  const bargeRef = useRef(allowBargeIn)
  const speakingRef = useRef(false)
  const thinkingRef = useRef(false)
  const hearingRef = useRef(false)
  const endOfSpeechRef = useRef(0)
  const logRef = useRef(null)
  const fastBargeRef = useRef(fastBarge)
  const bargeTimerRef = useRef(0)
  const stoppingRef = useRef(false)
  const retryRef = useRef(0)

  useEffect(() => { wakeRef.current = wakeRequired; savePref('wakeRequired', wakeRequired) }, [wakeRequired])
  useEffect(() => { bargeRef.current = allowBargeIn; savePref('bargeIn', allowBargeIn) }, [allowBargeIn])
  useEffect(() => { savePref('showIgnored', showIgnored) }, [showIgnored])
  useEffect(() => { savePref('autoStart', autoStart) }, [autoStart])
  useEffect(() => { fastBargeRef.current = fastBarge; savePref('fastBarge', fastBarge) }, [fastBarge])
  useEffect(() => { savePref('echoBoost', echoBoost) }, [echoBoost])

  // Fetch the voice-detector files while the page is idle, so Start is quick.
  useEffect(() => {
    import('@ricky0123/vad-web').catch(() => {})
    for (const f of ['vad/silero_vad_v5.onnx', 'vad/ort-wasm-simd-threaded.wasm', 'vad/vad.worklet.bundle.min.js']) {
      fetch(new URL(f, document.baseURI)).catch(() => {})
    }
  }, [])
  useEffect(() => { logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' }) }, [log])

  const refreshPhase = useCallback(() => {
    if (!wsRef.current) return setPhase('off')
    if (hearingRef.current) return setPhase('hearing')
    if (speakingRef.current) return setPhase('speaking')
    if (thinkingRef.current) return setPhase('thinking')
    const awake = !wakeRef.current || Date.now() < awakeUntilRef.current
    setPhase(awake ? 'awake' : 'asleep')
  }, [])

  // countdown ticker for the follow-up window
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => {
      setAwakeLeft(Math.max(0, Math.round((awakeUntilRef.current - Date.now()) / 1000)))
      refreshPhase()
    }, 500)
    return () => clearInterval(id)
  }, [running, refreshPhase])

  const addLog = (entry) => setLog((l) => [...l.slice(-80), entry])
  const updateTurn = (turn, fn) => setLog((l) => l.map((e) => (e.kind === 'agent' && e.turn === turn ? fn({ ...e }) : e)))

  const send = (obj) => {
    const ws = wsRef.current
    if (ws && ws.readyState === 1) ws.send(typeof obj === 'string' || obj instanceof ArrayBuffer ? obj : JSON.stringify(obj))
  }

  const handleEvent = useCallback((e) => {
    const player = playerRef.current
    switch (e.type) {
      case 'ready':
        setAgentName(e.agent_name)
        send({ type: 'config', wake: wakeRef.current })
        break
      case 'state':
        awakeUntilRef.current = e.awake_ms != null ? Date.now() + e.awake_ms : (e.awake ? Date.now() + 3600e3 : 0)
        setPendingContact(!!e.pending_contact)
        refreshPhase()
        break
      case 'heard':
        thinkingRef.current = true
        refreshPhase()
        break
      case 'ignored':
        thinkingRef.current = false
        player?.unduck()
        if (e.text) addLog({ kind: 'ignored', text: e.text, reason: e.reason })
        refreshPhase()
        break
      case 'accepted':
        thinkingRef.current = true
        turnRef.current = e.turn
        if (player && player.turn !== null && player.turn < e.turn) player.stop()
        addLog({ kind: 'user', text: e.text || '(calling name only)' })
        addLog({ kind: 'agent', turn: e.turn, text: '', sources: [] })
        refreshPhase()
        break
      case 'sentence':
        if (e.turn !== turnRef.current) break
        updateTurn(e.turn, (x) => ({ ...x, text: (x.text ? x.text + ' ' : '') + e.text }))
        break
      case 'audio':
        if (e.turn !== turnRef.current) break
        player?.push(e.wav_b64, e.turn, e.index)
        break
      case 'done':
        thinkingRef.current = false
        player?.markDone(e.turn)
        if (e.turn === turnRef.current) {
          setMetrics((m) => ({ ...e.timings, heard_ms: m?.pendingHeard ?? null }))
          updateTurn(e.turn, (x) => ({ ...x, sources: e.sources, unknown: e.unknown }))
        }
        refreshPhase()
        break
      case 'error':
        thinkingRef.current = false
        setError(e.message)
        refreshPhase()
        break
      default:
    }
  }, [refreshPhase])

  const stop = useCallback(async () => {
    stoppingRef.current = true
    clearTimeout(bargeTimerRef.current)
    try { await vadRef.current?.destroy() } catch { /* ignore */ }
    vadRef.current = null
    wsRef.current?.close()
    wsRef.current = null
    playerRef.current?.stop()
    speakingRef.current = thinkingRef.current = hearingRef.current = false
    setRunning(false)
    setPhase('off')
  }, [])

  const start = useCallback(async () => {
    setError('')
    setStarting(true)
    if (!window.isSecureContext) {
      setError('The microphone needs https or localhost. Open this page over https (see README) or on this machine.')
      setStarting(false)
      return
    }
    const player = playerRef.current || new ChunkPlayer()
    playerRef.current = player
    player.unlock(echoBoost) // inside the click when there is one (iOS / Safari need that for sound)
    player.onStart = (turn, index) => {
      if (index === 0 && endOfSpeechRef.current) {
        const ms = Math.round(performance.now() - endOfSpeechRef.current)
        endOfSpeechRef.current = 0
        setMetrics((m) => ({ ...(m || {}), heard_ms: ms, pendingHeard: ms }))
      }
      send({ type: 'progress', turn, index })
    }
    player.onIdle = (turn) => send({ type: 'idle', turn })
    player.onActivity = (playing) => {
      speakingRef.current = playing
      // While the agent talks, demand clearer speech before treating sound as the caller (less self-triggering)
      try {
        vadRef.current?.setOptions(playing
          ? { positiveSpeechThreshold: 0.85, negativeSpeechThreshold: 0.65 }
          : { positiveSpeechThreshold: 0.6, negativeSpeechThreshold: 0.4 })
      } catch { /* older detector */ }
      refreshPhase()
    }

    stoppingRef.current = false
    const connect = async () => {
      const ws = new WebSocket(wsUrl())
      ws.binaryType = 'arraybuffer'
      ws.onmessage = (m) => { try { handleEvent(JSON.parse(m.data)) } catch { /* ignore bad frame */ } }
      ws.onclose = () => {
        if (wsRef.current !== ws || stoppingRef.current) return
        // keep the microphone running and reconnect quietly
        setError('Connection lost, reconnecting…')
        setTimeout(() => { if (!stoppingRef.current) connect() }, Math.min(5000, 1000 * ++retryRef.current))
      }
      await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error('Cannot reach the agent')) })
        .catch((err) => { setError(err.message) })
      if (ws.readyState === 1) { wsRef.current = ws; retryRef.current = 0; setError('') }
      return ws.readyState === 1
    }
    if (!(await connect())) { setStarting(false); return }

    // Voice activity detection in the browser (Silero VAD). Only speech segments are sent.
    try {
      const { MicVAD } = await import('@ricky0123/vad-web')
      const base = new URL('vad/', document.baseURI).href
      vadRef.current = await MicVAD.new({
        model: 'v5',
        baseAssetPath: base,
        onnxWASMBasePath: base,
        positiveSpeechThreshold: 0.6,
        negativeSpeechThreshold: 0.4,
        minSpeechMs: 350,
        redemptionMs: 700,
        preSpeechPadMs: 300,
        getStream: () => navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        }),
        onSpeechStart: () => {
          if (player.busy && !bargeRef.current) return
          hearingRef.current = true
          if (player.busy) {
            player.duck(0.2)
            if (fastBargeRef.current) {
              // still talking after half a second: stop the agent at once, like a person would
              clearTimeout(bargeTimerRef.current)
              bargeTimerRef.current = setTimeout(() => {
                if (hearingRef.current && player.busy) {
                  player.stop()
                  send({ type: 'interrupt', turn: turnRef.current })
                  updateTurn(turnRef.current, (x) => ({ ...x, interrupted: true }))
                }
              }, 500)
            }
          }
          refreshPhase()
        },
        onVADMisfire: () => { clearTimeout(bargeTimerRef.current); hearingRef.current = false; player.unduck(); refreshPhase() },
        onSpeechEnd: (audio) => {
          clearTimeout(bargeTimerRef.current)
          hearingRef.current = false
          if (player.busy && !bargeRef.current) { refreshPhase(); return }
          endOfSpeechRef.current = performance.now()
          send(floatTo16BitPCM(audio))
          refreshPhase()
        },
      })
      await vadRef.current.start()
    } catch (err) {
      setError(`Microphone or voice detector failed: ${err.message || err}. Allow mic access and reload.`)
      stoppingRef.current = true
      wsRef.current?.close()
      wsRef.current = null
      setStarting(false)
      return
    }
    setRunning(true)
    setStarting(false)
    setNeedTap(player.needsGesture)
    refreshPhase()
  }, [handleEvent, refreshPhase, stop, echoBoost])

  // Start by itself when the page opens, if the microphone was allowed before.
  useEffect(() => {
    if (!autoStart) return
    let cancelled = false
    const t = setTimeout(async () => {
      try {
        const st = await navigator.permissions?.query({ name: 'microphone' })
        if (!cancelled && st?.state === 'granted' && !wsRef.current) start()
      } catch { /* Safari: no permission query, use the button */ }
    }, 300)
    return () => { cancelled = true; clearTimeout(t) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Browsers allow sound only after one tap or key press on the page.
  useEffect(() => {
    if (!needTap) return
    const on = () => {
      playerRef.current?.resume()
      setTimeout(() => setNeedTap(!!playerRef.current?.needsGesture), 300)
    }
    window.addEventListener('pointerdown', on)
    window.addEventListener('keydown', on)
    return () => { window.removeEventListener('pointerdown', on); window.removeEventListener('keydown', on) }
  }, [needTap])

  useEffect(() => () => { stop() }, [stop])

  const toggleWake = (v) => { setWakeRequired(v); wakeRef.current = v; send({ type: 'config', wake: v }); refreshPhase() }

  const sendTyped = (ev) => {
    ev.preventDefault()
    const t = typed.trim()
    if (!t || !wsRef.current) return
    playerRef.current?.unlock()
    endOfSpeechRef.current = performance.now()
    send({ type: 'text', text: t })
    setTyped('')
  }

  const reset = () => { playerRef.current?.stop(); send({ type: 'reset' }); setLog([]); setMetrics(null) }

  const label = {
    off: 'Press Start to begin',
    asleep: `Listening. Say "${agentName}" to start`,
    awake: wakeRequired ? `Go ahead, I'm listening${awakeLeft ? ` (${awakeLeft}s)` : ''}` : "Go ahead, I'm listening",
    hearing: 'Hearing you…',
    thinking: 'Thinking…',
    speaking: allowBargeIn ? 'Speaking. You can interrupt' : 'Speaking…',
  }[phase]

  return (
    <main className="talk">
      <section className="call">
        <div className={`orb ${phase}`}><span className="ring" /><span className="core" /></div>
        <p className="status">{label}</p>
        {pendingContact && <p className="hint">Waiting for your name and phone number</p>}
        {needTap && <p className="hint tap">Listening. Tap anywhere on the page to turn on the agent's voice.</p>}
        {!running ? (
          <button className="primary" onClick={start} disabled={starting}>{starting ? 'Starting…' : 'Start listening'}</button>
        ) : (
          <div className="row">
            <button className="ghost" onClick={stop}>Stop</button>
            <button className="ghost" onClick={reset}>New conversation</button>
          </div>
        )}
        <form className="typed" onSubmit={sendTyped}>
          <input value={typed} onChange={(e) => setTyped(e.target.value)} placeholder={running ? 'Or type a question…' : 'Start first to type'} disabled={!running} />
          <button type="submit" disabled={!running}>Send</button>
        </form>
        <div className="toggles">
          <label><input type="checkbox" checked={wakeRequired} onChange={(e) => toggleWake(e.target.checked)} /> Require calling name "{agentName}"</label>
          <label><input type="checkbox" checked={allowBargeIn} onChange={(e) => setAllowBargeIn(e.target.checked)} /> Allow interruptions</label>
          <label><input type="checkbox" checked={fastBarge} onChange={(e) => setFastBarge(e.target.checked)} disabled={!allowBargeIn} /> Stop instantly when I talk over it</label>
          <label><input type="checkbox" checked={showIgnored} onChange={(e) => setShowIgnored(e.target.checked)} /> Show ignored speech</label>
          <label><input type="checkbox" checked={autoStart} onChange={(e) => setAutoStart(e.target.checked)} /> Start listening when the page opens</label>
          <label title="Applies the next time you press Start"><input type="checkbox" checked={echoBoost} onChange={(e) => setEchoBoost(e.target.checked)} /> Echo cancellation boost (for laptop speakers)</label>
        </div>
        <p className="tip">On laptop speakers the agent can still hear itself sometimes. Headphones give the cleanest test; a phone call keeps the two voices apart anyway.</p>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="log" ref={logRef}>
        {log.length === 0 && (
          <div className="empty">
            <p>Press Start, then try:</p>
            <ul>
              <li>“{agentName}, what are the rates for pool villas?”</li>
              <li>“{agentName}, do you have airport pickup?”</li>
              <li>Then a follow-up without the name: “and for a family suite?”</li>
            </ul>
          </div>
        )}
        {log.map((e, i) => {
          if (e.kind === 'ignored') return showIgnored ? (
            <div key={i} className="ignored">Ignored ({e.reason}): “{e.text}”</div>
          ) : null
          if (e.kind === 'user') return <div key={i} className="bubble user"><span className="who">You</span>{e.text}</div>
          return (
            <div key={i} className={`bubble agent ${e.unknown ? 'unknown' : ''}`}>
              <span className="who">{agentName}</span>
              {e.text || <span className="dots"><i /><i /><i /></span>}
              {e.interrupted && <span className="sources">Interrupted by you</span>}
              {e.sources?.length > 0 && (
                <span className="sources">Retrieved: {e.sources.map((s) => `${s.title} (${s.score})`).join(' · ')}</span>
              )}
            </div>
          )
        })}
      </section>

      <aside className="metrics">
        <div className="metrics-header">
          <details open={typeof window !== 'undefined' && window.innerWidth > 980}>
            <summary>Last turn latency</summary>
          </details>
          <a className="ghost btn" href="#/conversations" style={{ fontSize: '12px', padding: '4px 10px' }}>View all conversations</a>
        </div>
          {metrics ? (
            <>
              <dl>
                {STAGES.map(([k, name]) => metrics[k] != null && (
                  <div key={k}><dt>{name}</dt><dd>{metrics[k]} ms</dd></div>
                ))}
              </dl>
              {metrics.heard_ms != null && (
                <p className="headline"><strong>{metrics.heard_ms} ms</strong> from end of your speech to first sound</p>
              )}
              <p className="note">Server times start when your speech reached the server.</p>
            </>
          ) : <p className="note">Speak to see per-stage timings.</p>}
          {health && (
            <dl className="stack">
              <div><dt>Device</dt><dd>{health.device}{health.mock ? ' (mock)' : ''}</dd></div>
              <div><dt>STT</dt><dd>{health.stt_engine}</dd></div>
              <div><dt>LLM</dt><dd>{health.llm}</dd></div>
              <div><dt>Voice</dt><dd>{health.tts}</dd></div>
              <div><dt>Knowledge</dt><dd>{health.kb_version} · {health.kb_chunks} chunks</dd></div>
              <div><dt>v2 Redis</dt><dd>{health.v2?.redis_ok != null ? (health.v2.redis_ok ? 'connected' : 'unavailable') : 'n/a'}</dd></div>
              <div><dt>v2 SQLite</dt><dd>{health.v2?.sqlite_ok != null ? (health.v2.sqlite_ok ? 'connected' : 'unavailable') : 'n/a'}</dd></div>
              <div><dt>v2 Hybrid RAG</dt><dd>{health.v2?.hybrid_rag ? 'on' : 'off'}</dd></div>
            </dl>
          )}
        </details>
      </aside>
    </main>
  )
}
