// Audio helpers: gapless playback of WAV chunks with ducking (for barge-in), and PCM conversion.

export function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length)
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]))
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  return out.buffer
}

export class ChunkPlayer {
  constructor() {
    this.ctx = null
    this.gain = null
    this.nextTime = 0
    this.sources = new Set()
    this.chain = Promise.resolve()
    this.turn = null
    this.doneTurns = new Set()
    this.onStart = null // (turn, index) when a sentence starts playing
    this.onIdle = null // (turn) when a finished turn has played out
    this.onActivity = null // (playing: boolean)
    this.timers = []
  }

  // echoBoost: play the agent's voice through a local WebRTC connection instead of straight to the speakers.
  // Chrome's echo canceller removes audio that arrives over WebRTC from the microphone signal, which helps
  // the agent tell your voice apart from its own when you use laptop speakers. Falls back silently.
  unlock(echoBoost = false) {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext
      this.ctx = new AC()
      this.gain = this.ctx.createGain()
      if (echoBoost && window.RTCPeerConnection && this.ctx.createMediaStreamDestination) {
        this.setupLoopback().catch(() => this.gain.connect(this.ctx.destination))
      } else {
        this.gain.connect(this.ctx.destination)
      }
    }
    this.resume()
    return this.ctx
  }

  async setupLoopback() {
    const dest = this.ctx.createMediaStreamDestination()
    this.gain.connect(dest)
    const a = new RTCPeerConnection()
    const b = new RTCPeerConnection()
    a.onicecandidate = (e) => e.candidate && b.addIceCandidate(e.candidate).catch(() => {})
    b.onicecandidate = (e) => e.candidate && a.addIceCandidate(e.candidate).catch(() => {})
    const remote = new Promise((res) => { b.ontrack = (e) => res(e.streams[0] || new MediaStream([e.track])) })
    dest.stream.getTracks().forEach((t) => a.addTrack(t, dest.stream))
    const offer = await a.createOffer()
    await a.setLocalDescription(offer)
    await b.setRemoteDescription(offer)
    const answer = await b.createAnswer()
    await b.setLocalDescription(answer)
    await a.setRemoteDescription(answer)
    const stream = await Promise.race([remote, new Promise((_, rej) => setTimeout(() => rej(new Error('loopback timeout')), 4000))])
    const el = new Audio()
    el.autoplay = true
    el.srcObject = stream
    this.audioEl = el
    this.peers = [a, b]
    this.loopback = true
    await el.play().catch(() => {})
  }

  resume() {
    if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume().catch(() => {})
    if (this.audioEl && this.audioEl.paused) this.audioEl.play().catch(() => {})
  }

  get needsGesture() {
    return !!this.ctx && (this.ctx.state !== 'running' || (this.audioEl ? this.audioEl.paused : false))
  }

  push(b64, turn, index) {
    this.unlock()
    if (this.turn !== turn) {
      this.stop()
      this.turn = turn
    }
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
    this.chain = this.chain.then(async () => {
      if (this.turn !== turn) return
      let buf
      try { buf = await this.ctx.decodeAudioData(bytes.buffer) } catch { return }
      if (this.turn !== turn) return
      const src = this.ctx.createBufferSource()
      src.buffer = buf
      src.connect(this.gain)
      const at = Math.max(this.ctx.currentTime + 0.03, this.nextTime)
      src.start(at)
      this.nextTime = at + buf.duration
      const delay = Math.max(0, (at - this.ctx.currentTime) * 1000)
      this.timers.push(setTimeout(() => this.onStart && this.onStart(turn, index), delay))
      this.sources.add(src)
      this.onActivity && this.onActivity(true)
      src.onended = () => {
        this.sources.delete(src)
        this.checkIdle()
      }
    })
  }

  markDone(turn) {
    this.doneTurns.add(turn)
    this.chain.then(() => this.checkIdle())
  }

  checkIdle() {
    if (this.sources.size === 0) {
      this.onActivity && this.onActivity(false)
      if (this.turn != null && this.doneTurns.has(this.turn)) {
        const t = this.turn
        this.turn = null
        this.onIdle && this.onIdle(t)
      }
    }
  }

  duck(level = 0.2) {
    if (this.gain) this.gain.gain.setTargetAtTime(level, this.ctx.currentTime, 0.03)
  }

  unduck() {
    if (this.gain) this.gain.gain.setTargetAtTime(1, this.ctx.currentTime, 0.05)
  }

  stop() {
    this.sources.forEach((s) => { try { s.stop() } catch { /* already stopped */ } })
    this.sources.clear()
    this.timers.forEach(clearTimeout)
    this.timers = []
    this.nextTime = 0
    this.chain = Promise.resolve()
    this.turn = null
    this.unduck()
    this.onActivity && this.onActivity(false)
  }

  get busy() { return this.sources.size > 0 }
}

export function playOnce(b64) {
  const AC = window.AudioContext || window.webkitAudioContext
  const ctx = new AC()
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
  ctx.decodeAudioData(bytes.buffer).then((buf) => {
    const s = ctx.createBufferSource()
    s.buffer = buf
    s.connect(ctx.destination)
    s.start()
    s.onended = () => ctx.close()
  })
}
