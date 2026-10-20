import { useEffect, useRef, useState } from 'react'
import { playOnce } from './audio.js'

function getToken() { try { return sessionStorage.getItem('adminToken') || '' } catch { return '' } }
function setToken(t) { try { t ? sessionStorage.setItem('adminToken', t) : sessionStorage.removeItem('adminToken') } catch { /* ignore */ } }

async function api(path, { method = 'GET', body, form, raw } = {}) {
  const headers = { Authorization: `Bearer ${getToken()}` }
  let payload
  if (form) payload = form
  else if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body) }
  const res = await fetch(path, { method, headers, body: payload })
  if (res.status === 401) { setToken(''); throw Object.assign(new Error('Please log in again'), { auth: true }) }
  if (raw) return res
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
  return data
}

function Login({ onDone }) {
  const [pw, setPw] = useState('')
  const [err, setErr] = useState('')
  const submit = async (e) => {
    e.preventDefault()
    setErr('')
    try {
      const r = await fetch('/api/admin/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ password: pw }) })
      const d = await r.json()
      if (!r.ok) throw new Error(d.detail || 'Login failed')
      setToken(d.token)
      onDone()
    } catch (e2) { setErr(e2.message) }
  }
  return (
    <form className="card login" onSubmit={submit}>
      <h2>Settings login</h2>
      <p className="note">The password is printed in the agent's terminal at startup, or set ADMIN_PASSWORD in .env.</p>
      <input type="password" value={pw} onChange={(e) => setPw(e.target.value)} placeholder="Admin password" autoFocus />
      <button className="primary" type="submit">Log in</button>
      {err && <p className="error">{err}</p>}
    </form>
  )
}

function Persona({ onSaved }) {
  const [s, setS] = useState(null)
  const [voices, setVoices] = useState([])
  const [engines, setEngines] = useState([])
  const [sarvamKey, setSarvamKey] = useState(false)
  const [msg, setMsg] = useState('')
  useEffect(() => { api('/api/admin/settings').then((d) => { setS(d.settings); setVoices(d.voices); setEngines(d.stt_engines || []); setSarvamKey(!!d.sarvam_key_set) }).catch((e) => setMsg(e.message)) }, [])
  if (!s) return <p className="note">{msg || 'Loading…'}</p>
  const set = (k) => (e) => setS({ ...s, [k]: e.target.type === 'checkbox' ? e.target.checked : e.target.value })
  const save = async () => {
    setMsg('Saving…')
    try {
      const d = await api('/api/admin/settings', { method: 'PUT', body: { ...s, speed: Number(s.speed), followup_seconds: Number(s.followup_seconds), min_retrieval_score: Number(s.min_retrieval_score) } })
      setS(d.settings); setMsg('Saved. New conversations use these settings straight away.'); onSaved?.()
    } catch (e) { setMsg(e.message) }
  }
  const sample = async () => {
    setMsg('Generating sample…')
    try { const d = await api(`/api/admin/voice-sample?voice=${s.voice}&speed=${s.speed}`); playOnce(d.wav_b64); setMsg('') } catch (e) { setMsg(e.message) }
  }
  return (
    <div className="card form">
      <div className="grid2">
        <label>Calling name<input value={s.agent_name} onChange={set('agent_name')} /></label>
        <label>Business name<input value={s.business_name} onChange={set('business_name')} /></label>
        <label>Type of business<input value={s.business_type} onChange={set('business_type')} placeholder="resort, dental clinic, car dealer…" /></label>
        <label>Voice
          <div className="row">
            <select value={s.voice} onChange={set('voice')}>{voices.map((v) => <option key={v}>{v}</option>)}</select>
            <button type="button" className="ghost" onClick={sample}>Play</button>
          </div>
        </label>
        <label>Speech to text
          <select value={s.stt_engine} onChange={set('stt_engine')}>
            {engines.map((e) => <option key={e} value={e}>{{
              'whisper-small': 'Whisper small (fast, weaker on accents)',
              'whisper-medium': 'Whisper medium (slower, better)',
              'whisper-large-v3-turbo': 'Whisper large-v3-turbo (best local; first use downloads ~1.6 GB)',
              sarvam: `Sarvam saaras:v4 (cloud, Indian accents)${sarvamKey ? '' : ': needs SARVAM_API_KEY in .env'}`,
            }[e] || e}</option>)}
          </select>
        </label>
        <label>Speaking speed: {Number(s.speed).toFixed(2)}<input type="range" min="0.7" max="1.4" step="0.05" value={s.speed} onChange={set('speed')} /></label>
        <label>Follow-up window (seconds without calling name)<input type="number" min="0" max="120" value={s.followup_seconds} onChange={set('followup_seconds')} /></label>
      </div>
      <label>Greeting (when only the calling name is said; {'{agent}'} and {'{business}'} are filled in)<input value={s.greeting} onChange={set('greeting')} /></label>
      <label>Reply when it does not know<textarea rows="2" value={s.fallback_text} onChange={set('fallback_text')} /></label>
      <label>Extra instructions (tone, things to always mention or never say)<textarea rows="3" value={s.extra_instructions} onChange={set('extra_instructions')} /></label>
      <div className="grid2">
        <label className="check"><input type="checkbox" checked={s.wake_required_web} onChange={set('wake_required_web')} /> Require calling name in the web page (default for new sessions)</label>
        <label>Retrieval threshold: {Number(s.min_retrieval_score).toFixed(2)} <span className="note">(below this the agent says it does not know; tune with the Test tab)</span>
          <input type="range" min="0" max="0.8" step="0.01" value={s.min_retrieval_score} onChange={set('min_retrieval_score')} /></label>
      </div>
      <div className="row"><button className="primary" onClick={save}>Save</button><span className="note">{msg}</span></div>
    </div>
  )
}

function Knowledge({ onChanged }) {
  const [st, setSt] = useState(null)
  const [busy, setBusy] = useState('')
  const [msg, setMsg] = useState('')
  const [report, setReport] = useState(null)
  const [note, setNote] = useState('')
  const [q, setQ] = useState('')
  const [chunks, setChunks] = useState(null)
  const fileRef = useRef(null)

  const load = () => api('/api/admin/kb').then(setSt).catch((e) => setMsg(e.message))
  const loadChunks = (query = q) => api(`/api/admin/kb/chunks?limit=200&q=${encodeURIComponent(query)}`).then(setChunks).catch((e) => setMsg(e.message))
  useEffect(() => { load(); loadChunks('') }, [])

  const upload = async (files) => {
    if (!files?.length) return
    const form = new FormData()
    for (const f of files) form.append('files', f)
    setBusy('upload'); setMsg('')
    try {
      const d = await api('/api/admin/kb/upload', { method: 'POST', form })
      setMsg(d.errors.length ? d.errors.join('; ') : `Added ${d.added.map((a) => a.name).join(', ')}. Press Train to use them.`)
      load()
    } catch (e) { setMsg(e.message) }
    setBusy('')
    if (fileRef.current) fileRef.current.value = ''
  }
  const remove = async (name) => { await api(`/api/admin/kb/staged/${encodeURIComponent(name)}`, { method: 'DELETE' }); load() }
  const train = async () => {
    setBusy('train'); setMsg('Reading files and building the index…'); setReport(null)
    try {
      const d = await api('/api/admin/kb/train', { method: 'POST', body: { note } })
      setReport(d.version)
      setMsg(`Done. ${d.version.version} is now active with ${d.version.chunks} chunks.`)
      setNote(''); load(); loadChunks(''); onChanged?.()
    } catch (e) { setMsg(e.message) }
    setBusy('')
  }
  const activate = async (v) => {
    setBusy('activate')
    try { await api('/api/admin/kb/activate', { method: 'POST', body: { version: v } }); setMsg(`${v} is now active.`); load(); loadChunks(''); onChanged?.() } catch (e) { setMsg(e.message) }
    setBusy('')
  }

  if (!st) return <p className="note">{msg || 'Loading…'}</p>
  return (
    <div className="kb">
      <div className="card">
        <h3>Active knowledge: {st.active} <span className="note">({st.chunks} chunks)</span></h3>
        <p className="note">Upload DOCX, PDF, MD or TXT. FAQ documents (a question followed by its answer) work best. Scanned PDFs are not supported yet. Train builds a new version from all files listed below.</p>
        <div
          className="drop"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); upload(e.dataTransfer.files) }}
          onClick={() => fileRef.current?.click()}
        >
          {busy === 'upload' ? 'Uploading…' : 'Drop files here or click to choose'}
          <input ref={fileRef} type="file" multiple accept=".docx,.pdf,.md,.txt" hidden onChange={(e) => upload(e.target.files)} />
        </div>
        {st.staged.length > 0 ? (
          <ul className="files">
            {st.staged.map((f) => (
              <li key={f.name}><span>{f.name}</span><span className="note">{Math.ceil(f.size / 1024)} KB</span><button className="link" onClick={() => remove(f.name)}>Remove</button></li>
            ))}
          </ul>
        ) : <p className="note">No files waiting.</p>}
        <div className="row">
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Note for this version (optional)" />
          <button className="primary" onClick={train} disabled={!!busy || !st.staged.length}>{busy === 'train' ? 'Training…' : 'Train'}</button>
        </div>
        {msg && <p className="note strong">{msg}</p>}
        {report && (
          <ul className="files">
            {report.files.map((f) => <li key={f.file}><span>{f.file}</span><span className={f.ok ? 'ok' : 'error'}>{f.ok ? `${f.chunks} chunks` : f.error}</span></li>)}
          </ul>
        )}
      </div>

      <div className="card">
        <h3>Versions</h3>
        <table className="table">
          <thead><tr><th>Version</th><th>Created</th><th>Files</th><th>Chunks</th><th /></tr></thead>
          <tbody>
            {st.versions.map((v) => (
              <tr key={v.version}>
                <td>{v.version}{v.note ? <div className="note">{v.note}</div> : null}</td>
                <td>{v.created}</td>
                <td>{v.files.map((f) => f.file).join(', ')}</td>
                <td>{v.chunks}</td>
                <td>{v.version === st.active ? <span className="ok">Active</span> : <button className="ghost" disabled={!!busy} onClick={() => activate(v.version)}>Use this</button>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>What the agent knows ({chunks?.total ?? 0} chunks)</h3>
        <form className="row" onSubmit={(e) => { e.preventDefault(); loadChunks() }}>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search the chunks" />
          <button className="ghost" type="submit">Search</button>
        </form>
        <div className="chunks">
          {chunks?.chunks.map((c) => (
            <div key={c.id} className="chunk"><strong>{c.title}</strong><span className="note"> · {c.source}</span><p>{c.text}</p></div>
          ))}
        </div>
      </div>
    </div>
  )
}

function TestBox() {
  const [qn, setQn] = useState('')
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const ask = async (e) => {
    e.preventDefault()
    if (!qn.trim()) return
    setBusy(true); setErr('')
    try { setRes(await api('/api/admin/test', { method: 'POST', body: { question: qn } })) } catch (e2) { setErr(e2.message) }
    setBusy(false)
  }
  return (
    <div className="card">
      <p className="note">Ask a question as a customer would. You see the answer text, whether it counted as unknown, and the chunks retrieved with their scores. If good questions are rejected, lower the retrieval threshold in Persona; if unrelated questions get answered, raise it.</p>
      <form className="row" onSubmit={ask}>
        <input value={qn} onChange={(e) => setQn(e.target.value)} placeholder="e.g. Do you have parking?" />
        <button className="primary" disabled={busy}>{busy ? 'Asking…' : 'Ask'}</button>
      </form>
      {err && <p className="error">{err}</p>}
      {res && (
        <div className="test-result">
          <p className={res.unknown ? 'answer unknown' : 'answer'}>{res.answer}</p>
          <p className="note">{res.unknown ? `Unknown: ${res.reason}. ` : ''}Best score {res.best_score}{res.ms ? ` · ${res.ms} ms` : ''}</p>
          {res.retrieved.map((r, i) => (
            <div key={i} className="chunk"><strong>{r.score}</strong> · {r.title}<span className="note"> · {r.source}</span><p>{r.text}</p></div>
          ))}
        </div>
      )}
    </div>
  )
}

function Unanswered() {
  const [items, setItems] = useState(null)
  const [hideResolved, setHideResolved] = useState(true)
  const [err, setErr] = useState('')
  const load = () => api('/api/admin/unanswered').then((d) => setItems(d.items)).catch((e) => setErr(e.message))
  useEffect(() => { load() }, [])
  const toggle = async (it) => { await api(`/api/admin/unanswered/${it.id}`, { method: 'PATCH', body: { resolved: !it.resolved } }); load() }
  const download = async () => {
    const res = await api('/api/admin/unanswered.csv', { raw: true })
    const blob = await res.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = 'unanswered.csv'; a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 1000)
  }
  if (!items) return <p className="note">{err || 'Loading…'}</p>
  const shown = items.filter((i) => !hideResolved || !i.resolved)
  return (
    <div className="card">
      <div className="row spread">
        <label className="check"><input type="checkbox" checked={hideResolved} onChange={(e) => setHideResolved(e.target.checked)} /> Hide resolved</label>
        <div className="row"><button className="ghost" onClick={load}>Refresh</button><button className="ghost" onClick={download}>Download CSV</button></div>
      </div>
      {shown.length === 0 ? <p className="note">Nothing here.</p> : (
        <table className="table">
          <thead><tr><th>When</th><th>Question</th><th>Name</th><th>Phone</th><th>Note</th><th /></tr></thead>
          <tbody>
            {shown.map((it) => (
              <tr key={it.id} className={it.resolved ? 'resolved' : ''}>
                <td>{it.time}</td><td>{it.question}</td><td>{it.name}</td><td>{it.phone}</td><td>{it.note}</td>
                <td><button className="link" onClick={() => toggle(it)}>{it.resolved ? 'Reopen' : 'Mark done'}</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

const TABS = [['persona', 'Persona'], ['knowledge', 'Knowledge base'], ['test', 'Test'], ['unanswered', 'Unanswered']]

export default function Settings({ onChanged }) {
  const [authed, setAuthed] = useState(!!getToken())
  const [tab, setTab] = useState('persona')
  const [, force] = useState(0)

  useEffect(() => {
    const onErr = (e) => { if (e.reason?.auth) { setAuthed(false) } }
    window.addEventListener('unhandledrejection', onErr)
    return () => window.removeEventListener('unhandledrejection', onErr)
  }, [])

  if (!authed) return <main className="settings"><Login onDone={() => { setAuthed(true); force((n) => n + 1) }} /></main>
  return (
    <main className="settings">
      <nav className="tabs">
        {TABS.map(([k, name]) => <button key={k} className={tab === k ? 'on' : ''} onClick={() => setTab(k)}>{name}</button>)}
        <button className="link" onClick={() => { setToken(''); setAuthed(false) }}>Log out</button>
      </nav>
      {tab === 'persona' && <Persona onSaved={onChanged} />}
      {tab === 'knowledge' && <Knowledge onChanged={onChanged} />}
      {tab === 'test' && <TestBox />}
      {tab === 'unanswered' && <Unanswered />}
    </main>
  )
}
