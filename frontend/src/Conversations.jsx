import { useEffect, useRef, useState } from 'react'

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
    <form className="card login" onSubmit={submit}>       <h2>Log in</h2>
      <p className="note">The password is printed in the agent terminal at startup, or set ADMIN_PASSWORD in .env.</p>


      <input type="password" value={pw} onChange={(e) => setPw(e.target.value)} placeholder="Admin password" autoFocus />
      <button className="primary" type="submit">Log in</button>
      {err && <p className="error">{err}</p>}
    </form>
  )
}

export default function Conversations() {
  const [authed, setAuthed] = useState(!!getToken())
  const [tab, setTab] = useState('list')
  const [convList, setConvList] = useState(null)
  const [conv, setConv] = useState(null)
  const [q, setQ] = useState('')
  const [mode, setMode] = useState('')
  const [limit, setLimit] = useState(25)
  const [offset, setOffset] = useState(0)
  const [note, setNote] = useState('')
  const [exportFmt, setExportFmt] = useState('json')
  const [countLabel, setCountLabel] = useState('')
  const [activeConvId, setActiveConvId] = useState('')
  const [playingTurn, setPlayingTurn] = useState(null)
  const audioRef = useRef(null)

  const load = async (offsetOverrides) => {
    const off = offsetOverrides ?? offset
    try {
      const params = new URLSearchParams({ limit: String(limit), offset: String(off) })
      if (q) params.set('q', q)
      if (mode) params.set('mode', mode)
      const d = await api(`/api/admin/conversations?${params.toString()}`)
      setConvList(d)
      setCountLabel(d.rows ? d.rows.length + ' conversations' : '')
    } catch (e) { setConvList({ error: e.message }); setCountLabel('') }
  }

  const loadConv = async (id) => {
    setActiveConvId(id)
    setNote('')
    setPlayingTurn(null)
    try {
      const d = await api(`/api/admin/conversations/${id}/turns`)
      setConv(d)
      setOffset(0)
      setTab('detail')
    } catch (e) { setConv({ error: e.message }); setTab('detail') }
  }

  const saveNote = async () => {
    if (!activeConvId) return
    try {
      await api(`/api/admin/conversations/${activeConvId}/note`, { method: 'PATCH', body: { note } })
      loadConv(activeConvId)
    } catch (e) { alert(e.message) }
  }

  const exportConv = async (id, fmt) => {
    try {
      const d = await api(`/api/admin/conversations/${id}/export?format=${fmt}`, { raw: true, method: 'GET' })
      const blob = await d.blob()
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
      a.download = `conversation-${activeConvId}.${fmt}`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000)
    } catch (e) { alert(e.message) }
  }

  const playTurnAudio = async (turn) => {
    if (!conv || !conv.turns) return
    const t = conv.turns.find((x) => x.turn_no === turn)
    if (!t || !t.agent_text) return
    setPlayingTurn(turn)
    try {
      const d = await api(`/api/admin/voice-sample?voice=af_heart&speed=1.0`, { raw: true })
      const b64 = (await d.json()).wav_b64
      if (!audioRef.current) audioRef.current = new Audio()
      audioRef.current.src = `data:audio/wav;base64,${b64}`
      await audioRef.current.play()
      setTimeout(() => setPlayingTurn(null), 300)
    } catch (e) { setPlayingTurn(null); alert(e.message) }
  }

  useEffect(() => { load() }, [q, mode, limit])
  useEffect(() => {
    const onErr = (e) => { if (e.reason?.auth) { setAuthed(false) } }
    window.addEventListener('unhandledrejection', onErr)
    return () => window.removeEventListener('unhandledrejection', onErr)
  }, [])

  if (!authed) return <main className="settings"><Login onDone={() => setAuthed(true)} /></main>

  const rows = convList?.rows || []

  return (
    <main className="settings">
      <nav className="tabs">
        <button className={tab === 'list' ? 'on' : ''} onClick={() => { setTab('list'); setConv(null); setNote('') }}>Conversations</button>
        <button className={tab === 'detail' ? 'on' : ''} disabled={!activeConvId} onClick={() => activeConvId && setTab('detail')}>Detail</button>
        <button className="link" onClick={() => { setToken(''); setAuthed(false) }}>Log out</button>
      </nav>

      {tab === 'list' && (
        <div className="conversations-list">
          <div className="card">
            <h3>Conversations</h3>
            <p className="note">Every completed and active conversation with the voice agent is listed here. Use search to filter by conversation id, session id, or mode.</p>
            <div className="grid2">
              <label>Search<input value={q} onChange={(e) => { setQ(e.target.value); setOffset(0) }} placeholder="conversation id / session id / mode" /></label>
              <label>Mode
                <select value={mode} onChange={(e) => { setMode(e.target.value); setOffset(0) }}>
                  <option value="">All modes</option>
                  <option value="web">web</option>
                  <option value="api">api</option>
                </select>
              </label>
            </div>
            <div className="row">
              <button className="ghost" onClick={() => { setOffset(0); load() }}>Refresh</button>
              <span className="note">{countLabel || 'Loading…'}</span>
            </div>
          </div>

          {convList?.error ? (
            <div className="card"><p className="error">{convList.error}</p><p className="note">SQLite may be disabled, or the backend is not running.</p></div>
          ) : rows.length === 0 ? (
            <div className="card"><p className="note">No conversations found.</p></div>
          ) : (
            <div className="card">
              <table className="table">
                <thead><tr><th>Conversation</th><th>Mode</th><th>Session</th><th>Started</th><th>Ended</th><th>Duration</th><th>Status</th><th /></tr></thead>
                <tbody>
                  {rows.map((c) => (
                    <tr key={c.id}>
                      <td><code>{c.id}</code></td>
                      <td>{c.mode}</td>
                      <td>{c.session_id}</td>
                      <td>{c.started}</td>
                      <td>{c.ended || '—'}</td>
                      <td>{c.total_seconds ? (c.total_seconds / 60).toFixed(1) + ' min' : '—'}</td>
                      <td><span className={c.status === 'done' ? 'ok' : ''}>{c.status}</span></td>
                      <td><button className="ghost" onClick={() => loadConv(c.id)}>View</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {rows.length === limit && (
                <div className="row" style={{ marginTop: 12 }}>
                  <button className="ghost" onClick={() => setOffset((o) => o + limit)}>Show more</button>
                  <span className="note"> Showing {rows.length}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {tab === 'detail' && conv && (
        <div className="conversation-detail">
          <div className="card">
            <h3>Conversation {conv.conversation.id}</h3>
            <div className="grid2">
              <div><dt>Mode</dt><dd>{conv.conversation.mode}</dd></div>
              <div><dt>Session</dt><dd>{conv.conversation.session_id}</dd></div>
              <div><dt>Started</dt><dd>{conv.conversation.started}</dd></div>
              <div><dt>Ended</dt><dd>{conv.conversation.ended || '—'}</dd></div>
              <div><dt>Duration</dt><dd>{conv.conversation.total_seconds ? (conv.conversation.total_seconds / 60).toFixed(1) + ' min' : '—'}</dd></div>
              <div><dt>Status</dt><dd>{conv.conversation.status}</dd></div>
            </div>
          </div>

          {conv.turns && conv.turns.length > 0 && (
            <div className="card">
              <h3>Transcript ({conv.turns.length} turns)</h3>
              <div className="row">
                <label>Export
                  <select value={exportFmt} onChange={(e) => setExportFmt(e.target.value)}>
                    <option value="json">JSON</option>
                    <option value="csv">CSV</option>
                  </select>
                </label>
                <button className="ghost" onClick={() => exportConv(conv.conversation.id, exportFmt)}>Download</button>
                <span className="note">Turns are stored in SQLite when v2 is enabled.</span>
              </div>
              <div className="transcript">
                {conv.turns.map((t) => (
                  <div key={t.id} className="turn">
                    <div className="turn-head">
                      <span className="turn-no">Turn {t.turn_no}</span>
                      <span className="turn-time">{t.created}</span>
                      {t.unknown ? <span className="unknown">unknown</span> : ''}
                      {t.retrieval_best_score != null ? <span className="note">score {t.retrieval_best_score}</span> : ''}
                      {t.latency_ms ? <span className="note">{t.latency_ms} ms</span> : ''}
                    </div>
                    <div className="turn-body">
                      <div className="bubble user"><span className="who">You</span>{t.user_text || '—'}</div>
                      <div className="bubble agent">
                        <span className="who">Agent</span>
                        {t.agent_text || '—'}
                        {t.sources && t.sources.length > 0 && (
                          <span className="sources">Retrieved: {t.sources.map((s) => `${s.title} (${s.score})`).join(' · ')}</span>
                        )}
                      </div>
                      <div className="turn-actions">
                        <button className="ghost" onClick={() => playTurnAudio(t.turn_no)} disabled={playingTurn === t.turn_no}>
                          {playingTurn === t.turn_no ? 'Playing…' : 'Play agent line'}
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {conv.contacts && conv.contacts.length > 0 && (
            <div className="card">
              <h3>Contacts captured in this conversation</h3>
              <table className="table">
                <thead><tr><th>Name</th><th>Phone</th><th>Captured at</th><th>Status</th></tr></thead>
                <tbody>
                  {conv.contacts.map((c) => (
                    <tr key={c.id}>
                      <td>{c.name || '—'}</td>
                      <td>{c.phone || '—'}</td>
                      <td>{c.captured_at}</td>
                      <td>{c.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="card">
            <h3>Notes</h3>
            <p className="note">Add internal notes for this conversation. Notes are stored in the SQLite conversation record.</p>
            <textarea rows="4" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Internal note for this conversation…" />
            <div className="row">
              <button className="primary" onClick={saveNote}>Save note</button>
              <span className="note">{note ? 'Unsaved changes' : ''}</span>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}
