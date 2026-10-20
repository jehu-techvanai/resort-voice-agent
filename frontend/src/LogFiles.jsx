import { useEffect, useRef, useState } from 'react'

function getToken() { try { return sessionStorage.getItem('adminToken') || '' } catch { return '' } }

async function api(path, { method = 'GET', body, raw } = {}) {
  const headers = { Authorization: `Bearer ${getToken()}` }
  let payload
  if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body) }
  const res = await fetch(path, { method, headers, body: payload })
  if (res.status === 401) throw Object.assign(new Error('Please log in again'), { auth: true })
  if (raw) return res
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
  return data
}

export default function LogFiles({ onBack }) {
  const [files, setFiles] = useState(null)
  const [selected, setSelected] = useState(null)
  const [content, setContent] = useState(null)
  const [loading, setLoading] = useState(false)
  const viewerRef = useRef(null)

  const loadFiles = async () => {
    try {
      const d = await api('/api/admin/log-files')
      setFiles(d.files || [])
      setSelected(null)
      setContent(null)
    } catch (e) { setFiles({ error: e.message }) }
  }

  const viewFile = async (f) => {
    setSelected(f)
    setLoading(true)
    try {
      const d = await api(`/api/admin/log-files/${f.name}/content`)
      setContent(d.content)
    } catch (e) { setContent({ error: e.message }) }
    finally { setLoading(false) }
  }

  const downloadFile = (f) => {
    window.location.href = `/api/admin/log-files/${f.name}/download`
  }

  useEffect(() => { loadFiles() }, [])

  useEffect(() => {
    if (viewerRef.current && content && !content.error) {
      viewerRef.current.scrollTop = viewerRef.current.scrollHeight
    }
  }, [content])

  return (
    <main className="settings">
      <nav className="tabs">
        <button className="link" onClick={onBack}>← Back to conversations</button>
        <button className="link" onClick={() => { setSelected(null); setContent(null) }}>
          {selected ? 'Hide viewer' : 'Refresh'}
        </button>
      </nav>

      <div className="log-files">
        <div className="card">
          <h3>Log files</h3>
          <p className="note">Per-conversation transcript and agent log files written to the data directory.</p>
          {!files ? (
            <p className="note">Loading…</p>
          ) : files.error ? (
            <p className="error">{files.error}</p>
          ) : files.length === 0 ? (
            <p className="note">No log files yet. Start a conversation to generate one.</p>
          ) : (
            <div className="log-file-list">
              {files.map((f) => (
                <div key={f.name} className={`log-file-item ${selected?.name === f.name ? 'selected' : ''}`}>
                  <div className="log-file-info">
                    <span className="log-file-name">{f.name}</span>
                    <span className="log-file-meta">
                      <span className={f.type === 'transcript' ? 'badge transcript' : 'badge agent'}>{f.type}</span>
                      <span className="log-file-tag">{f.tag}</span>
                      <span className="log-file-size">{(f.size / 1024).toFixed(1)} KB</span>
                    </span>
                  </div>
                  <div className="log-file-actions">
                    <button className="ghost" onClick={() => viewFile(f)}>View</button>
                    <button className="ghost" onClick={() => downloadFile(f)}>Download</button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {selected && (
          <div className="card">
            <div className="log-viewer-header">
              <h3>{selected.name}</h3>
              <div className="row">
                <span className={`badge ${selected.type === 'transcript' ? 'transcript' : 'agent'}`}>{selected.type}</span>
                <span className="log-file-tag">{selected.tag}</span>
                <button className="ghost" onClick={() => downloadFile(selected)}>Download</button>
              </div>
            </div>
            {loading ? (
              <p className="note">Loading…</p>
            ) : content && !content.error ? (
              <pre className="log-viewer" ref={viewerRef}>{content}</pre>
            ) : (
              <p className="error">{content?.error || 'Unknown error'}</p>
            )}
          </div>
        )}
      </div>
    </main>
  )
}
