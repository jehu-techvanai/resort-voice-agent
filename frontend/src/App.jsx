import { useCallback, useEffect, useState } from 'react'
import Talk from './Talk.jsx'
import Settings from './Settings.jsx'
import Conversations from './Conversations.jsx'
import LogFiles from './LogFiles.jsx'

function useHashRoute() {
  const [route, setRoute] = useState(() => location.hash.replace(/^#\/?/, '') || 'talk')
  useEffect(() => {
    const on = () => setRoute(location.hash.replace(/^#\/?/, '') || 'talk')
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}


export default function App() {
  const route = useHashRoute()
  const [health, setHealth] = useState(null)
  const [err, setErr] = useState('')
  const refresh = useCallback(() => {
    fetch('/api/health').then((r) => r.json()).then(setHealth).catch(() => setErr('Agent backend not reachable'))
  }, [])
  useEffect(() => { refresh() }, [refresh, route])

  return (
    <div className="page">
      <header>
        <div>
          <p className="eyebrow">{route === 'settings' ? 'Settings' : route === 'conversations' ? 'Conversations' : 'Front desk · voice agent'}</p>
          <h1>{health?.resort || 'Voice agent'}</h1>
        </div>
        <nav className="top">
          <a className={`ghost btn ${route === 'talk' ? 'on' : ''}`} href="#/">Agent</a>
          <a className={`ghost btn ${route === 'settings' ? 'on' : ''}`} href="#/settings">Settings</a>
          <a className={`ghost btn ${route === 'conversations' || route === 'logs' ? 'on' : ''}`} href="#/conversations">Conversations</a>
        </nav>
      </header>
      {err && <p className="error">{err}</p>}
      {route === 'settings' ? <Settings onChanged={refresh} />
       : route === 'conversations' ? <Conversations />
       : route === 'logs' ? <LogFiles onBack={() => { location.hash = '#/conversations' }} />
       : <Talk key={health?.agent || 'x'} health={health} />}
    </div>
  )
}
