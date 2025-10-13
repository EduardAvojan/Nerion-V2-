import React, { useState, useEffect } from 'react'
import TopBar from './components/TopBar'
import ImmuneVitalsPanel from './components/ImmuneVitalsPanel'
import SignalHealthPanel from './components/SignalHealthPanel'
import MemorySnapshotPanel from './components/MemorySnapshotPanel'
import Terminal from './components/Terminal'
import ArtifactsPanel from './components/ArtifactsPanel'
import UpgradeLanePanel from './components/UpgradeLanePanel'
import LearningTimelinePanel from './components/LearningTimelinePanel'
import './App.css'

function App() {
  const [systemStatus, setSystemStatus] = useState({
    status: 'healthy',
    uptime: 0,
    buildHealth: 98
  })

  const [signals, setSignals] = useState({
    voice: 'online',
    network: 'online',
    learning: 'active',
    llm: 'claude',
    coverage: 87,
    errors: 0
  })

  const [immune, setImmune] = useState({
    buildHealth: 98,
    active: true,
    threats: 2,
    autoFixes24h: 23
  })

  const [memory, setMemory] = useState({
    count: 234,
    pinned: [],
    recent: []
  })

  // Connect to events WebSocket
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/api/events')

    ws.onopen = () => {
      console.log('[Events] Connected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      console.log('[Events]', data)

      switch (data.type) {
        case 'health_update':
          setImmune(prev => ({ ...prev, ...data.data }))
          break
        case 'signal_update':
          setSignals(prev => ({ ...prev, ...data.data }))
          break
        case 'memory_update':
          setMemory(prev => ({ ...prev, ...data.data }))
          break
        // Add more event handlers as needed
      }
    }

    ws.onerror = (error) => {
      console.error('[Events] Error:', error)
    }

    ws.onclose = () => {
      console.log('[Events] Disconnected')
    }

    return () => ws.close()
  }, [])

  // Fetch initial data
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setSystemStatus(data))
      .catch(err => console.error('Failed to fetch health:', err))

    fetch('/api/memory')
      .then(res => res.json())
      .then(data => setMemory(data))
      .catch(err => console.error('Failed to fetch memory:', err))
  }, [])

  return (
    <div className="app">
      <TopBar
        status={systemStatus.status}
        uptime={systemStatus.uptime}
      />

      <div className="status-panels">
        <ImmuneVitalsPanel {...immune} />
        <SignalHealthPanel {...signals} />
        <MemorySnapshotPanel {...memory} />
      </div>

      <div className="main-interface">
        <Terminal />
      </div>

      <div className="control-panels">
        <ArtifactsPanel />
        <UpgradeLanePanel />
        <LearningTimelinePanel />
      </div>
    </div>
  )
}

export default App
