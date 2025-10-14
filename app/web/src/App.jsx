import React, { useState, useEffect } from 'react'
import TopBar from './components/TopBar'
import ImmuneVitalsPanel from './components/ImmuneVitalsPanel'
import SignalHealthPanel from './components/SignalHealthPanel'
import MemorySnapshotPanel from './components/MemorySnapshotPanel'
import Terminal from './components/Terminal'
import GenesisView from './components/GenesisView'
import SettingsPanel from './components/SettingsPanel'
import ArtifactsPanel from './components/ArtifactsPanel'
import TrainingDashboard from './components/TrainingDashboard'
import ThoughtProcessPanel from './components/ThoughtProcessPanel'
import AmbientBackground from './components/AmbientBackground'
import './App.css'

function App() {
  const [mode, setMode] = useState('terminal') // 'terminal' or 'genesis'
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [artifactsOpen, setArtifactsOpen] = useState(false)
  const [trainingOpen, setTrainingOpen] = useState(false)
  const [trainingNeedsAttention, setTrainingNeedsAttention] = useState(true) // Set to true when training has updates
  const [theme, setTheme] = useState('dark')

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

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

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
      <AmbientBackground />

      <TopBar
        status={systemStatus.status}
        uptime={systemStatus.uptime}
        onSettingsClick={() => setSettingsOpen(true)}
        onArtifactsClick={() => setArtifactsOpen(true)}
        onTrainingClick={() => {
          setTrainingOpen(true)
          setTrainingNeedsAttention(false) // Clear notification when user opens dashboard
        }}
        trainingNeedsAttention={trainingNeedsAttention}
      />

      <div className="app-body">
        {/* Left Sidebar - Status Panels */}
        <div className="status-panels">
          <ImmuneVitalsPanel {...immune} />
          <SignalHealthPanel {...signals} />
          <MemorySnapshotPanel {...memory} />
        </div>

        {/* Center - Main Interface */}
        <div className="app-main">
          <div className="main-interface">
            <div className="mode-switcher">
              <button
                className={`mode-btn ${mode === 'terminal' ? 'active' : ''}`}
                onClick={() => setMode('terminal')}
              >
                <span className="mode-icon">⌨️</span>
                Terminal
              </button>
              <button
                className={`mode-btn ${mode === 'genesis' ? 'active' : ''}`}
                onClick={() => setMode('genesis')}
              >
                <span className="mode-icon">🧬</span>
                Genesis
              </button>
            </div>

            <div className="interface-content">
              {mode === 'terminal' && <Terminal />}
              {mode === 'genesis' && <GenesisView />}
            </div>
          </div>
        </div>

        {/* Right Sidebar - Thought Process */}
        <ThoughtProcessPanel />
      </div>

      <SettingsPanel
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={theme}
        onThemeChange={setTheme}
      />

      <ArtifactsPanel
        isOpen={artifactsOpen}
        onClose={() => setArtifactsOpen(false)}
      />

      <TrainingDashboard
        isOpen={trainingOpen}
        onClose={() => setTrainingOpen(false)}
      />
    </div>
  )
}

export default App
