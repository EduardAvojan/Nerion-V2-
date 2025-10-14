import React from 'react'
import './TopBar.css'

export default function TopBar({ status, uptime, onSettingsClick, onArtifactsClick, onTrainingClick, trainingNeedsAttention }) {
  const formatUptime = (seconds) => {
    const days = Math.floor(seconds / 86400)
    const hours = Math.floor((seconds % 86400) / 3600)
    return `${days}d ${hours}h`
  }

  return (
    <div className="top-bar">
      <div className="top-bar-left">
        <h1 className="app-title">
          <span className="app-icon">🧬</span>
          NERION MISSION CONTROL
        </h1>
        <div className="app-subtitle">Codebase: Nerion V2 • v1.0.0</div>
      </div>

      <div className="top-bar-center">
        <div className={`system-status status-${status}`}>
          <span className={`status-indicator status-${status === 'healthy' ? 'online' : 'error'}`}></span>
          Status: {status.toUpperCase()}
        </div>
        <div className="uptime">Uptime: {formatUptime(uptime)}</div>
      </div>

      <div className="top-bar-right">
        <button className="icon-btn" title="Artifacts" onClick={onArtifactsClick}>📄</button>
        <button
          className={`icon-btn ${trainingNeedsAttention ? 'glow-pulse' : ''}`}
          title="Training Dashboard"
          onClick={onTrainingClick}
        >
          📊
        </button>
        <button className="icon-btn" title="Settings" onClick={onSettingsClick}>⚙️</button>
        <button className="icon-btn" title="Help">❓</button>
      </div>
    </div>
  )
}
