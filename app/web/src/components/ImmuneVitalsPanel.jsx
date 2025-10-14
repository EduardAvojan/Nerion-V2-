import React from 'react'

export default function ImmuneVitalsPanel({ buildHealth, active, threats, autoFixes24h }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          ❤️ IMMUNE SYSTEM VITALS
        </div>
      </div>
      <div className="panel-content">
        <div className="stat">
          <div className="stat-label">Build Health</div>
          <div className="stat-value">
            {buildHealth}%
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${buildHealth}%` }}></div>
            </div>
          </div>
        </div>

        <div className="stat">
          <div className="stat-label">Protection Status</div>
          <div className="stat-value">
            <span className={`status-indicator ${active ? 'status-online' : 'status-error'}`}></span>
            {active ? 'ACTIVE' : 'INACTIVE'}
          </div>
        </div>

        <div className="stat">
          <div className="stat-label">Active Threats</div>
          <div className="stat-value">
            🦠 {threats}
          </div>
        </div>

        <div className="stat">
          <div className="stat-label">Auto-Fixes (24h)</div>
          <div className="stat-value">
            💉 {autoFixes24h}
          </div>
        </div>

        <div style={{ marginTop: '16px', display: 'flex', gap: '8px' }}>
          <button className="btn btn-small">Pause</button>
          <button className="btn btn-small">Full Scan</button>
        </div>
      </div>
    </div>
  )
}
