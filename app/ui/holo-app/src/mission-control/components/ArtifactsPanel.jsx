import React, { useState, useEffect } from 'react'
import './ArtifactsPanel.css'

export default function ArtifactsPanel({ isOpen, onClose }) {
  const [artifacts, setArtifacts] = useState([])
  const [filter, setFilter] = useState('all') // 'all', 'security', 'plan', 'analysis'

  useEffect(() => {
    if (isOpen) {
      fetch('/api/artifacts')
        .then(res => res.json())
        .then(data => setArtifacts(data.artifacts || []))
        .catch(err => console.error('Failed to fetch artifacts:', err))
    }
  }, [isOpen])

  if (!isOpen) return null

  const filteredArtifacts = filter === 'all'
    ? artifacts
    : artifacts.filter(a => a.type === filter)

  const getIcon = (type) => {
    switch(type) {
      case 'security': return '🔒'
      case 'plan': return '📊'
      case 'analysis': return '🔍'
      default: return '📄'
    }
  }

  return (
    <>
      <div className="artifacts-overlay" onClick={onClose} />
      <div className="artifacts-panel">
        <div className="artifacts-header">
          <h2>📄 Artifacts ({filteredArtifacts.length})</h2>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="artifacts-filters">
          <button
            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All
          </button>
          <button
            className={`filter-btn ${filter === 'security' ? 'active' : ''}`}
            onClick={() => setFilter('security')}
          >
            🔒 Security
          </button>
          <button
            className={`filter-btn ${filter === 'plan' ? 'active' : ''}`}
            onClick={() => setFilter('plan')}
          >
            📊 Plans
          </button>
          <button
            className={`filter-btn ${filter === 'analysis' ? 'active' : ''}`}
            onClick={() => setFilter('analysis')}
          >
            🔍 Analysis
          </button>
        </div>

        <div className="artifacts-content">
          {filteredArtifacts.length > 0 ? (
            <ul className="artifacts-list">
              {filteredArtifacts.map((artifact, i) => (
                <li key={i} className="artifact-item">
                  <div className="artifact-icon">{getIcon(artifact.type)}</div>
                  <div className="artifact-info">
                    <div className="artifact-name">{artifact.name}</div>
                    <div className="artifact-meta">
                      <span className="artifact-type">{artifact.type}</span>
                      {artifact.date && <span className="artifact-date">{artifact.date}</span>}
                    </div>
                  </div>
                  <div className="artifact-actions">
                    <button className="artifact-btn" title="View">👁️</button>
                    <button className="artifact-btn" title="Download">⬇️</button>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="artifacts-empty">
              <div className="empty-icon">📄</div>
              <div className="empty-text">No artifacts yet</div>
              <div className="empty-hint">
                Artifacts will appear here as the system generates them
              </div>
            </div>
          )}
        </div>

        <div className="artifacts-footer">
          <button className="btn-secondary" onClick={() => {
            if (confirm('Clear all artifacts?')) {
              setArtifacts([])
            }
          }}>
            Clear All
          </button>
          <button className="btn-primary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </>
  )
}
