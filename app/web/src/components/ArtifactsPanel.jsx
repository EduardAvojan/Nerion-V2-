import React, { useState, useEffect } from 'react'

export default function ArtifactsPanel() {
  const [artifacts, setArtifacts] = useState([])

  useEffect(() => {
    fetch('/api/artifacts')
      .then(res => res.json())
      .then(data => setArtifacts(data.artifacts || []))
      .catch(err => console.error('Failed to fetch artifacts:', err))
  }, [])

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          📄 ARTIFACTS ({artifacts.length})
        </div>
      </div>
      <div className="panel-content">
        {artifacts.length > 0 ? (
          <ul className="item-list">
            {artifacts.slice(0, 5).map((artifact, i) => (
              <li key={i}>
                {artifact.type === 'security' && '🔒 '}
                {artifact.type === 'plan' && '📊 '}
                {artifact.type === 'analysis' && '🔍 '}
                {artifact.name}
              </li>
            ))}
          </ul>
        ) : (
          <div style={{ color: 'var(--text-dim)', fontSize: '13px' }}>
            No artifacts yet
          </div>
        )}
        <button className="btn btn-small" style={{ marginTop: '12px', width: '100%' }}>
          Browse All →
        </button>
      </div>
    </div>
  )
}
