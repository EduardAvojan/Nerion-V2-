import React, { useState, useEffect } from 'react'

export default function UpgradeLanePanel() {
  const [upgrades, setUpgrades] = useState([])

  useEffect(() => {
    fetch('/api/upgrades/pending')
      .then(res => res.json())
      .then(data => setUpgrades(data.pending || []))
      .catch(err => console.error('Failed to fetch upgrades:', err))
  }, [])

  const upgrade = upgrades[0]

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          ⚡ UPGRADE LANE
        </div>
      </div>
      <div className="panel-content">
        {upgrade ? (
          <div>
            <div style={{ fontWeight: '600', marginBottom: '8px', color: 'var(--text-primary)' }}>
              {upgrade.title}
            </div>
            <div style={{ fontSize: '12px', marginBottom: '12px', color: 'var(--text-secondary)' }}>
              {upgrade.description}
            </div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              <span className={`badge badge-${upgrade.impact === 'low' ? 'info' : upgrade.impact === 'medium' ? 'warning' : 'error'}`}>
                Impact: {upgrade.impact}
              </span>
              <span className={`badge badge-${upgrade.risk === 'low' ? 'success' : upgrade.risk === 'medium' ? 'warning' : 'error'}`}>
                Risk: {upgrade.risk}
              </span>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button className="btn btn-small">Review Plan</button>
              <button className="btn btn-small">Apply Now</button>
              <button className="btn btn-small">Remind Later</button>
            </div>
          </div>
        ) : (
          <div style={{ color: 'var(--text-dim)', fontSize: '13px' }}>
            No pending upgrades
          </div>
        )}
      </div>
    </div>
  )
}
