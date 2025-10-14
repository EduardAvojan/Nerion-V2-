import React from 'react'

export default function MemorySnapshotPanel({ count, pinned, recent }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          🧠 MEMORY
        </div>
      </div>
      <div className="panel-content">
        <div className="stat">
          <div className="stat-label">Total Entries</div>
          <div className="stat-value">📌 {count}</div>
        </div>

        {pinned && pinned.length > 0 && (
          <div style={{ marginTop: '12px' }}>
            <div className="stat-label">Pinned:</div>
            <ul className="item-list">
              {pinned.slice(0, 3).map((item, i) => (
                <li key={i}>• {item.fact || item}</li>
              ))}
            </ul>
          </div>
        )}

        {recent && recent.length > 0 && (
          <div style={{ marginTop: '12px' }}>
            <div className="stat-label">Recent:</div>
            <ul className="item-list">
              {recent.slice(0, 2).map((item, i) => (
                <li key={i}>• {item.fact || item}</li>
              ))}
            </ul>
          </div>
        )}

        <button className="btn btn-small" style={{ marginTop: '12px', width: '100%' }}>
          View All →
        </button>
      </div>
    </div>
  )
}
