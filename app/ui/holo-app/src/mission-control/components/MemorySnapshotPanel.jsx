import React, { useState, useEffect } from 'react'

export default function MemorySnapshotPanel() {
  const [memories, setMemories] = useState([])

  useEffect(() => {
    if (!window.nerion) {
      console.warn('[MemorySnapshot] window.nerion not available')
      return
    }

    const unsubscribe = window.nerion.onEvent((event) => {
      if (!event || !event.type) return

      console.log('[MemorySnapshot] Received event:', event.type, event)

      // Handle memory session events from backend
      if (event.type === 'memory_session') {
        const payload = event.payload || {}
        const items = payload.items || []
        console.log('[MemorySnapshot] Setting memories:', items)
        setMemories(items)
      }
    })

    // Request initial memory snapshot from backend
    if (window.nerion.send) {
      window.nerion.send('memory', { action: 'refresh' })
    }

    return () => {
      if (unsubscribe) unsubscribe()
    }
  }, [])

  const pinned = memories.filter(m => m.pinned)
  const recent = memories.filter(m => !m.pinned).slice(0, 2)

  const handleViewAll = () => {
    if (window.nerion && window.nerion.send) {
      window.nerion.send('memory', { action: 'drawer' })
    }
  }

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
          <div className="stat-value">📌 {memories.length}</div>
        </div>

        {pinned.length > 0 && (
          <div style={{ marginTop: '12px' }}>
            <div className="stat-label">Pinned:</div>
            <ul className="item-list">
              {pinned.slice(0, 3).map((item, i) => (
                <li key={item.id || i}>• {item.fact}</li>
              ))}
            </ul>
          </div>
        )}

        {recent.length > 0 && (
          <div style={{ marginTop: '12px' }}>
            <div className="stat-label">Recent:</div>
            <ul className="item-list">
              {recent.map((item, i) => (
                <li key={item.id || i}>• {item.fact}</li>
              ))}
            </ul>
          </div>
        )}

        <button
          className="btn btn-small"
          style={{ marginTop: '12px', width: '100%' }}
          onClick={handleViewAll}
        >
          View All →
        </button>
      </div>
    </div>
  )
}
