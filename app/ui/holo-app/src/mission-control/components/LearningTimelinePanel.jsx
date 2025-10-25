import React, { useState, useEffect } from 'react'

export default function LearningTimelinePanel() {
  const [timeline, setTimeline] = useState([])

  useEffect(() => {
    if (!window.nerion) {
      console.warn('[LearningTimeline] window.nerion not available')
      return
    }

    const unsubscribe = window.nerion.onEvent((event) => {
      if (!event || !event.type) return

      console.log('[LearningTimeline] Received event:', event.type, event)

      // Handle learning timeline events from backend
      if (event.type === 'learning_timeline') {
        const payload = event.payload || {}
        const events = payload.events || []

        console.log('[LearningTimeline] Got events:', events)

        // Transform backend events to display format
        const formatted = events.map(evt => ({
          id: evt.id,
          time: evt.timestamp || 'now',
          event: evt.summary || evt.key || 'Learning event',
          scope: evt.scope,
          confidence: evt.confidence,
          details: evt.details
        }))

        console.log('[LearningTimeline] Setting timeline:', formatted)
        setTimeline(formatted)
      }
    })

    // Request initial learning timeline from backend
    if (window.nerion.send) {
      window.nerion.send('learning', { action: 'refresh' })
    }

    return () => {
      if (unsubscribe) unsubscribe()
    }
  }, [])

  const handleShowFullHistory = () => {
    if (window.nerion && window.nerion.send) {
      window.nerion.send('learning', { action: 'drawer' })
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          📚 LEARNING TIMELINE
        </div>
      </div>
      <div className="panel-content">
        {timeline.length > 0 ? (
          <ul className="item-list">
            {timeline.slice(0, 5).map((event, i) => (
              <li key={event.id || i}>
                <span style={{ color: 'var(--text-dim)', fontSize: '11px' }}>
                  {event.time}
                </span>
                {' '}
                <span style={{ color: 'var(--success)' }}>✓</span>
                {' '}
                {event.event}
              </li>
            ))}
          </ul>
        ) : (
          <div style={{ color: 'var(--text-dim)', fontSize: '13px' }}>
            No learning events yet
          </div>
        )}
        <button
          className="btn btn-small"
          style={{ marginTop: '12px', width: '100%' }}
          onClick={handleShowFullHistory}
        >
          Show Full History →
        </button>
      </div>
    </div>
  )
}
