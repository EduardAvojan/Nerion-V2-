import React, { useState, useEffect } from 'react'

export default function LearningTimelinePanel() {
  const [timeline, setTimeline] = useState([])

  useEffect(() => {
    fetch('/api/learning/timeline')
      .then(res => res.json())
      .then(data => setTimeline(data.events || []))
      .catch(err => console.error('Failed to fetch learning timeline:', err))
  }, [])

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
              <li key={i}>
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
        <button className="btn btn-small" style={{ marginTop: '12px', width: '100%' }}>
          Show Full History →
        </button>
      </div>
    </div>
  )
}
