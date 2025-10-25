import React, { useState, useEffect, useRef } from 'react'
import './ThoughtProcessPanel.css'

export default function ThoughtProcessPanel() {
  const [thoughts, setThoughts] = useState([])
  const [showDetails, setShowDetails] = useState(false)
  const [explainability, setExplainability] = useState([])
  const [patchReview, setPatchReview] = useState(null)
  const thoughtsEndRef = useRef(null)

  // Auto-scroll to latest thought
  useEffect(() => {
    thoughtsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [thoughts])

  // Listen to real-time thought process events from backend
  useEffect(() => {
    if (!window.nerion) {
      console.warn('[ThoughtProcess] window.nerion not available')
      return
    }

    const unsubscribe = window.nerion.onEvent((event) => {
      if (!event || !event.type) return

      // Handle thought step events
      if (event.type === 'thought_step') {
        const payload = event.payload || {}
        const thoughtId = payload.id
        const title = payload.title || ''
        const detail = payload.detail || ''
        const status = payload.status || 'pending' // pending, active, complete, failed
        const timestamp = new Date(payload.ts || Date.now())

        setThoughts(prev => {
          // Check if thought already exists
          const existingIndex = prev.findIndex(t => t.id === thoughtId)

          if (existingIndex >= 0) {
            // Update existing thought
            const updated = [...prev]
            updated[existingIndex] = {
              ...updated[existingIndex],
              text: title,
              detail: detail,
              status: status === 'active' ? 'in_progress' : status,
              timestamp: timestamp,
              duration: payload.duration
            }
            return updated
          } else {
            // Add new thought
            return [...prev, {
              id: thoughtId,
              text: title,
              detail: detail,
              status: status === 'active' ? 'in_progress' : status,
              timestamp: timestamp,
              duration: payload.duration
            }]
          }
        })
      }

      // Handle confidence/explainability events
      if (event.type === 'confidence') {
        const payload = event.payload || {}
        const score = payload.score || payload.value || 0
        const drivers = payload.drivers || []

        if (drivers.length > 0) {
          setExplainability(drivers.map(driver => ({
            reason: driver,
            confidence: score,
            description: driver
          })))
        }
      }

      // Handle state events to reset thoughts on new turns
      if (event.type === 'state' && event.payload?.reset_thoughts) {
        setThoughts([])
        setExplainability([])
      }
    })

    return () => {
      if (unsubscribe) unsubscribe()
    }
  }, [])

  const handleClearThoughts = () => {
    setThoughts([])
    setExplainability([])
  }

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  return (
    <aside className="thought-panel">
      <div className="thought-panel__box">
      <header className="thought-panel__header">
        <span className="thought-panel__title">💭 Thought Process</span>
        <div className="thought-panel__actions">
          <button
            className={`thought-panel__action ${showDetails ? 'active' : ''}`}
            onClick={() => setShowDetails(!showDetails)}
            title="Toggle details"
          >
            {showDetails ? 'Hide' : 'Show'} Details
          </button>
          <button
            className="thought-panel__action"
            onClick={handleClearThoughts}
            title="Clear thoughts"
          >
            Clear
          </button>
        </div>
      </header>

      <div className="thought-ribbon">
        {thoughts.length === 0 ? (
          <div className="thought-ribbon__empty">
            No active reasoning. System idle.
          </div>
        ) : (
          <ol className="thought-ribbon__list">
            {thoughts.map((thought) => (
              <li
                key={thought.id}
                className={`thought-item thought-item--${thought.status}`}
              >
                <div className="thought-item__indicator">
                  {thought.status === 'in_progress' && (
                    <div className="thought-item__spinner"></div>
                  )}
                  {thought.status === 'completed' && (
                    <span className="thought-item__check">✓</span>
                  )}
                  {thought.status === 'error' && (
                    <span className="thought-item__error">✕</span>
                  )}
                </div>
                <div className="thought-item__content">
                  <div className="thought-item__text">{thought.text}</div>
                  {showDetails && (
                    <div className="thought-item__meta">
                      <span className="thought-item__time">{formatTime(thought.timestamp)}</span>
                      {thought.duration && (
                        <span className="thought-item__duration">{thought.duration}ms</span>
                      )}
                    </div>
                  )}
                </div>
              </li>
            ))}
            <div ref={thoughtsEndRef} />
          </ol>
        )}
      </div>

      {explainability.length > 0 && (
        <section className="explainability">
          <h3 className="explainability__title">🔍 Explainability</h3>
          <ul className="explainability__list">
            {explainability.map((item, i) => (
              <li key={i} className="explain-item">
                <div className="explain-item__header">
                  <span className="explain-item__reason">{item.reason}</span>
                  <span className="explain-item__confidence">
                    {(item.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                {showDetails && (
                  <p className="explain-item__description">{item.description}</p>
                )}
                <div className="explain-item__bar">
                  <div
                    className="explain-item__fill"
                    style={{ width: `${item.confidence * 100}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {patchReview && (
        <section className="patch-review">
          <header className="patch-review__header">
            <span className="patch-review__title">🔧 Patch Review</span>
            <div className="patch-review__actions">
              <button className="patch-review__action patch-review__action--approve">
                Approve
              </button>
              <button className="patch-review__action">
                Reject
              </button>
            </div>
          </header>
          <div className="patch-review__content">
            <div className="patch-review__summary">
              <strong>File:</strong> {patchReview.file}
            </div>
            <div className="patch-review__summary">
              <strong>Type:</strong> {patchReview.type}
            </div>
            <div className="patch-review__summary">
              <strong>Impact:</strong>
              <span className={`patch-review__impact patch-review__impact--${patchReview.impact}`}>
                {patchReview.impact}
              </span>
            </div>
            <div className="patch-review__summary">
              <strong>Risk:</strong>
              <span className={`patch-review__risk patch-review__risk--${patchReview.risk}`}>
                {patchReview.risk}
              </span>
            </div>
          </div>
        </section>
      )}

      <div className="thought-panel__status">
        <div className="status-indicator">
          <span className="status-indicator__dot status-indicator__dot--active"></span>
          <span className="status-indicator__label">Reasoning Active</span>
        </div>
        <div className="thought-panel__stats">
          {thoughts.length} thoughts • {explainability.length} factors
        </div>
      </div>
      </div>
    </aside>
  )
}
