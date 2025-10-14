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

  // Example data - will be replaced with real WebSocket events
  useEffect(() => {
    // Simulate initial thoughts
    const demoThoughts = [
      { id: 1, text: 'Analyzing codebase structure...', timestamp: new Date(), status: 'completed' },
      { id: 2, text: 'Scanning for potential bugs...', timestamp: new Date(), status: 'completed' },
      { id: 3, text: 'Evaluating test coverage...', timestamp: new Date(), status: 'in_progress' }
    ]
    setThoughts(demoThoughts)

    const demoExplainability = [
      { reason: 'Pattern matching', confidence: 0.92, description: 'Detected common bug pattern in auth.py' },
      { reason: 'Context analysis', confidence: 0.87, description: 'Similar fix applied 3 times before' },
      { reason: 'Risk assessment', confidence: 0.95, description: 'Low risk - test coverage 98%' }
    ]
    setExplainability(demoExplainability)
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
