import React, { useState, useEffect, useRef } from 'react'
import './GenesisView.css'

export default function GenesisView() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Hello! I\'m Genesis, Nerion\'s self-evolving brain. I can help you improve the codebase, run experiments, and apply autonomous modifications. What would you like me to work on?' }
  ])
  const [inputValue, setInputValue] = useState('')
  const messagesEndRef = useRef(null)

  const [phase, setPhase] = useState('Research')
  const [progress, setProgress] = useState(45)
  const [experiments, setExperiments] = useState([
    { id: 1, name: 'Security hardening trial #127', status: 'running', progress: 78 },
    { id: 2, name: 'Performance optimization', status: 'analyzing', progress: 100 },
    { id: 3, name: 'Memory efficiency test', status: 'queued', progress: 0 }
  ])
  const [pendingApprovals, setPendingApprovals] = useState([
    { id: 1, file: 'llm_router.py', action: 'Apply caching strategy', impact: 'medium', risk: 'low', type: 'optimization' },
    { id: 2, file: 'scoring.py', action: 'Refactor scoring patterns', impact: 'low', risk: 'low', type: 'refactor' },
    { id: 3, file: 'auth.py', action: 'Add rate limiting', impact: 'high', risk: 'medium', type: 'security' }
  ])
  const [learningTimeline, setLearningTimeline] = useState([])
  const [stats, setStats] = useState({
    activeExperiments: 3,
    pendingMods: 3,
    completedToday: 5
  })
  const [governorState, setGovernorState] = useState({
    status: 'ACTIVE',
    safety: 'ENABLED',
    confidence: 94
  })

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Listen for REAL Nerion responses
  useEffect(() => {
    if (!window.nerion?.onEvent) return

    const unsubscribe = window.nerion.onEvent((data) => {
      // Listen for chat_turn events (Nerion's response format)
      if (data.type === 'chat_turn' && data.payload?.role === 'assistant') {
        const text = data.payload.text

        // Filter out system prompts (upgrade messages, etc.)
        const isSystemPrompt = text.includes('Self Learning Upgrade Available') ||
                               text.includes('Say \'upgrade now\'') ||
                               text.includes('remind me later')

        if (text && !isSystemPrompt) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            text: text
          }])
        }
      }

      // Listen for learning_timeline events
      if (data.type === 'learning_timeline') {
        const payload = data.payload || {}
        const events = payload.events || []

        console.log('[Genesis] Received learning_timeline events:', events)

        // Transform backend events to display format
        const formatted = events.map(evt => {
          // Format the event text based on available fields
          let eventText = evt.summary || evt.key || 'Learning event'

          // If we have both key and value, create a readable message
          if (evt.key && evt.value) {
            const keyFormatted = evt.key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
            eventText = `${keyFormatted}: ${evt.value}`
          }

          return {
            id: evt.id,
            time: evt.timestamp || 'now',
            event: eventText,
            success: true,
            scope: evt.scope,
            confidence: evt.confidence
          }
        })

        console.log('[Genesis] Setting learning timeline:', formatted)
        setLearningTimeline(formatted)
      }
    })

    // Request initial learning timeline from backend
    if (window.nerion.send) {
      window.nerion.send('learning', { action: 'refresh' })
    }

    return () => {
      if (typeof unsubscribe === 'function') {
        unsubscribe()
      }
    }
  }, [])

  // Simulate phase progression
  useEffect(() => {
    const phases = ['Research', 'Planning', 'Implementation', 'Validation']
    const interval = setInterval(() => {
      setPhase(prev => {
        const currentIndex = phases.indexOf(prev)
        return phases[(currentIndex + 1) % phases.length]
      })
      setProgress(prev => (prev + 5) % 100)
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const handleSendMessage = (e) => {
    e.preventDefault()
    if (!inputValue.trim()) return

    // Add user message
    setMessages(prev => [...prev, { role: 'user', text: inputValue }])

    // Send to REAL Nerion via IPC
    console.log('[Genesis] window.nerion available:', !!window.nerion)
    console.log('[Genesis] window.nerion:', window.nerion)

    try {
      if (!window.nerion || !window.nerion.send) {
        throw new Error('window.nerion.send not available')
      }
      console.log('[Genesis] Sending message to Nerion:', inputValue)
      window.nerion.send('chat', { text: inputValue })
      console.log('[Genesis] Message sent successfully')
    } catch (error) {
      console.error('[Genesis] Failed to send message:', error)
      // Fallback to fake response if IPC not available
      setTimeout(() => {
        setMessages(prev => [...prev, {
          role: 'assistant',
          text: `[Offline Mode] I'm analyzing your request: "${inputValue}". Starting experiments to validate the approach...`
        }])
      }, 1000)
    }

    setInputValue('')
  }

  const handleApprove = (id) => {
    setPendingApprovals(prev => prev.filter(item => item.id !== id))
    setMessages(prev => [...prev, {
      role: 'assistant',
      text: `✓ Modification approved! Applying changes...`
    }])
  }

  const handleReject = (id) => {
    setPendingApprovals(prev => prev.filter(item => item.id !== id))
    setMessages(prev => [...prev, {
      role: 'assistant',
      text: `✗ Modification rejected. Exploring alternative approaches...`
    }])
  }

  const getPhaseIcon = (phaseName) => {
    const icons = {
      'Research': '⚡',
      'Planning': '⏳',
      'Implementation': '⚙️',
      'Validation': '✓'
    }
    return phase === phaseName ? icons[phaseName] : '⏸️'
  }

  const getStatusColor = (status) => {
    const colors = {
      running: 'var(--accent)',
      analyzing: 'var(--warning)',
      queued: 'var(--text-dim)',
      in_progress: 'var(--accent)',
      pending: 'var(--text-secondary)'
    }
    return colors[status] || 'var(--text-secondary)'
  }

  return (
    <div className="genesis-view">
      <div className="genesis-split">
        {/* LEFT: Conversation */}
        <div className="genesis-left">
          <div className="conversation-header">
            <h3>💬 Conversation with Genesis</h3>
            <div className="phase-indicator">
              <span className="phase-label">{phase}</span>
              <div className="phase-progress-mini">
                <div className="progress-fill" style={{ width: `${progress}%` }}></div>
              </div>
              <span className="phase-percent">{progress}%</span>
            </div>
          </div>

          <div className="conversation-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message message-${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🧬'}
                </div>
                <div className="message-content">
                  <div className="message-text">{msg.text}</div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form className="conversation-input" onSubmit={handleSendMessage}>
            <input
              type="text"
              placeholder="Instruct Genesis: 'Optimize scoring algorithm' or 'Fix memory leaks'..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              className="input-field"
            />
            <button type="submit" className="btn btn-send">
              Send
            </button>
          </form>
        </div>

        {/* RIGHT: Brain Status & Controls */}
        <div className="genesis-right">
          {/* Brain Status */}
          <div className="status-card">
            <h4>🧬 Brain Status</h4>
            <div className="status-grid">
              <div className="status-item">
                <span className="status-icon">⚡</span>
                <div className="status-info">
                  <div className="status-label">Active Experiments</div>
                  <div className="status-value">{stats.activeExperiments}</div>
                </div>
              </div>
              <div className="status-item">
                <span className="status-icon">⚙️</span>
                <div className="status-info">
                  <div className="status-label">Pending Approvals</div>
                  <div className="status-value">{stats.pendingMods}</div>
                </div>
              </div>
              <div className="status-item">
                <span className="status-icon">✓</span>
                <div className="status-info">
                  <div className="status-label">Completed Today</div>
                  <div className="status-value">{stats.completedToday}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Active Experiments */}
          <div className="status-card">
            <h4>🔬 Active Experiments</h4>
            <div className="experiments-compact">
              {experiments.map(exp => (
                <div key={exp.id} className="experiment-compact">
                  <div className="experiment-name-compact">{exp.name}</div>
                  {exp.status === 'running' && (
                    <div className="experiment-progress-compact">
                      <div className="progress-bar small">
                        <div className="progress-fill" style={{ width: `${exp.progress}%` }}></div>
                      </div>
                      <span className="progress-text">{exp.progress}%</span>
                    </div>
                  )}
                  {exp.status !== 'running' && (
                    <span className="experiment-status-compact" style={{ color: getStatusColor(exp.status) }}>
                      {exp.status}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Pending Approvals */}
          {pendingApprovals.length > 0 && (
            <div className="status-card approvals-card">
              <h4>📋 Pending Approvals</h4>
              <div className="approvals-list">
                {pendingApprovals.map(approval => (
                  <div key={approval.id} className="approval-item">
                    <div className="approval-header">
                      <span className="approval-action">{approval.action}</span>
                      <div className="approval-badges">
                        <span className={`approval-impact impact-${approval.impact}`}>
                          {approval.impact}
                        </span>
                        <span className={`approval-risk risk-${approval.risk}`}>
                          risk: {approval.risk}
                        </span>
                      </div>
                    </div>
                    <div className="approval-file">{approval.file}</div>
                    <div className="approval-actions">
                      <button
                        className="btn-approve"
                        onClick={() => handleApprove(approval.id)}
                      >
                        ✓ Approve
                      </button>
                      <button
                        className="btn-reject"
                        onClick={() => handleReject(approval.id)}
                      >
                        ✗ Reject
                      </button>
                      <button className="btn-view">
                        View Diff
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Learning Timeline */}
          <div className="status-card">
            <h4>📚 Learning Timeline</h4>
            <div className="learning-timeline">
              {learningTimeline.length > 0 ? (
                learningTimeline.map(event => (
                  <div key={event.id} className="timeline-event">
                    <div className="timeline-marker">
                      {event.success ? '✓' : '✕'}
                    </div>
                    <div className="timeline-content">
                      <div className="timeline-text">{event.event}</div>
                      <div className="timeline-time">{event.time}</div>
                    </div>
                  </div>
                ))
              ) : (
                <div style={{ color: 'var(--text-dim)', fontSize: '13px', padding: '8px' }}>
                  No learning events yet
                </div>
              )}
            </div>
          </div>

          {/* Governor */}
          <div className="status-card governor-card">
            <h4>🎯 Governor</h4>
            <div className="governor-status">
              <div className="governor-stat">
                Status: <strong style={{ color: 'var(--success)' }}>{governorState.status}</strong>
              </div>
              <div className="governor-stat">
                Safety: <strong style={{ color: 'var(--success)' }}>{governorState.safety}</strong>
              </div>
              <div className="governor-stat">
                Confidence: <strong style={{ color: 'var(--accent)' }}>{governorState.confidence}%</strong>
              </div>
            </div>
            <div className="governor-controls-compact">
              <button className="btn btn-small">Pause</button>
              <button className="btn btn-small">Override</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
