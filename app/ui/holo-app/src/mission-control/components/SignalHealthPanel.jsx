import React from 'react'

export default function SignalHealthPanel({ voice, network, learning, llm, coverage, errors }) {
  const getStatusColor = (status) => {
    if (status === 'online' || status === 'active') return 'status-online'
    if (status === 'warning') return 'status-warning'
    return 'status-error'
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          📡 SIGNAL HEALTH
        </div>
      </div>
      <div className="panel-content">
        <div className="stat">
          <div className="stat-label">Voice System</div>
          <div className="stat-value">
            <span className={`status-indicator ${getStatusColor(voice)}`}></span>
            {voice}
          </div>
        </div>

        <div className="stat">
          <div className="stat-label">Network Gate</div>
          <div className="stat-value">
            <span className={`status-indicator ${getStatusColor(network)}`}></span>
            {network}
          </div>
        </div>

        <div className="stat">
          <div className="stat-label">Learning System</div>
          <div className="stat-value">
            <span className={`status-indicator ${getStatusColor(learning)}`}></span>
            {learning}
          </div>
        </div>

        <div className="stat">
          <div className="stat-label">LLM Provider</div>
          <div className="stat-value">
            <span className={`status-indicator status-online`}></span>
            {llm}
          </div>
        </div>

        <div className="stat">
          <div className="stat-label">Coverage / Errors</div>
          <div className="stat-value">
            {coverage}% / {errors}
          </div>
        </div>
      </div>
    </div>
  )
}
