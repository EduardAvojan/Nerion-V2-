import React, { useState, useEffect, useRef } from 'react'
import './TrainingDashboard.css'

export default function TrainingDashboard({ isOpen, onClose, liveStatus }) {
  const [activeTab, setActiveTab] = useState('overview') // 'overview', 'data', 'episodes', 'memory', 'logs'
  const [selectedDataSample, setSelectedDataSample] = useState(0)
  const [scanHistory, setScanHistory] = useState([])
  const historyRef = useRef([])

  // Initialize with ZERO/Empty data (no stale mocks)
  const [trainingData, setTrainingData] = useState({
    // Core GNN Metrics
    policyMode: 'curiosity',
    surprise: 0.0,
    uncertainty: 0.0,
    epsilon: 0.0,
    entropy: 0.0,
    memorySize: 0,

    // Performance Metrics
    successRate: 0.0,
    totalTasks: 0,
    avgSurprise: 0.0,
    avgDuration: 0.0,

    // Session Info
    currentEpisode: 0,
    activeExperiments: 0,

    // Arrays
    trainingDataset: [],
    recentEpisodes: [],
    trainingLogs: []
  })

  const [isTraining, setIsTraining] = useState(false)

  // Track live status history
  useEffect(() => {
    if (liveStatus && liveStatus.status !== 'idle') {
      // Avoid duplicates if the same status object is passed
      const last = historyRef.current[0]
      if (!last || last.lastUpdate !== liveStatus.lastUpdate || last.file !== liveStatus.file || last.status !== liveStatus.status) {
        const newEntry = { ...liveStatus, timestamp: new Date().toLocaleTimeString() }
        const newHistory = [newEntry, ...historyRef.current].slice(0, 50) // Keep last 50
        historyRef.current = newHistory
        setScanHistory(newHistory)
      }
    }
  }, [liveStatus])

  const handleStartTraining = async () => {
    try {
      await fetch('/api/control/start_autonomy', { method: 'POST' })
      setIsTraining(true)
    } catch (err) {
      console.error('Failed to start autonomy:', err)
    }
  }

  const handleStopTraining = async () => {
    try {
      await fetch('/api/control/stop_autonomy', { method: 'POST' })
      setIsTraining(false)
    } catch (err) {
      console.error('Failed to stop autonomy:', err)
    }
  }

  useEffect(() => {
    if (isOpen) {
      // Fetch real training data from backend
      fetch('/api/training/status')
        .then(res => res.json())
        .then(data => {
          if (data.training_data) {
            setTrainingData(prev => ({ ...prev, ...data.training_data }))
          }
          if (data.autonomous_testing !== undefined) {
            setIsTraining(data.autonomous_testing)
          }
        })
        .catch(err => console.error('Failed to fetch training data:', err))
    }
  }, [isOpen])

  if (!isOpen) return null

  const getSurpriseLevel = (score) => {
    if (score < 0.2) return { level: 'confident', label: 'Confident', color: 'var(--success)' }
    if (score < 0.4) return { level: 'learning', label: 'Learning', color: 'var(--accent)' }
    if (score < 0.6) return { level: 'surprised', label: 'Surprised', color: 'var(--warning)' }
    return { level: 'very-surprised', label: 'Very Surprised', color: 'var(--error)' }
  }

  const surpriseInfo = getSurpriseLevel(trainingData.surprise)
  const successPercent = Math.round(trainingData.successRate * 100)

  return (
    <>
      <div className="training-overlay" onClick={onClose} />
      <div className="training-dashboard">
        {/* Header */}
        <div className="training-header">
          <div className="header-left">
            <h2>Autonomous Defense System</h2>
            <span className="header-subtitle">Real-time Immune System Operations</span>
          </div>
          <button className="close-btn" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M15 5L5 15M5 5L15 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="training-tabs">
          <button
            className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Live Operations
          </button>
          <button
            className={`tab ${activeTab === 'data' ? 'active' : ''}`}
            onClick={() => setActiveTab('data')}
          >
            Training Data
          </button>
          <button
            className={`tab ${activeTab === 'episodes' ? 'active' : ''}`}
            onClick={() => setActiveTab('episodes')}
          >
            Episode History
          </button>
          <button
            className={`tab ${activeTab === 'memory' ? 'active' : ''}`}
            onClick={() => setActiveTab('memory')}
          >
            Memory Explorer
          </button>
          <button
            className={`tab ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
          >
            System Logs
          </button>
        </div>

        {/* Content */}
        <div className="training-content">
          {activeTab === 'overview' && (
            <div className="overview-container">
              {/* Hero Status Section */}
              <div className={`hero-status-card ${isTraining ? 'active' : 'idle'}`}>
                <div className="hero-status-header">
                  <span className="status-dot"></span>
                  <h3>{isTraining ? 'SYSTEM ACTIVE' : 'SYSTEM IDLE'}</h3>
                </div>

                {isTraining ? (
                  <div className="hero-live-activity">
                    {liveStatus && liveStatus.status === 'running' ? (
                      <div className="activity-row running">
                        <div className="spinner"></div>
                        <div className="activity-details">
                          <span className="activity-action">SCANNING & ANALYZING</span>
                          <span className="activity-target">{liveStatus.file}</span>
                        </div>
                      </div>
                    ) : liveStatus && (liveStatus.status === 'passed' || liveStatus.status === 'failed') ? (
                      <div className={`activity-row result ${liveStatus.status}`}>
                        <span className="result-icon">{liveStatus.status === 'passed' ? '✅' : '❌'}</span>
                        <div className="activity-details">
                          <span className="activity-action">TEST VERIFICATION COMPLETE</span>
                          <span className="activity-target">{liveStatus.file}</span>
                          <span className="activity-outcome">{liveStatus.status.toUpperCase()}</span>
                        </div>
                      </div>
                    ) : (
                      <div className="activity-row waiting">
                        <span className="activity-text">Waiting for next scan cycle...</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="hero-idle-message">
                    Autonomous testing is paused. Click "Start" to resume operations.
                  </div>
                )}
              </div>

              <div className="dashboard-split">
                {/* Left: Metrics */}
                <div className="metrics-column">
                  <div className="metric-tile">
                    <label>Success Rate</label>
                    <div className="value">{successPercent}%</div>
                    <div className="sub">Global Fix Rate</div>
                  </div>
                  <div className="metric-tile">
                    <label>Total Fixes</label>
                    <div className="value">{trainingData.totalTasks}</div>
                    <div className="sub">Applied & Verified</div>
                  </div>
                  <div className="metric-tile">
                    <label>Surprise</label>
                    <div className="value" style={{ color: surpriseInfo.color }}>{trainingData.surprise.toFixed(3)}</div>
                    <div className="sub">{surpriseInfo.label}</div>
                  </div>
                  <div className="metric-tile">
                    <label>Memory</label>
                    <div className="value">{trainingData.memorySize}</div>
                    <div className="sub">Experiences</div>
                  </div>
                </div>

                {/* Right: Live Feed */}
                <div className="feed-column">
                  <h3>Recent Activity Feed</h3>
                  <div className="activity-feed-list">
                    {scanHistory.length === 0 ? (
                      <div className="empty-feed">No recent activity recorded this session.</div>
                    ) : (
                      scanHistory.map((entry, idx) => (
                        <div key={idx} className={`feed-item ${entry.status}`}>
                          <span className="feed-time">{entry.timestamp}</span>
                          <span className="feed-status">
                            {entry.status === 'running' ? '🔄 SCAN' :
                              entry.status === 'passed' ? '✅ PASS' : '❌ FAIL'}
                          </span>
                          <span className="feed-file" title={entry.file}>
                            {entry.file.split('/').pop()}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'data' && (
            <div className="data-view">
              <div className="view-header">
                <h3>Training Data</h3>
                <span className="view-subtitle">Code graphs and test outcomes used for training</span>
              </div>
              {trainingData.trainingDataset.length === 0 ? (
                <div className="empty-state">No training data available yet.</div>
              ) : (
                // Existing data view logic...
                <div className="dataset-stats">
                  {/* ... (Keep existing stats logic if needed, or simplify) ... */}
                  <div className="dataset-stat-card">
                    <div className="dataset-stat-label">Total Samples</div>
                    <div className="dataset-stat-value">{trainingData.trainingDataset.length}</div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'episodes' && (
            <div className="episodes-view">
              <div className="view-header">
                <h3>Episode History</h3>
              </div>
              {trainingData.recentEpisodes.length === 0 ? (
                <div className="empty-state">No episodes recorded yet.</div>
              ) : (
                <div className="episodes-table-wrapper">
                  {/* Keep existing table logic */}
                  <table className="episodes-table">
                    <thead>
                      <tr>
                        <th>Episode</th>
                        <th>Action</th>
                        <th>Outcome</th>
                        <th>Surprise</th>
                      </tr>
                    </thead>
                    <tbody>
                      {trainingData.recentEpisodes.map((ep, idx) => (
                        <tr key={idx}>
                          <td>{ep.episode}</td>
                          <td>{ep.action}</td>
                          <td>{ep.outcome_is_success ? '✓' : '✗'}</td>
                          <td>{ep.surprise.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {activeTab === 'memory' && (
            <div className="memory-view">
              <h3>Memory Explorer</h3>
              <div className="memory-stats">
                <div className="memory-stat-card">
                  <div className="memory-stat-label">Buffer Size</div>
                  <div className="memory-stat-value">{trainingData.memorySize}</div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="logs-view">
              <div className="view-header">
                <h3>System Logs</h3>
              </div>
              <div className="logs-container">
                {trainingData.trainingLogs.length === 0 ? (
                  <div className="empty-state">No logs available.</div>
                ) : (
                  trainingData.trainingLogs.map((log, idx) => (
                    <div key={idx} className={`log-entry ${log.level.toLowerCase()}`}>
                      <span className="log-time">{log.time}</span>
                      <span className={`log-level ${log.level.toLowerCase()}`}>{log.level}</span>
                      <span className="log-message">{log.message}</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="training-footer">
          {isTraining ? (
            <button className="btn-action secondary" onClick={handleStopTraining}>
              Pause Autonomous Testing
            </button>
          ) : (
            <button className="btn-action primary" onClick={handleStartTraining}>
              Start Autonomous Testing
            </button>
          )}
          <button className="btn-action secondary">Export Metrics</button>
        </div>
      </div>
    </>
  )
}
