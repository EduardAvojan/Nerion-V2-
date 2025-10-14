import React, { useState, useEffect } from 'react'
import './TrainingDashboard.css'

export default function TrainingDashboard({ isOpen, onClose }) {
  const [activeTab, setActiveTab] = useState('overview') // 'overview', 'data', 'episodes', 'memory', 'logs'
  const [selectedDataSample, setSelectedDataSample] = useState(0)

  const [trainingData, setTrainingData] = useState({
    // Core GNN Metrics (from EpisodeResult)
    policyMode: 'curiosity',  // 'epsilon' or 'curiosity'
    surprise: 0.32,           // prediction_error / uncertainty (KEY METRIC)
    uncertainty: 0.15,        // model confidence
    epsilon: 0.12,            // exploration rate
    entropy: 2.14,            // decision diversity
    memorySize: 5420,         // experience replay buffer

    // Performance Metrics
    successRate: 0.87,        // outcome_is_success ratio
    totalTasks: 1243,         // completed tasks
    avgSurprise: 0.28,        // average surprise over time
    avgDuration: 4.2,         // avg seconds per task

    // Session Info
    currentEpisode: 1243,
    activeExperiments: 3,

    // Training Dataset (from learn() method)
    trainingDataset: [
      {
        id: 1,
        before_code: `def multiply(a, b):\n    return a * b\n\ndef calculate_total(x, y):\n    result = multiply(x, y)\n    return result`,
        after_code: `def multiply(a, b):\n    return a + b  # BUG: Changed to add\n\ndef calculate_total(x, y):\n    result = multiply(x, y)\n    return result`,
        test_code: `assert multiply(3, 4) == 12\nassert calculate_total(5, 6) == 30`,
        outcome: { passed: 0, failed: 2, errored: 0 },
        graph_stats: { nodes: 12, edges: 15, feature_dim: 768 },
        label: 1,  // 0 = pass, 1 = fail
        timestamp: '2025-10-13 14:20:15'
      },
      {
        id: 2,
        before_code: `def add(a, b):\n    return a + b\n\ndef process_values(x, y):\n    total = add(x, y)\n    return total * 2`,
        after_code: `def add(a, b):\n    return a + b\n\ndef process_values(x, y):\n    total = add(x, y)\n    return total * 2`,
        test_code: `assert add(2, 3) == 5\nassert process_values(4, 6) == 20`,
        outcome: { passed: 2, failed: 0, errored: 0 },
        graph_stats: { nodes: 10, edges: 12, feature_dim: 768 },
        label: 0,  // 0 = pass, 1 = fail
        timestamp: '2025-10-13 14:19:42'
      },
      {
        id: 3,
        before_code: `def divide(a, b):\n    return a / b\n\ndef safe_divide(x, y):\n    if y != 0:\n        return divide(x, y)\n    return None`,
        after_code: `def divide(a, b):\n    return a / b  # No zero check\n\ndef safe_divide(x, y):\n    return divide(x, y)  # BUG: Removed check`,
        test_code: `assert divide(10, 2) == 5\nassert safe_divide(10, 0) is None`,
        outcome: { passed: 1, failed: 0, errored: 1 },
        graph_stats: { nodes: 14, edges: 18, feature_dim: 768 },
        label: 1,  // 0 = pass, 1 = fail
        timestamp: '2025-10-13 14:18:33'
      }
    ],

    // Recent Episodes (EpisodeResult data)
    recentEpisodes: [
      {
        episode: 1243,
        action: 'refactor_function',
        predicted_pass: 0.82,
        predicted_fail: 0.18,
        outcome_is_success: true,
        surprise: 0.28,
        memory_size: 5420,
        policy_mode: 'curiosity',
        policy_epsilon: 0.12,
        policy_uncertainty: 0.15,
        policy_entropy: 2.14,
        policy_entropy_bonus: 0.05,
        policy_visit_count: 3,
        policy_epsilon_next: 0.118,
        action_tags: ['refactor', 'function'],
        action_metadata: { file: 'logic_v2.py', complexity: 'medium' },
        timestamp: '2025-10-13 14:32:15'
      },
      {
        episode: 1242,
        action: 'fix_bug',
        predicted_pass: 0.65,
        predicted_fail: 0.35,
        outcome_is_success: false,
        surprise: 0.45,
        memory_size: 5419,
        policy_mode: 'epsilon',
        policy_epsilon: 0.12,
        policy_uncertainty: 0.22,
        policy_entropy: 1.98,
        policy_entropy_bonus: 0.03,
        policy_visit_count: 1,
        policy_epsilon_next: 0.12,
        action_tags: ['fix', 'bug'],
        action_metadata: { file: 'utils.py', complexity: 'high' },
        timestamp: '2025-10-13 14:31:58'
      },
      {
        episode: 1241,
        action: 'add_test',
        predicted_pass: 0.91,
        predicted_fail: 0.09,
        outcome_is_success: true,
        surprise: 0.12,
        memory_size: 5418,
        policy_mode: 'curiosity',
        policy_epsilon: 0.12,
        policy_uncertainty: 0.08,
        policy_entropy: 2.35,
        policy_entropy_bonus: 0.06,
        policy_visit_count: 5,
        policy_epsilon_next: 0.115,
        action_tags: ['test', 'coverage'],
        action_metadata: { file: 'test_logic.py', complexity: 'low' },
        timestamp: '2025-10-13 14:31:42'
      }
    ],

    // Training Logs
    trainingLogs: [
      { time: '14:32:15', level: 'INFO', message: 'Episode 1243 complete - Success' },
      { time: '14:31:58', level: 'WARN', message: 'Episode 1242 complete - Failed (High surprise: 0.45)' },
      { time: '14:31:42', level: 'INFO', message: 'Episode 1241 complete - Success' },
      { time: '14:31:20', level: 'INFO', message: 'Memory buffer updated: 5420 experiences' }
    ]
  })

  useEffect(() => {
    if (isOpen) {
      // Fetch real training data from backend
      fetch('/api/training/status')
        .then(res => res.json())
        .then(data => {
          if (data.training_data) {
            setTrainingData(prev => ({ ...prev, ...data.training_data }))
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
            <h2>GNN Training</h2>
            <span className="header-subtitle">Real-time curiosity-driven learning</span>
          </div>
          <button className="close-btn" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M15 5L5 15M5 5L15 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="training-tabs">
          <button
            className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
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
            Training Logs
          </button>
        </div>

        {/* Content */}
        <div className="training-content">
          {activeTab === 'overview' && (
            <>
          {/* Surprise - Hero Metric */}
          <div className="hero-metric">
            <div className="hero-label">Current Surprise</div>
            <div className="hero-value" style={{ color: surpriseInfo.color }}>
              {trainingData.surprise.toFixed(3)}
            </div>
            <div className="hero-status" style={{ color: surpriseInfo.color }}>
              {surpriseInfo.label}
            </div>
            <div className="surprise-bar">
              <div
                className={`surprise-fill ${surpriseInfo.level}`}
                style={{ width: `${trainingData.surprise * 100}%` }}
              />
            </div>
            <div className="surprise-scale">
              <div className="scale-segment">
                <div className="scale-marker confident" />
                <span>0.0 - 0.2</span>
              </div>
              <div className="scale-segment">
                <div className="scale-marker learning" />
                <span>0.2 - 0.4</span>
              </div>
              <div className="scale-segment">
                <div className="scale-marker surprised" />
                <span>0.4 - 0.6</span>
              </div>
              <div className="scale-segment">
                <div className="scale-marker very-surprised" />
                <span>0.6+</span>
              </div>
            </div>
          </div>

          {/* Core Metrics Grid */}
          <div className="metrics-grid">
            {/* Policy Mode */}
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Policy Mode</span>
              </div>
              <div className="metric-value">{trainingData.policyMode}</div>
            </div>

            {/* Success Rate */}
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Success Rate</span>
              </div>
              <div className="metric-value">{successPercent}%</div>
              <div className="metric-bar">
                <div className="metric-bar-fill" style={{ width: `${successPercent}%` }} />
              </div>
            </div>

            {/* Uncertainty */}
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Uncertainty</span>
                <span className="metric-hint">Model confidence</span>
              </div>
              <div className="metric-value">{trainingData.uncertainty.toFixed(3)}</div>
            </div>

            {/* Epsilon */}
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Epsilon (ε)</span>
                <span className="metric-hint">Exploration rate</span>
              </div>
              <div className="metric-value">{trainingData.epsilon.toFixed(3)}</div>
            </div>

            {/* Entropy */}
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Entropy</span>
                <span className="metric-hint">Decision diversity</span>
              </div>
              <div className="metric-value">{trainingData.entropy.toFixed(2)}</div>
            </div>

            {/* Memory Size */}
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-label">Memory Size</span>
                <span className="metric-hint">Experience buffer</span>
              </div>
              <div className="metric-value">{trainingData.memorySize.toLocaleString()}</div>
            </div>
          </div>

          {/* Statistics */}
          <div className="stats-section">
            <div className="stats-card">
              <div className="stat-label">Total Tasks</div>
              <div className="stat-value">{trainingData.totalTasks.toLocaleString()}</div>
            </div>
            <div className="stats-card">
              <div className="stat-label">Current Episode</div>
              <div className="stat-value">{trainingData.currentEpisode.toLocaleString()}</div>
            </div>
            <div className="stats-card">
              <div className="stat-label">Avg Surprise</div>
              <div className="stat-value">{trainingData.avgSurprise.toFixed(3)}</div>
            </div>
            <div className="stats-card">
              <div className="stat-label">Avg Duration</div>
              <div className="stat-value">{trainingData.avgDuration.toFixed(1)}s</div>
            </div>
          </div>
            </>
          )}

          {activeTab === 'data' && (
            <div className="data-view">
              <div className="view-header">
                <h3>Training Data</h3>
                <span className="view-subtitle">Code graphs and test outcomes used for training</span>
              </div>

              {/* Dataset Statistics */}
              <div className="dataset-stats">
                <div className="dataset-stat-card">
                  <div className="dataset-stat-label">Total Samples</div>
                  <div className="dataset-stat-value">{trainingData.trainingDataset.length}</div>
                </div>
                <div className="dataset-stat-card">
                  <div className="dataset-stat-label">Pass Samples</div>
                  <div className="dataset-stat-value">
                    {trainingData.trainingDataset.filter(d => d.label === 0).length}
                  </div>
                  <div className="dataset-stat-hint">label = 0</div>
                </div>
                <div className="dataset-stat-card">
                  <div className="dataset-stat-label">Fail Samples</div>
                  <div className="dataset-stat-value">
                    {trainingData.trainingDataset.filter(d => d.label === 1).length}
                  </div>
                  <div className="dataset-stat-hint">label = 1</div>
                </div>
                <div className="dataset-stat-card">
                  <div className="dataset-stat-label">Avg Graph Size</div>
                  <div className="dataset-stat-value">
                    {Math.round(trainingData.trainingDataset.reduce((sum, d) => sum + d.graph_stats.nodes, 0) / trainingData.trainingDataset.length)}
                  </div>
                  <div className="dataset-stat-hint">nodes</div>
                </div>
              </div>

              {/* Sample List */}
              <div className="data-samples">
                <h4>Training Samples</h4>
                <div className="samples-list">
                  {trainingData.trainingDataset.map((sample, idx) => (
                    <div
                      key={sample.id}
                      className={`sample-item ${selectedDataSample === idx ? 'selected' : ''}`}
                      onClick={() => setSelectedDataSample(idx)}
                    >
                      <div className="sample-header">
                        <span className="sample-id">Sample #{sample.id}</span>
                        <span className={`sample-label ${sample.label === 0 ? 'pass' : 'fail'}`}>
                          {sample.label === 0 ? 'PASS' : 'FAIL'}
                        </span>
                      </div>
                      <div className="sample-meta">
                        <span>Nodes: {sample.graph_stats.nodes}</span>
                        <span>Edges: {sample.graph_stats.edges}</span>
                        <span>Tests: {sample.outcome.passed}✓ {sample.outcome.failed}✗ {sample.outcome.errored}⚠</span>
                      </div>
                      <div className="sample-time">{sample.timestamp}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Sample Detail View */}
              <div className="data-detail">
                <h4>Sample #{trainingData.trainingDataset[selectedDataSample].id} Detail</h4>

                <div className="detail-section">
                  <div className="detail-section-header">
                    <h5>Graph Statistics</h5>
                  </div>
                  <div className="graph-stats-grid">
                    <div className="graph-stat">
                      <span className="graph-stat-label">Nodes</span>
                      <span className="graph-stat-value">{trainingData.trainingDataset[selectedDataSample].graph_stats.nodes}</span>
                    </div>
                    <div className="graph-stat">
                      <span className="graph-stat-label">Edges</span>
                      <span className="graph-stat-value">{trainingData.trainingDataset[selectedDataSample].graph_stats.edges}</span>
                    </div>
                    <div className="graph-stat">
                      <span className="graph-stat-label">Feature Dim</span>
                      <span className="graph-stat-value">{trainingData.trainingDataset[selectedDataSample].graph_stats.feature_dim}</span>
                    </div>
                    <div className="graph-stat">
                      <span className="graph-stat-label">Label</span>
                      <span className={`graph-stat-value ${trainingData.trainingDataset[selectedDataSample].label === 0 ? 'success' : 'error'}`}>
                        {trainingData.trainingDataset[selectedDataSample].label}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="detail-section">
                  <div className="detail-section-header">
                    <h5>Test Outcome</h5>
                  </div>
                  <div className="outcome-grid">
                    <div className="outcome-stat">
                      <span className="outcome-stat-label">Passed</span>
                      <span className="outcome-stat-value success">{trainingData.trainingDataset[selectedDataSample].outcome.passed}</span>
                    </div>
                    <div className="outcome-stat">
                      <span className="outcome-stat-label">Failed</span>
                      <span className="outcome-stat-value error">{trainingData.trainingDataset[selectedDataSample].outcome.failed}</span>
                    </div>
                    <div className="outcome-stat">
                      <span className="outcome-stat-label">Errored</span>
                      <span className="outcome-stat-value warning">{trainingData.trainingDataset[selectedDataSample].outcome.errored}</span>
                    </div>
                  </div>
                </div>

                <div className="code-comparison">
                  <div className="code-panel">
                    <div className="code-panel-header">Before Code</div>
                    <pre className="code-content">{trainingData.trainingDataset[selectedDataSample].before_code}</pre>
                  </div>
                  <div className="code-panel">
                    <div className="code-panel-header">After Code</div>
                    <pre className="code-content">{trainingData.trainingDataset[selectedDataSample].after_code}</pre>
                  </div>
                </div>

                <div className="detail-section">
                  <div className="detail-section-header">
                    <h5>Test Code</h5>
                  </div>
                  <pre className="code-content">{trainingData.trainingDataset[selectedDataSample].test_code}</pre>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'episodes' && (
            <div className="episodes-view">
              <div className="view-header">
                <h3>Episode History</h3>
                <span className="view-subtitle">Complete EpisodeResult data for recent training episodes</span>
              </div>

              <div className="episodes-table-wrapper">
                <table className="episodes-table">
                  <thead>
                    <tr>
                      <th>Episode</th>
                      <th>Action</th>
                      <th>Pred Pass</th>
                      <th>Pred Fail</th>
                      <th>Outcome</th>
                      <th>Surprise</th>
                      <th>Mode</th>
                      <th>Epsilon (ε)</th>
                      <th>Uncertainty</th>
                      <th>Entropy</th>
                      <th>Visit Count</th>
                      <th>Tags</th>
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trainingData.recentEpisodes.map((ep, idx) => (
                      <tr key={idx}>
                        <td className="mono">{ep.episode}</td>
                        <td className="action-cell">{ep.action}</td>
                        <td className="mono">{ep.predicted_pass.toFixed(3)}</td>
                        <td className="mono">{ep.predicted_fail.toFixed(3)}</td>
                        <td>
                          <span className={`outcome-badge ${ep.outcome_is_success ? 'success' : 'fail'}`}>
                            {ep.outcome_is_success ? '✓' : '✗'}
                          </span>
                        </td>
                        <td className="mono surprise-cell" style={{
                          color: getSurpriseLevel(ep.surprise).color
                        }}>
                          {ep.surprise.toFixed(3)}
                        </td>
                        <td className="mode-cell">{ep.policy_mode}</td>
                        <td className="mono">{ep.policy_epsilon.toFixed(3)}</td>
                        <td className="mono">{ep.policy_uncertainty.toFixed(3)}</td>
                        <td className="mono">{ep.policy_entropy.toFixed(2)}</td>
                        <td className="mono">{ep.policy_visit_count}</td>
                        <td>
                          <div className="tags">
                            {ep.action_tags.map((tag, i) => (
                              <span key={i} className="tag">{tag}</span>
                            ))}
                          </div>
                        </td>
                        <td className="mono time-cell">{ep.timestamp}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="episode-details">
                <h4>Selected Episode Details</h4>
                <div className="details-grid">
                  <div className="detail-item">
                    <span className="detail-label">Entropy Bonus</span>
                    <span className="detail-value">{trainingData.recentEpisodes[0].policy_entropy_bonus.toFixed(3)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Next Epsilon</span>
                    <span className="detail-value">{trainingData.recentEpisodes[0].policy_epsilon_next.toFixed(3)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">File</span>
                    <span className="detail-value mono">{trainingData.recentEpisodes[0].action_metadata.file}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Complexity</span>
                    <span className="detail-value">{trainingData.recentEpisodes[0].action_metadata.complexity}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'memory' && (
            <div className="memory-view">
              <div className="view-header">
                <h3>Memory Explorer</h3>
                <span className="view-subtitle">Experience replay buffer - {trainingData.memorySize.toLocaleString()} stored experiences</span>
              </div>

              <div className="memory-stats">
                <div className="memory-stat-card">
                  <div className="memory-stat-label">Buffer Size</div>
                  <div className="memory-stat-value">{trainingData.memorySize.toLocaleString()}</div>
                </div>
                <div className="memory-stat-card">
                  <div className="memory-stat-label">High Surprise</div>
                  <div className="memory-stat-value">342</div>
                  <div className="memory-stat-hint">Experiences with surprise &gt; 0.4</div>
                </div>
                <div className="memory-stat-card">
                  <div className="memory-stat-label">Success Rate</div>
                  <div className="memory-stat-value">{successPercent}%</div>
                  <div className="memory-stat-hint">In memory buffer</div>
                </div>
                <div className="memory-stat-card">
                  <div className="memory-stat-label">Avg Age</div>
                  <div className="memory-stat-value">1.2h</div>
                  <div className="memory-stat-hint">Average experience age</div>
                </div>
              </div>

              <div className="memory-distribution">
                <h4>Surprise Distribution in Memory</h4>
                <div className="distribution-bars">
                  <div className="dist-bar">
                    <div className="dist-label">0.0 - 0.2 (Confident)</div>
                    <div className="dist-visual">
                      <div className="dist-fill confident" style={{ width: '45%' }}>45%</div>
                    </div>
                  </div>
                  <div className="dist-bar">
                    <div className="dist-label">0.2 - 0.4 (Learning)</div>
                    <div className="dist-visual">
                      <div className="dist-fill learning" style={{ width: '35%' }}>35%</div>
                    </div>
                  </div>
                  <div className="dist-bar">
                    <div className="dist-label">0.4 - 0.6 (Surprised)</div>
                    <div className="dist-visual">
                      <div className="dist-fill surprised" style={{ width: '15%' }}>15%</div>
                    </div>
                  </div>
                  <div className="dist-bar">
                    <div className="dist-label">0.6+ (Very Surprised)</div>
                    <div className="dist-visual">
                      <div className="dist-fill very-surprised" style={{ width: '5%' }}>5%</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="memory-actions">
                <button className="btn-action secondary">Export Memory Buffer</button>
                <button className="btn-action secondary">Clear Low-Priority Experiences</button>
                <button className="btn-action secondary">Analyze Patterns</button>
              </div>
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="logs-view">
              <div className="view-header">
                <h3>Training Logs</h3>
                <span className="view-subtitle">Real-time training output and events</span>
              </div>

              <div className="logs-container">
                {trainingData.trainingLogs.map((log, idx) => (
                  <div key={idx} className={`log-entry ${log.level.toLowerCase()}`}>
                    <span className="log-time">{log.time}</span>
                    <span className={`log-level ${log.level.toLowerCase()}`}>{log.level}</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))}
              </div>

              <div className="logs-actions">
                <button className="btn-action secondary">Clear Logs</button>
                <button className="btn-action secondary">Export Logs</button>
                <button className="btn-action secondary">Filter</button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="training-footer">
          <button className="btn-action secondary">Pause Training</button>
          <button className="btn-action primary">Start Training</button>
          <button className="btn-action secondary">Export Metrics</button>
        </div>
      </div>
    </>
  )
}
