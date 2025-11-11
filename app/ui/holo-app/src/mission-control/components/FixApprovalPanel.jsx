import React, { useState, useEffect } from 'react'
import './FixApprovalPanel.css'

function FixApprovalPanel() {
  const [pendingFixes, setPendingFixes] = useState([])
  const [selectedFix, setSelectedFix] = useState(null)

  useEffect(() => {
    // Check if daemon IPC is available
    if (!window.daemon) {
      console.error('[FixApprovalPanel] window.daemon not available - IPC bridge not loaded')
      return
    }

    console.log('[FixApprovalPanel] Setting up daemon IPC listeners')

    // Listen for fix proposals from daemon
    const unsubProposal = window.daemon.onFixProposal((data) => {
      console.log('[FixApprovalPanel] Received fix proposal:', data)
      setPendingFixes(prev => [...prev, data])
    })

    // Listen for pending fixes list
    const unsubPending = window.daemon.onPendingFixes((data) => {
      console.log('[FixApprovalPanel] Received pending fixes list:', data)
      setPendingFixes(data || [])
    })

    // Listen for approval/rejection responses
    const unsubResponse = window.daemon.onFixResponse((message) => {
      console.log('[FixApprovalPanel] Received fix response:', message)
      if (message.type === 'fix_approved' || message.type === 'fix_rejected') {
        // Remove approved/rejected fix from list
        setPendingFixes(prev => prev.filter(fix => fix.fix_id !== message.data.fix_id))
        if (selectedFix && selectedFix.fix_id === message.data.fix_id) {
          setSelectedFix(null)
        }
      }
    })

    // Request pending fixes on mount
    console.log('[FixApprovalPanel] Requesting pending fixes from daemon')
    window.daemon.getPendingFixes()

    // Cleanup subscriptions on unmount
    return () => {
      console.log('[FixApprovalPanel] Cleaning up daemon IPC listeners')
      unsubProposal()
      unsubPending()
      unsubResponse()
    }
  }, [selectedFix])

  const approveFix = (fix) => {
    if (!window.daemon) {
      console.error('[FixApprovalPanel] Cannot approve fix - window.daemon not available')
      return
    }

    console.log('[FixApprovalPanel] Approving fix:', fix.fix_id)
    window.daemon.send({
      type: 'approve_fix',
      fix_id: fix.fix_id
    })
  }

  const rejectFix = (fix) => {
    if (!window.daemon) {
      console.error('[FixApprovalPanel] Cannot reject fix - window.daemon not available')
      return
    }

    console.log('[FixApprovalPanel] Rejecting fix:', fix.fix_id)
    window.daemon.send({
      type: 'reject_fix',
      fix_id: fix.fix_id
    })
  }

  if (pendingFixes.length === 0) {
    return (
      <div className="fix-approval-panel empty">
        <div className="panel-header">
          <span className="panel-icon">🩹</span>
          <span className="panel-title">Immune System Fixes</span>
        </div>
        <div className="empty-state">
          <div className="empty-icon">✨</div>
          <div className="empty-text">No pending fixes</div>
          <div className="empty-subtext">Immune system is monitoring...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="fix-approval-panel">
      <div className="panel-header">
        <span className="panel-icon">🩹</span>
        <span className="panel-title">Pending Fixes ({pendingFixes.length})</span>
      </div>

      <div className="fixes-list">
        {pendingFixes.map(fix => (
          <div
            key={fix.fix_id}
            className={`fix-item ${selectedFix?.fix_id === fix.fix_id ? 'selected' : ''}`}
            onClick={() => setSelectedFix(fix)}
          >
            <div className="fix-header">
              <span className="fix-file">{fix.filepath.split('/').pop()}</span>
              <span className={`fix-confidence ${fix.bug_confidence > 0.95 ? 'critical' : 'high'}`}>
                {(fix.bug_confidence * 100).toFixed(0)}%
              </span>
            </div>

            <div className="fix-path">{fix.filepath}</div>

            {fix.rationale && (
              <div className="fix-rationale">
                <span className="rationale-label">Why:</span>
                <span className="rationale-text">{fix.rationale}</span>
              </div>
            )}

            {fix.analysis && (
              <div className="fix-analysis">
                <span className="analysis-label">Agent Analysis:</span>
                <span className="analysis-text">
                  {fix.analysis.rationale || 'Multi-agent analysis complete'}
                </span>
                <span className="agent-info">
                  by {fix.analysis.agent_id} (confidence: {(fix.analysis.confidence * 100).toFixed(0)}%)
                </span>
              </div>
            )}

            {fix.original_code && fix.proposed_code && (
              <div className="code-diff">
                <div className="diff-header">
                  <span className="diff-label">📝 Code Changes:</span>
                  <button
                    className="btn-expand-diff"
                    onClick={(e) => {
                      e.stopPropagation()
                      const diffContent = e.target.closest('.code-diff').querySelector('.diff-content')
                      diffContent.classList.toggle('expanded')
                    }}
                  >
                    {selectedFix?.fix_id === fix.fix_id ? 'Hide' : 'Show'} Diff
                  </button>
                </div>
                <div className="diff-content">
                  <div className="diff-column">
                    <div className="diff-column-header">Original (Buggy)</div>
                    <pre className="code-block original">{fix.original_code}</pre>
                  </div>
                  <div className="diff-column">
                    <div className="diff-column-header">Proposed Fix</div>
                    <pre className="code-block proposed">{fix.proposed_code}</pre>
                  </div>
                </div>
              </div>
            )}

            <div className="fix-actions">
              <button
                className="btn-approve"
                onClick={(e) => {
                  e.stopPropagation()
                  approveFix(fix)
                }}
              >
                ✓ Approve & Apply
              </button>
              <button
                className="btn-reject"
                onClick={(e) => {
                  e.stopPropagation()
                  rejectFix(fix)
                }}
              >
                ✗ Reject
              </button>
            </div>

            <div className="fix-timestamp">
              {new Date(fix.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default FixApprovalPanel
