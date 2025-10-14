import React, { useState, useRef, useEffect } from 'react'
import './ChatView.css'

export default function ChatView() {
  const [messages, setMessages] = useState([
    {
      role: 'system',
      content: '🧬 Nerion Chat Mode Active',
      timestamp: new Date().toLocaleTimeString()
    }
  ])
  const [input, setInput] = useState('')
  const [isThinking, setIsThinking] = useState(false)
  const messagesEndRef = useRef(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || isThinking) return

    // Add user message
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString()
    }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsThinking(true)

    // TODO: Connect to nerion-chat WebSocket or API
    // For now, simulate response
    setTimeout(() => {
      const assistantMessage = {
        role: 'assistant',
        content: `I received your message: "${input}"\n\nChat mode will be connected to the Nerion reasoning engine once the backend is integrated.`,
        timestamp: new Date().toLocaleTimeString(),
        confidence: 0.85
      }
      setMessages(prev => [...prev, assistantMessage])
      setIsThinking(false)
    }, 1000)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-view">
      <div className="chat-header">
        <div className="chat-title">💬 Nerion Chat</div>
        <div className="chat-status">
          <span className={`status-dot ${isThinking ? 'thinking' : 'ready'}`}></span>
          {isThinking ? 'Thinking...' : 'Ready'}
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            {msg.role === 'system' && (
              <div className="message-system">
                {msg.content}
              </div>
            )}

            {msg.role === 'user' && (
              <div className="message-user">
                <div className="message-header">
                  <span className="message-role">You</span>
                  <span className="message-time">{msg.timestamp}</span>
                </div>
                <div className="message-content">{msg.content}</div>
              </div>
            )}

            {msg.role === 'assistant' && (
              <div className="message-assistant">
                <div className="message-header">
                  <span className="message-role">🧬 Nerion</span>
                  <span className="message-time">{msg.timestamp}</span>
                  {msg.confidence && (
                    <span className="message-confidence">
                      {(msg.confidence * 100).toFixed(0)}% confident
                    </span>
                  )}
                </div>
                <div className="message-content">
                  {msg.content}
                </div>
                {msg.thoughtProcess && (
                  <details className="thought-process">
                    <summary>View reasoning</summary>
                    <pre>{msg.thoughtProcess}</pre>
                  </details>
                )}
              </div>
            )}
          </div>
        ))}

        {isThinking && (
          <div className="chat-message assistant">
            <div className="message-assistant">
              <div className="thinking-indicator">
                <span className="thinking-dot"></span>
                <span className="thinking-dot"></span>
                <span className="thinking-dot"></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <textarea
          className="chat-input"
          placeholder="Ask Nerion anything... (Shift+Enter for new line)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isThinking}
          rows={1}
        />
        <button
          className="chat-send-btn"
          onClick={handleSend}
          disabled={!input.trim() || isThinking}
        >
          Send
        </button>
      </div>

      <div className="chat-hints">
        <div className="hint">💡 Tip: Use Terminal mode for precise commands, Chat mode for exploration</div>
      </div>
    </div>
  )
}
