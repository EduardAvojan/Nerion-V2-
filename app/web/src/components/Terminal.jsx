import React, { useEffect, useRef, useState } from 'react'
import { Terminal as XTerm } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import '@xterm/xterm/css/xterm.css'
import './Terminal.css'

export default function Terminal() {
  const terminalRef = useRef(null)
  const xtermRef = useRef(null)
  const fitAddonRef = useRef(null)
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const reconnectAttemptsRef = useRef(0)
  const [connected, setConnected] = useState(false)
  const [reconnecting, setReconnecting] = useState(false)

  useEffect(() => {
    if (!terminalRef.current) return

    // Initialize xterm
    const term = new XTerm({
      cursorBlink: true,
      fontSize: 14,
      fontFamily: 'Menlo, Monaco, "Courier New", monospace',
      theme: {
        background: 'transparent',
        foreground: '#e0e0e0',
        cursor: '#00d9ff',
        cursorAccent: '#0a0e14',
        selection: 'rgba(0, 217, 255, 0.3)',
        black: '#000000',
        red: '#ff3333',
        green: '#00ff88',
        yellow: '#ffcc00',
        blue: '#00aaff',
        magenta: '#ff66ff',
        cyan: '#00d9ff',
        white: '#ffffff',
        brightBlack: '#666666',
        brightRed: '#ff6666',
        brightGreen: '#66ffaa',
        brightYellow: '#ffdd66',
        brightBlue: '#66bbff',
        brightMagenta: '#ff99ff',
        brightCyan: '#66eeff',
        brightWhite: '#ffffff'
      }
    })

    const fitAddon = new FitAddon()
    term.loadAddon(fitAddon)

    term.open(terminalRef.current)
    fitAddon.fit()

    xtermRef.current = term
    fitAddonRef.current = fitAddon

    // Welcome message
    term.writeln('\x1b[1;36m╔═════════════════════════════════════════════════╗\x1b[0m')
    term.writeln('\x1b[1;36m║  🧬 NERION MISSION CONTROL - Terminal            ║\x1b[0m')
    term.writeln('\x1b[1;36m╚═════════════════════════════════════════════════╝\x1b[0m')
    term.writeln('')
    term.writeln('\x1b[1;33mConnecting to terminal server...\x1b[0m')
    term.writeln('')

    // Automatic reconnection with exponential backoff
    const connectWebSocket = () => {
      // Clear any existing reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }

      // Close existing connection if any
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close()
      }

      const ws = new WebSocket('ws://localhost:8000/api/terminal')
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = () => {
        console.log('[Terminal] Connected')
        setConnected(true)
        setReconnecting(false)
        reconnectAttemptsRef.current = 0

        // Send terminal size
        ws.send(JSON.stringify({
          type: 'resize',
          cols: term.cols,
          rows: term.rows
        }))

        term.write('\x1b[1;32m[Connected]\x1b[0m\r\n')
      }

      ws.onclose = (event) => {
        console.log('[Terminal] Disconnected', event.code, event.reason)
        setConnected(false)

        // Only attempt reconnection if not intentionally closed
        if (event.code !== 1000) {
          attemptReconnect()
        } else {
          term.write('\r\n\x1b[1;33m[Connection closed normally]\x1b[0m\r\n')
        }
      }

      ws.onerror = (error) => {
        console.error('[Terminal] WebSocket error:', error)
        setConnected(false)
      }

      ws.onmessage = (event) => {
        // Check if it's JSON (ping/pong) or binary (terminal data)
        if (typeof event.data === 'string') {
          try {
            const message = JSON.parse(event.data)
            if (message.type === 'ping') {
              // Respond to ping
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }))
              }
            }
          } catch (e) {
            // Not JSON, ignore
          }
        } else {
          // Terminal output from server
          const data = new Uint8Array(event.data)
          term.write(data)
        }
      }

      // Send user input to server
      const onDataDisposable = term.onData(data => {
        if (ws.readyState === WebSocket.OPEN) {
          const encoder = new TextEncoder()
          ws.send(encoder.encode(data))
        }
      })

      return onDataDisposable
    }

    // Reconnection with exponential backoff
    const attemptReconnect = () => {
      const maxAttempts = 10
      const baseDelay = 1000 // 1 second
      const maxDelay = 30000 // 30 seconds

      if (reconnectAttemptsRef.current >= maxAttempts) {
        term.write('\r\n\x1b[1;31m[Max reconnection attempts reached. Please refresh the page.]\x1b[0m\r\n')
        setReconnecting(false)
        return
      }

      reconnectAttemptsRef.current++
      setReconnecting(true)

      // Calculate delay with exponential backoff
      const delay = Math.min(baseDelay * Math.pow(2, reconnectAttemptsRef.current - 1), maxDelay)

      term.write(`\r\n\x1b[1;33m[Reconnecting in ${(delay / 1000).toFixed(0)}s... (attempt ${reconnectAttemptsRef.current}/${maxAttempts})]\x1b[0m\r\n`)

      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`[Terminal] Reconnection attempt ${reconnectAttemptsRef.current}`)
        connectWebSocket()
      }, delay)
    }

    // Initial connection
    const onDataDisposable = connectWebSocket()

    // Handle window resize
    const handleResize = () => {
      fitAddon.fit()
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'resize',
          cols: term.cols,
          rows: term.rows
        }))
      }
    }

    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)

      // Clear reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }

      // Close WebSocket
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted')
      }

      // Dispose terminal
      if (onDataDisposable) onDataDisposable.dispose()
      if (term) term.dispose()
    }
  }, [])

  return (
    <div className="terminal-container">
      <div className="terminal-header">
        <div className="terminal-title">
          🖥️ Terminal
        </div>
        <div className="terminal-status">
          <span className={`status-indicator ${
            connected ? 'status-online' : reconnecting ? 'status-warning' : 'status-error'
          }`}></span>
          {connected ? 'Connected' : reconnecting ? 'Reconnecting...' : 'Disconnected'}
        </div>
      </div>
      <div ref={terminalRef} className="terminal"></div>
    </div>
  )
}
