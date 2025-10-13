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
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    if (!terminalRef.current) return

    // Initialize xterm
    const term = new XTerm({
      cursorBlink: true,
      fontSize: 14,
      fontFamily: 'Menlo, Monaco, "Courier New", monospace',
      theme: {
        background: '#0a0e14',
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
    term.writeln('\x1b[1;36m╔════════════════════════════════════════════════════╗\x1b[0m')
    term.writeln('\x1b[1;36m║  🧬 NERION MISSION CONTROL - Terminal             ║\x1b[0m')
    term.writeln('\x1b[1;36m╚════════════════════════════════════════════════════╝\x1b[0m')
    term.writeln('')
    term.writeln('\x1b[1;33mConnecting to terminal server...\x1b[0m')
    term.writeln('')

    // Connect to WebSocket
    const ws = new WebSocket('ws://localhost:8000/api/terminal')
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[Terminal] Connected')
      setConnected(true)

      // Send terminal size
      ws.send(JSON.stringify({
        type: 'resize',
        cols: term.cols,
        rows: term.rows
      }))
    }

    ws.onclose = () => {
      console.log('[Terminal] Disconnected')
      setConnected(false)
      term.write('\r\n\n\x1b[1;31m[Connection closed]\x1b[0m\r\n')
    }

    ws.onerror = (error) => {
      console.error('[Terminal] Error:', error)
      setConnected(false)
    }

    ws.onmessage = (event) => {
      // Terminal output from server
      const data = new Uint8Array(event.data)
      term.write(data)
    }

    // Send user input to server
    term.onData(data => {
      if (ws.readyState === WebSocket.OPEN) {
        const encoder = new TextEncoder()
        ws.send(encoder.encode(data))
      }
    })

    // Handle window resize
    const handleResize = () => {
      fitAddon.fit()
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
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
      if (ws) ws.close()
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
          <span className={`status-indicator ${connected ? 'status-online' : 'status-error'}`}></span>
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
      <div ref={terminalRef} className="terminal"></div>
    </div>
  )
}
