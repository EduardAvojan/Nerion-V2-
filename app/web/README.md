# Nerion Mission Control - Web Frontend

React-based web UI for the Nerion Mission Control cockpit.

## Features

- **Dashboard Panels**: System health, signals, memory, artifacts, learning timeline
- **Embedded Terminal**: Real terminal with xterm.js connected to backend PTY
- **Real-Time Updates**: WebSocket events for live dashboard updates
- **Responsive Layout**: Adapts to different screen sizes
- **Nerion Theme**: Beautiful dark theme with cyan accents

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/ed/Nerion-V2/app/web
npm install
```

### 2. Start Backend (Required)

In another terminal:
```bash
cd /Users/ed/Nerion-V2/app/api
python terminal_server.py
```

### 3. Start Frontend

```bash
npm run dev
```

The app will open at `http://localhost:3000`

## Project Structure

```
app/web/
├── index.html              # Entry HTML
├── package.json            # Dependencies
├── vite.config.js          # Vite configuration
└── src/
    ├── main.jsx            # React entry point
    ├── App.jsx             # Main app component
    ├── App.css             # Main layout styles
    ├── index.css           # Global styles
    └── components/
        ├── TopBar.jsx                    # Top status bar
        ├── ImmuneVitalsPanel.jsx         # System health panel
        ├── SignalHealthPanel.jsx         # Component status panel
        ├── MemorySnapshotPanel.jsx       # Memory entries panel
        ├── Terminal.jsx                  # xterm.js terminal
        ├── ArtifactsPanel.jsx            # Generated artifacts
        ├── UpgradeLanePanel.jsx          # Self-improvement proposals
        └── LearningTimelinePanel.jsx     # Learning events
```

## Components

### TopBar
- System status (healthy/warning/critical)
- Uptime display
- Settings and help buttons

### Status Panels (Top Row)
- **Immune Vitals**: Build health, protection status, threats, auto-fixes
- **Signal Health**: Voice, network, learning, LLM status
- **Memory**: Total entries, pinned facts, recent entries

### Terminal (Center)
- Full xterm.js terminal emulator
- WebSocket connection to backend PTY
- Real bash shell with Nerion commands
- Connection status indicator

### Control Panels (Bottom Row)
- **Artifacts**: Generated documents (security audits, plans, analyses)
- **Upgrade Lane**: Self-improvement proposals with impact/risk
- **Learning Timeline**: Recent learned preferences and adjustments

## Development

### Add New Panel

1. Create component in `src/components/YourPanel.jsx`
2. Import in `App.jsx`
3. Add to appropriate panel row

### Styling

- Global styles: `src/index.css`
- Layout: `src/App.css`
- Component-specific: Create `ComponentName.css` alongside component

### WebSocket Events

Handle new event types in `App.jsx`:

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data)

  switch (data.type) {
    case 'your_event':
      // Handle your event
      break
  }
}
```

## Build for Production

```bash
npm run build
```

Output in `dist/` directory. Serve with any static file server.

## Troubleshooting

### "Module not found" errors
```bash
npm install
```

### Terminal not connecting
- Ensure backend is running on `localhost:8000`
- Check browser console for WebSocket errors

### Blank page
- Check browser console for errors
- Verify Vite dev server is running

## Next Steps

- [ ] Add chat mode toggle
- [ ] Implement settings panel
- [ ] Add terminal output parsing for events
- [ ] Mobile responsive improvements
- [ ] Dark/light theme toggle

## References

- [Mission Control Design Doc](../../docs/MISSION_CONTROL_DESIGN.md)
- [React Documentation](https://react.dev/)
- [xterm.js Documentation](https://xtermjs.org/)
- [Vite Documentation](https://vitejs.dev/)
