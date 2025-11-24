const { contextBridge, ipcRenderer } = require('electron');

console.log('[PRELOAD] Script loaded successfully');
console.log('[PRELOAD] contextBridge available:', !!contextBridge);

function safeSend(type, payload = {}) {
  if (!type || typeof type !== 'string') {
    throw new Error('nerion.send requires a message type');
  }
  ipcRenderer.send('nerion-command', { type, payload });
}

function subscribe(channel, handler) {
  const wrapped = (_event, data) => handler(data);
  ipcRenderer.on(channel, wrapped);
  return () => ipcRenderer.removeListener(channel, wrapped);
}

contextBridge.exposeInMainWorld('nerion', {
  ready() {
    console.log('[PRELOAD] Sending nerion-ready');
    ipcRenderer.send('nerion-ready');
  },
  send: safeSend,
  onEvent(handler) {
    console.log('[PRELOAD] Registering event handler');
    const wrapped = (_event, data) => {
      console.log('[PRELOAD] Received event:', data && data.type ? data.type : 'unknown', data);
      handler(data);
    };
    ipcRenderer.on('nerion-event', wrapped);
    return () => {
      console.log('[PRELOAD] Unregistering event handler');
      ipcRenderer.removeListener('nerion-event', wrapped);
    };
  },
  onStatus(handler) {
    return subscribe('nerion-status', handler);
  }
});

// Daemon IPC for fix approval system
contextBridge.exposeInMainWorld('daemon', {
  send(command) {
    console.log('[PRELOAD] Sending daemon command:', command.type);
    ipcRenderer.send('daemon-command', command);
  },
  onFixProposal(handler) {
    console.log('[PRELOAD] Registering fix proposal handler');
    return subscribe('daemon-fix-proposal', handler);
  },
  onPendingFixes(handler) {
    console.log('[PRELOAD] Registering pending fixes handler');
    return subscribe('daemon-pending-fixes', handler);
  },
  onFixResponse(handler) {
    console.log('[PRELOAD] Registering fix response handler');
    return subscribe('daemon-fix-response', handler);
  },
  getPendingFixes() {
    console.log('[PRELOAD] Requesting pending fixes');
    ipcRenderer.send('daemon-get-pending-fixes');
  },
  onStatus(handler) {
    return subscribe('daemon-status', handler);
  }
});

console.log('[PRELOAD] window.nerion and window.daemon exposed successfully');
