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

console.log('[PRELOAD] window.nerion exposed successfully');
