const { contextBridge, ipcRenderer } = require('electron');

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
    ipcRenderer.send('nerion-ready');
  },
  send: safeSend,
  onEvent(handler) {
    return subscribe('nerion-event', handler);
  },
  onStatus(handler) {
    return subscribe('nerion-status', handler);
  }
});
