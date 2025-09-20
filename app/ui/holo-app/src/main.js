const { app, BrowserWindow, ipcMain, Tray, Menu, nativeImage, globalShortcut } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let tray;
let pythonProcess;
let pythonBuffer = '';

const DEFAULT_WIDTH = 1280;
const DEFAULT_HEIGHT = 720;
const PTT_ACCELERATOR = process.platform === 'darwin' ? 'Cmd+Shift+Space' : 'Ctrl+Shift+Space';

function resolvePreload() {
  return path.join(__dirname, 'preload.js');
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
    minWidth: 960,
    minHeight: 600,
    backgroundColor: '#020617',
    autoHideMenuBar: true,
    title: 'Nerion HOLO',
    webPreferences: {
      preload: resolvePreload(),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function ensureTray() {
  if (tray) {
    return;
  }
  const size = 16;
  const image = nativeImage.createEmpty();
  image.resize({ width: size, height: size });
  tray = new Tray(image);
  tray.setToolTip('Nerion HOLO');
  const menu = Menu.buildFromTemplate([
    {
      label: 'Show',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        } else {
          createWindow();
        }
      },
    },
    {
      label: 'Hide',
      click: () => {
        if (mainWindow) {
          mainWindow.hide();
        }
      },
    },
    { type: 'separator' },
    {
      label: 'Quit Nerion',
      click: () => {
        app.quit();
      },
    },
  ]);
  tray.setContextMenu(menu);
  tray.on('click', () => {
    if (!mainWindow) {
      createWindow();
    } else if (mainWindow.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow.show();
    }
  });
}

function splitArgs(source) {
  if (!source || typeof source !== 'string') {
    return [];
  }
  return source.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
}

function stripQuotes(value) {
  if (!value) {
    return value;
  }
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    return value.slice(1, -1);
  }
  return value;
}

function spawnPythonBridge() {
  if (pythonProcess) {
    return pythonProcess;
  }
  // Prefer project-local venv if present
  const projectRoot = path.resolve(__dirname, '../../../..');
  const venvPy = path.join(projectRoot, '.venv', 'bin', process.platform === 'win32' ? 'python.exe' : 'python');
  const pythonCmd = process.env.NERION_PYTHON || (require('fs').existsSync(venvPy) ? venvPy : 'python3');
  const pyEntry = process.env.NERION_PY_ENTRY || 'app.nerion_chat';
  const pyArgs = splitArgs(process.env.NERION_PY_ARGS).map(stripQuotes);
  const entryArgs = pyEntry.endsWith('.py') ? [pyEntry] : ['-m', pyEntry];
  const args = [...entryArgs, ...pyArgs];
  // Ensure we run from the project root so Python can import 'app.nerion_chat'
  const env = {
    ...process.env,
    NERION_UI_CHANNEL: 'holo-electron',
    PYTHONPATH: [projectRoot, process.env.PYTHONPATH || ''].filter(Boolean).join(path.delimiter),
  };

  pythonProcess = spawn(pythonCmd, args, {
    env,
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: projectRoot,
  });

  pythonProcess.stdout.on('data', (chunk) => {
    pythonBuffer += chunk.toString();
    let index;
    while ((index = pythonBuffer.indexOf('\n')) >= 0) {
      const line = pythonBuffer.slice(0, index).trim();
      pythonBuffer = pythonBuffer.slice(index + 1);
      if (!line) {
        continue;
      }
      try {
        const payload = JSON.parse(line);
        dispatchToRenderer(payload);
      } catch (error) {
        console.error('[HOLO] Failed to parse Python event:', error, '\nLine:', line);
      }
    }
  });

  pythonProcess.stderr.on('data', (chunk) => {
    console.error('[HOLO][python]', chunk.toString());
  });

  pythonProcess.on('exit', (code, signal) => {
    console.log('[HOLO] Python process exited', { code, signal });
    pythonProcess = null;
    pythonBuffer = '';
    if (mainWindow) {
      mainWindow.webContents.send('nerion-status', {
        type: 'python_exit',
        code,
        signal,
      });
    }
  });

  pythonProcess.on('error', (err) => {
    console.error('[HOLO] Failed to start Python runtime', err);
  });

  return pythonProcess;
}

function dispatchToRenderer(message) {
  if (mainWindow && mainWindow.webContents) {
    mainWindow.webContents.send('nerion-event', message);
  }
}

function sendToPython(message) {
  if (!pythonProcess || !pythonProcess.stdin || !pythonProcess.stdin.writable) {
    console.warn('[HOLO] Python stdin not writable, dropping message:', message);
    return;
  }
  try {
    pythonProcess.stdin.write(`${JSON.stringify(message)}\n`);
  } catch (error) {
    console.error('[HOLO] Failed to send message to python', error);
  }
}

function registerIpcHandlers() {
  ipcMain.on('nerion-command', (_event, payload) => {
    if (!payload || typeof payload !== 'object') {
      return;
    }
    sendToPython(payload);
  });

  ipcMain.on('nerion-ready', () => {
    spawnPythonBridge();
  });
}

function registerShortcuts() {
  globalShortcut.register(PTT_ACCELERATOR, () => {
    const payload = {
      type: 'ptt',
      payload: {
        state: 'toggle',
        source: 'global_shortcut',
      },
    };
    sendToPython(payload);
    if (mainWindow) {
      mainWindow.webContents.send('nerion-event', {
        type: 'ptt_shortcut',
        payload,
        ts: Date.now(),
      });
    }
  });
}

function unregisterShortcuts() {
  globalShortcut.unregisterAll();
}

function cleanupPython() {
  if (!pythonProcess) {
    return;
  }
  try {
    pythonProcess.kill();
  } catch (error) {
    console.error('[HOLO] Error while terminating python process', error);
  }
  pythonProcess = null;
}

app.on('ready', () => {
  createWindow();
  ensureTray();
  registerIpcHandlers();
  registerShortcuts();
  spawnPythonBridge();
});

app.on('before-quit', () => {
  unregisterShortcuts();
  cleanupPython();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (!mainWindow) {
    createWindow();
  }
});

process.on('SIGINT', () => {
  app.quit();
});

process.on('SIGTERM', () => {
  app.quit();
});
