const { app, BrowserWindow, ipcMain, Tray, Menu, nativeImage, globalShortcut } = require('electron');
const path = require('path');
const { spawn, exec } = require('child_process');
const net = require('net');
const os = require('os');

let mainWindow;
let tray;
let pythonProcess;
let pythonBuffer = '';
let daemonSocket = null;
let daemonStatus = { health: 'unknown', status: 'disconnected' };
let isQuitting = false;

const DEFAULT_WIDTH = 1280;
const DEFAULT_HEIGHT = 720;
const PTT_ACCELERATOR = process.platform === 'darwin' ? 'Cmd+Shift+Space' : 'Ctrl+Shift+Space';
const DAEMON_SOCKET_PATH = path.join(os.homedir(), '.nerion', 'daemon.sock');

function getShellEnv() {
  return new Promise((resolve, reject) => {
    const shell = process.env.SHELL || (process.platform === 'win32' ? 'powershell.exe' : 'bash');
    // For non-Windows, execute `env` in a login shell to get full environment
    const command = process.platform === 'win32'
      ? '$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User"); echo "---ENV_START---"; Get-ChildItem env: | Format-Table -HideTableHeaders -AutoSize | Out-String -Width 1024; echo "---ENV_END---"'
      : `${shell} -lc 'echo "---ENV_START---" && env && echo "---ENV_END---"'`;

    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`[HOLO] Failed to get shell env: ${stderr}`);
        return reject(error);
      }
      const match = stdout.match(/---ENV_START---([\s\S]*)---ENV_END---/);
      if (!match || !match[1]) {
        console.error('[HOLO] Failed to parse shell env output.');
        return resolve({});
      }
      const env = {};
      match[1].trim().split('\n').forEach(line => {
        const parts = line.split('=');
        if (parts.length >= 2) {
          const key = parts.shift();
          const value = parts.join('=');
          if (key && key.trim()) {
            env[key.trim()] = value.trim();
          }
        }
      });
      resolve(env);
    });
  });
}

function resolvePreload() {
  return path.join(__dirname, 'preload.js');
}

function createWindow() {
  const iconPath = path.join(__dirname, '../assets/icons/nerion-icon-512.png');

  mainWindow = new BrowserWindow({
    width: 1600,
    height: 900,
    minWidth: 1280,
    minHeight: 720,
    backgroundColor: '#0f172a',
    autoHideMenuBar: true,
    title: 'Nerion Mission Control',
    icon: iconPath,
    webPreferences: {
      preload: resolvePreload(),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  // Load the built React app
  const distPath = path.join(__dirname, '../dist/index.html');
  mainWindow.loadFile(distPath);

  // Minimize to tray instead of closing
  mainWindow.on('close', (event) => {
    if (!isQuitting) {
      event.preventDefault();
      mainWindow.hide();

      // Show notification that Nerion is still running
      if (tray && !mainWindow.isVisible()) {
        tray.displayBalloon({
          title: 'Nerion Still Running',
          content: 'Nerion immune system continues monitoring in the background. Click the tray icon to reopen.'
        });
      }
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Connect to daemon
  connectToDaemon();
}

function updateTrayMenu() {
  if (!tray) return;

  const statusIcon = daemonStatus.health === 'healthy' ? '🟢' :
                     daemonStatus.health === 'warning' ? '🟡' :
                     daemonStatus.health === 'critical' ? '🔴' : '⚪';

  const menu = Menu.buildFromTemplate([
    {
      label: `${statusIcon} Nerion Immune System`,
      enabled: false,
    },
    {
      label: `Status: ${daemonStatus.status}`,
      enabled: false,
    },
    {
      label: `Health: ${daemonStatus.health}`,
      enabled: false,
    },
    { type: 'separator' },
    {
      label: 'Show Mission Control',
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
      label: 'Hide Mission Control',
      click: () => {
        if (mainWindow) {
          mainWindow.hide();
        }
      },
    },
    { type: 'separator' },
    {
      label: 'Quit Nerion (stops immune system)',
      click: () => {
        isQuitting = true;
        app.quit();
      },
    },
  ]);
  tray.setContextMenu(menu);
  tray.setToolTip(`Nerion: ${daemonStatus.health}`);
}

function ensureTray() {
  if (tray) {
    return;
  }

  // Use platform-appropriate tray icon size
  const trayIconSize = process.platform === 'darwin' ? 16 : 32;
  const trayIconPath = path.join(__dirname, `../assets/icons/nerion-icon-${trayIconSize}.png`);

  let trayIcon;
  try {
    trayIcon = nativeImage.createFromPath(trayIconPath);
    if (process.platform === 'darwin') {
      trayIcon = trayIcon.resize({ width: 16, height: 16 });
    }
  } catch (error) {
    console.error('[HOLO] Failed to load tray icon, using empty image:', error);
    trayIcon = nativeImage.createEmpty();
  }

  tray = new Tray(trayIcon);
  updateTrayMenu();

  tray.on('click', () => {
    if (!mainWindow) {
      createWindow();
    } else if (mainWindow.isVisible()) {
      mainWindow.hide();
    } else {
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

function connectToDaemon() {
  if (daemonSocket) {
    daemonSocket.destroy();
  }

  console.log('[HOLO] Connecting to Nerion daemon...');

  daemonSocket = net.createConnection(DAEMON_SOCKET_PATH, () => {
    console.log('[HOLO] Connected to Nerion daemon');
    daemonStatus.status = 'connected';
    updateTrayMenu();

    // Request initial status
    const message = JSON.stringify({ type: 'get_status' }) + '\n';
    daemonSocket.write(message);
  });

  let buffer = '';
  daemonSocket.on('data', (data) => {
    buffer += data.toString();

    // Process complete messages (newline-delimited JSON)
    const lines = buffer.split('\n');
    buffer = lines.pop(); // Keep incomplete line in buffer

    for (const line of lines) {
      if (!line.trim()) continue;

      try {
        const message = JSON.parse(line);
        handleDaemonMessage(message);
      } catch (err) {
        console.error('[HOLO] Failed to parse daemon message:', err);
      }
    }
  });

  daemonSocket.on('error', (err) => {
    console.error('[HOLO] Daemon connection error:', err.message);
    daemonStatus.status = 'disconnected';
    daemonStatus.health = 'unknown';
    updateTrayMenu();

    // Retry connection after 5 seconds
    setTimeout(connectToDaemon, 5000);
  });

  daemonSocket.on('close', () => {
    console.log('[HOLO] Daemon connection closed');
    daemonStatus.status = 'disconnected';
    updateTrayMenu();
    daemonSocket = null;

    // Retry connection after 5 seconds
    setTimeout(connectToDaemon, 5000);
  });
}

function handleDaemonMessage(message) {
  const { type, data } = message;

  if (type === 'status' || type === 'status_update') {
    daemonStatus = {
      health: data.health || 'unknown',
      status: data.status || 'unknown',
      threats: data.threats_detected || 0,
      fixes: data.auto_fixes_applied || 0,
      files: data.files_monitored || 0,
      training: data.gnn_training || false,
    };
    updateTrayMenu();

    // Forward to renderer
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('daemon-status', data);
    }
  }
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

function spawnPythonBridge(shellEnv = {}) {
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
    ...shellEnv,
    ...process.env,
    NERION_UI_CHANNEL: 'holo-electron',
    PYTHONPATH: [projectRoot, shellEnv.PYTHONPATH || '', process.env.PYTHONPATH || ''].filter(Boolean).join(path.delimiter),
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
    getShellEnv().then(shellEnv => {
      spawnPythonBridge(shellEnv);
    }).catch(() => {
      console.warn('[HOLO] Failed to get shell env, spawning with default env.');
      spawnPythonBridge(); // Fallback to original behavior
    });
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
  getShellEnv().then(shellEnv => {
    spawnPythonBridge(shellEnv);
  }).catch(() => {
    console.warn('[HOLO] Failed to get shell env, spawning with default env.');
    spawnPythonBridge(); // Fallback
  });
});

app.on('before-quit', () => {
  isQuitting = true;
  unregisterShortcuts();
  cleanupPython();

  // Send shutdown command to daemon
  if (daemonSocket && daemonSocket.writable) {
    const message = JSON.stringify({ type: 'shutdown' }) + '\n';
    daemonSocket.write(message);
  }
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
