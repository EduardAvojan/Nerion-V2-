# Nerion Quick Start Guide

## 🚀 Launch Nerion (3 Ways)

### Option 1: Double-Click the App (Easiest!)
```
📁 /Users/ed/Nerion-V2/Nerion.app
```
Just **double-click `Nerion.app`** in Finder - that's it! The app will:
- Start the Mission Control GUI
- Show the Nerion icon in your dock and system tray
- Connect to the daemon (if running)

### Option 2: From Finder
1. Open `/Users/ed/Nerion-V2/` in Finder
2. Double-click **Nerion.app**

### Option 3: From Command Line
```bash
# Quick launch
open /Users/ed/Nerion-V2/Nerion.app

# Or from project root
cd /Users/ed/Nerion-V2
open Nerion.app
```

## 📦 Installation (Optional)

### Add to Applications Folder
```bash
# Copy to Applications so you can launch from Spotlight
cp -r /Users/ed/Nerion-V2/app/ui/holo-app/build/mac-arm64/Nerion.app /Applications/

# Now you can:
# - Press Cmd+Space and type "Nerion"
# - Find it in Launchpad
# - Add to Dock
```

### Add to Dock (Recommended)
1. Launch Nerion.app
2. Right-click the Nerion icon in the Dock
3. Options → Keep in Dock

Now Nerion is always one click away!

## 🧬 Background Daemon (24/7 Immune System)

### Check if Daemon is Running
```bash
ls ~/.nerion/daemon.sock
```

### Start Daemon (Auto-start on boot)
```bash
cd daemon
./install_service.sh
```

### View Daemon Logs
```bash
tail -f ~/.nerion/daemon.log
```

### Stop Daemon
```bash
# Via GUI: Right-click tray icon → "Quit Nerion"

# Or manually:
launchctl unload ~/Library/LaunchAgents/com.nerion.daemon.plist
```

## 🎯 How It Works

```
┌─────────────────────────────────────┐
│  Nerion.app (Double-click to run)  │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Mission Control GUI        │   │
│  │  - Terminal                 │   │
│  │  - Training Dashboard       │   │
│  │  - Thought Process          │   │
│  │  - Artifacts                │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  System Tray Icon           │   │
│  │  🟢 Healthy                 │   │
│  │  🟡 Warning                 │   │
│  │  🔴 Critical                │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
          ↕ Socket
┌─────────────────────────────────────┐
│  Nerion Daemon (Background)         │
│  - Watches codebase 24/7            │
│  - Trains GNN                       │
│  - Detects threats                  │
│  - Auto-fixes issues                │
└─────────────────────────────────────┘
```

## 🔧 Development

### Rebuild the App (After Code Changes)
```bash
cd /Users/ed/Nerion-V2/app/ui/holo-app

# Build React UI
npm run build:react

# Build macOS app
npm run build:dir

# Or do both at once
npm run build
```

### Run in Development Mode
```bash
cd /Users/ed/Nerion-V2/app/ui/holo-app

# Start in dev mode (no packaging)
npm start
```

## 📂 File Locations

| Item | Location |
|------|----------|
| **Nerion.app** | `/Users/ed/Nerion-V2/Nerion.app` (symlink) |
| **Actual app** | `/Users/ed/Nerion-V2/app/ui/holo-app/build/mac-arm64/Nerion.app` |
| **Daemon socket** | `~/.nerion/daemon.sock` |
| **Daemon logs** | `~/.nerion/daemon.log` |
| **LaunchAgent** | `~/Library/LaunchAgents/com.nerion.daemon.plist` |

## 🎨 App Icon

The Nerion icon represents:
- 🧬 DNA helix (biological immune system)
- 🧠 Neural network (AI/ML)
- 🛡️ Shield (protection)
- 💠 Cyan theme (Mission Control)

## ❓ Troubleshooting

### App Won't Open
```bash
# Check if app is valid
ls -la /Users/ed/Nerion-V2/Nerion.app

# Try opening from terminal to see errors
open /Users/ed/Nerion-V2/Nerion.app
```

### "App is damaged" Error (macOS Security)
```bash
# Remove quarantine attribute
xattr -cr /Users/ed/Nerion-V2/app/ui/holo-app/build/mac-arm64/Nerion.app
```

### Daemon Won't Connect
```bash
# Check daemon is running
ls ~/.nerion/daemon.sock

# Start daemon manually
cd /Users/ed/Nerion-V2
./start_nerion.sh
```

### Rebuild Everything
```bash
cd /Users/ed/Nerion-V2/app/ui/holo-app

# Clean build
rm -rf build/ dist/ node_modules/.vite

# Rebuild
npm run build:react
npm run build:dir
```

## 🚀 Next Steps

1. **Launch Nerion**: Double-click `Nerion.app`
2. **Install daemon**: Run `daemon/install_service.sh` for 24/7 operation
3. **Add to Dock**: Right-click dock icon → Keep in Dock
4. **Optional**: Copy to `/Applications/` for Spotlight access

---

**You're all set!** Nerion is now a proper macOS application. Just double-click the icon and start protecting your codebase!
