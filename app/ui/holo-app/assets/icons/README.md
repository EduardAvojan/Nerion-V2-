# Nerion App Icons

This folder contains the Nerion application icon in multiple formats and sizes.

## Files

- **nerion-icon.svg** - Master SVG icon (512x512)
  - Vector format, can be scaled to any size
  - Design represents Nerion's immune system concept:
    - DNA double helix structure (biological immune system)
    - Neural network nodes and connections (AI/ML)
    - Shield overlay (protection concept)
    - Cyan color scheme (#00d9ff) matching Mission Control UI
    - Large "N" branding

- **nerion-icon-{size}.png** - PNG exports at various sizes
  - 16x16 - Tray icon (macOS)
  - 32x32 - Tray icon (Windows/Linux), small icons
  - 64x64 - Standard application icon
  - 128x128 - High-DPI application icon
  - 256x256 - High-DPI application icon (2x)
  - 512x512 - Main application icon
  - 1024x1024 - Retina display application icon

## Usage in Electron

```javascript
// Main window icon (main.js line 59)
const iconPath = path.join(__dirname, '../assets/icons/nerion-icon-512.png');
mainWindow = new BrowserWindow({
  icon: iconPath,
  // ... other options
});

// Tray icon (main.js line 164)
const trayIconSize = process.platform === 'darwin' ? 16 : 32;
const trayIconPath = path.join(__dirname, `../assets/icons/nerion-icon-${trayIconSize}.png`);
tray = new Tray(nativeImage.createFromPath(trayIconPath));
```

## Regenerating Icons

If you modify the SVG source, regenerate PNG icons:

```bash
npm run generate:icons
```

This runs `scripts/generate-icons.js` which uses Sharp to convert the SVG to PNG at all required sizes.

## Future: Platform-Specific Formats

For production builds with electron-builder, you'll want:

- **macOS**: `.icns` format (all sizes bundled)
- **Windows**: `.ico` format (multiple sizes)
- **Linux**: PNG files at standard sizes (48, 64, 128, 256, 512)

Generate with:
```bash
# Install electron-icon-builder
npm install --save-dev electron-icon-builder

# Generate .icns and .ico from PNG
npx electron-icon-builder --input=./assets/icons/nerion-icon-1024.png --output=./assets/icons/build
```

## Design Attribution

Icon designed to represent:
- 🧬 Biological immune system (DNA helix)
- 🧠 Neural networks (connected nodes)
- 🛡️ Protection (shield outline)
- 💠 Nerion branding (stylized "N")

Color palette: Cyan (#00d9ff) with dark background (#0a0e14 → #1a1f2e gradient)
