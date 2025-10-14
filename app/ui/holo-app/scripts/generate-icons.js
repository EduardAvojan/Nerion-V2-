#!/usr/bin/env node

const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const svgPath = path.join(__dirname, '../assets/icons/nerion-icon.svg');
const outputDir = path.join(__dirname, '../assets/icons');

// Ensure output directory exists
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Icon sizes needed for Electron
const sizes = [16, 32, 64, 128, 256, 512, 1024];

async function generateIcons() {
  console.log('🎨 Generating Nerion icons from SVG...\n');

  for (const size of sizes) {
    const outputPath = path.join(outputDir, `nerion-icon-${size}.png`);

    try {
      await sharp(svgPath)
        .resize(size, size)
        .png()
        .toFile(outputPath);

      console.log(`✅ Generated ${size}x${size} PNG`);
    } catch (error) {
      console.error(`❌ Failed to generate ${size}x${size}:`, error.message);
    }
  }

  console.log('\n🎉 Icon generation complete!');
  console.log(`📁 Icons saved to: ${outputDir}`);
}

generateIcons().catch(console.error);
