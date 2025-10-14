import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  root: 'src/mission-control',
  base: './',  // Use relative paths for Electron
  build: {
    outDir: '../../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'src/mission-control/index.html')
      }
    }
  },
  server: {
    port: 5173
  }
})
