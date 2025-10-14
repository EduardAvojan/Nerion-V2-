import React, { useState } from 'react'
import './SettingsPanel.css'

export default function SettingsPanel({ isOpen, onClose, theme, onThemeChange }) {
  const [settings, setSettings] = useState({
    voice: {
      enabled: true,
      provider: 'openai',
      model: 'whisper-1'
    },
    llm: {
      provider: 'anthropic',
      model: 'claude-3-5-sonnet-20241022',
      temperature: 0.7
    },
    network: {
      autoUpdate: true,
      checkInterval: 300,
      proxy: ''
    },
    learning: {
      enabled: true,
      autoPin: true,
      confidenceThreshold: 0.8
    },
    immune: {
      autoFix: true,
      threatSensitivity: 'medium',
      maxAutoFixes: 10
    },
    ui: {
      theme: theme || 'dark',
      compactMode: false,
      showTooltips: true
    }
  })

  const handleChange = (category, key, value) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value
      }
    }))

    // Handle theme change specially
    if (category === 'ui' && key === 'theme' && onThemeChange) {
      onThemeChange(value)
    }
  }

  const handleSave = () => {
    // TODO: Save settings to backend/localStorage
    console.log('Saving settings:', settings)
    onClose()
  }

  const handleReset = () => {
    if (confirm('Reset all settings to defaults?')) {
      // Reset to defaults
      window.location.reload()
    }
  }

  if (!isOpen) return null

  return (
    <>
      <div className="settings-overlay" onClick={onClose} />
      <div className="settings-panel">
        <div className="settings-header">
          <h2>⚙️ Settings</h2>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="settings-content">
          {/* Voice Settings */}
          <section className="settings-section">
            <h3>🎤 Voice</h3>
            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={settings.voice.enabled}
                  onChange={(e) => handleChange('voice', 'enabled', e.target.checked)}
                />
                Enable voice input
              </label>
            </div>
            <div className="setting-item">
              <label>Provider</label>
              <select
                value={settings.voice.provider}
                onChange={(e) => handleChange('voice', 'provider', e.target.value)}
              >
                <option value="openai">OpenAI Whisper</option>
                <option value="google">Google Speech</option>
                <option value="local">Local (Vosk)</option>
              </select>
            </div>
          </section>

          {/* LLM Settings */}
          <section className="settings-section">
            <h3>🤖 Language Model</h3>
            <div className="setting-item">
              <label>Provider</label>
              <select
                value={settings.llm.provider}
                onChange={(e) => handleChange('llm', 'provider', e.target.value)}
              >
                <option value="anthropic">Anthropic Claude</option>
                <option value="openai">OpenAI GPT</option>
                <option value="ollama">Ollama (Local)</option>
              </select>
            </div>
            <div className="setting-item">
              <label>Model</label>
              <select
                value={settings.llm.model}
                onChange={(e) => handleChange('llm', 'model', e.target.value)}
              >
                {settings.llm.provider === 'anthropic' && (
                  <>
                    <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
                    <option value="claude-3-opus-20240229">Claude 3 Opus</option>
                    <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
                  </>
                )}
                {settings.llm.provider === 'openai' && (
                  <>
                    <option value="gpt-4-turbo">GPT-4 Turbo</option>
                    <option value="gpt-4">GPT-4</option>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                  </>
                )}
                {settings.llm.provider === 'ollama' && (
                  <>
                    <option value="llama2">Llama 2</option>
                    <option value="codellama">Code Llama</option>
                    <option value="mistral">Mistral</option>
                  </>
                )}
              </select>
            </div>
            <div className="setting-item">
              <label>Temperature: {settings.llm.temperature}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={settings.llm.temperature}
                onChange={(e) => handleChange('llm', 'temperature', parseFloat(e.target.value))}
              />
            </div>
          </section>

          {/* Network Settings */}
          <section className="settings-section">
            <h3>🌐 Network</h3>
            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={settings.network.autoUpdate}
                  onChange={(e) => handleChange('network', 'autoUpdate', e.target.checked)}
                />
                Auto-update dependencies
              </label>
            </div>
            <div className="setting-item">
              <label>Check interval (seconds)</label>
              <input
                type="number"
                value={settings.network.checkInterval}
                onChange={(e) => handleChange('network', 'checkInterval', parseInt(e.target.value))}
              />
            </div>
          </section>

          {/* Learning Settings */}
          <section className="settings-section">
            <h3>📚 Learning</h3>
            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={settings.learning.enabled}
                  onChange={(e) => handleChange('learning', 'enabled', e.target.checked)}
                />
                Enable learning
              </label>
            </div>
            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={settings.learning.autoPin}
                  onChange={(e) => handleChange('learning', 'autoPin', e.target.checked)}
                />
                Auto-pin important memories
              </label>
            </div>
            <div className="setting-item">
              <label>Confidence threshold: {settings.learning.confidenceThreshold}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={settings.learning.confidenceThreshold}
                onChange={(e) => handleChange('learning', 'confidenceThreshold', parseFloat(e.target.value))}
              />
            </div>
          </section>

          {/* Immune System Settings */}
          <section className="settings-section">
            <h3>🦠 Immune System</h3>
            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={settings.immune.autoFix}
                  onChange={(e) => handleChange('immune', 'autoFix', e.target.checked)}
                />
                Enable auto-fix
              </label>
            </div>
            <div className="setting-item">
              <label>Threat sensitivity</label>
              <select
                value={settings.immune.threatSensitivity}
                onChange={(e) => handleChange('immune', 'threatSensitivity', e.target.value)}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
            <div className="setting-item">
              <label>Max auto-fixes per session</label>
              <input
                type="number"
                value={settings.immune.maxAutoFixes}
                onChange={(e) => handleChange('immune', 'maxAutoFixes', parseInt(e.target.value))}
              />
            </div>
          </section>

          {/* UI Settings */}
          <section className="settings-section">
            <h3>🎨 Interface</h3>
            <div className="setting-item">
              <label>Theme</label>
              <select
                value={settings.ui.theme}
                onChange={(e) => handleChange('ui', 'theme', e.target.value)}
              >
                <option value="dark">Dark</option>
                <option value="light">Light</option>
              </select>
            </div>
            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={settings.ui.compactMode}
                  onChange={(e) => handleChange('ui', 'compactMode', e.target.checked)}
                />
                Compact mode
              </label>
            </div>
            <div className="setting-item">
              <label>
                <input
                  type="checkbox"
                  checked={settings.ui.showTooltips}
                  onChange={(e) => handleChange('ui', 'showTooltips', e.target.checked)}
                />
                Show tooltips
              </label>
            </div>
          </section>
        </div>

        <div className="settings-footer">
          <button className="btn-secondary" onClick={handleReset}>
            Reset to Defaults
          </button>
          <button className="btn-primary" onClick={handleSave}>
            Save Settings
          </button>
        </div>
      </div>
    </>
  )
}
