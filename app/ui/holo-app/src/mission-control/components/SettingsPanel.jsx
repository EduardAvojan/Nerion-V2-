import React, { useState } from 'react'
import './SettingsPanel.css'

export default function SettingsPanel({ isOpen, onClose, theme, onThemeChange }) {
  // Model catalog mapping (from config/model_catalog.yaml)
  const modelsByProvider = {
    anthropic: {
      chat: [
        { value: 'claude-sonnet-4-5-20250929', label: 'Claude Sonnet 4.5' },
        { value: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4' },
        { value: 'claude-3-7-sonnet-20250219', label: 'Claude 3.7 Sonnet' },
        { value: 'claude-opus-4-1-20250805', label: 'Claude Opus 4.1' },
        { value: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku' }
      ],
      code: [
        { value: 'claude-sonnet-4-5-20250929', label: 'Claude Sonnet 4.5' },
        { value: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4' },
        { value: 'claude-3-7-sonnet-20250219', label: 'Claude 3.7 Sonnet' },
        { value: 'claude-opus-4-1-20250805', label: 'Claude Opus 4.1' },
        { value: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku' }
      ],
      planner: [
        { value: 'claude-sonnet-4-5-20250929', label: 'Claude Sonnet 4.5' },
        { value: 'claude-sonnet-4-20250514', label: 'Claude Sonnet 4' },
        { value: 'claude-3-7-sonnet-20250219', label: 'Claude 3.7 Sonnet' },
        { value: 'claude-opus-4-1-20250805', label: 'Claude Opus 4.1' }
      ]
    },
    openai: {
      chat: [
        { value: 'gpt-5', label: 'GPT-5' },
        { value: 'gpt-4', label: 'GPT-4' }
      ],
      code: [
        { value: 'gpt-5', label: 'GPT-5' },
        { value: 'gpt-4', label: 'GPT-4' }
      ],
      planner: []
    },
    google: {
      chat: [
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' }
      ],
      code: [
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' }
      ],
      planner: [
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' }
      ]
    },
    vertexai: {
      chat: [
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
        { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
        { value: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite' },
        { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' }
      ],
      code: [
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
        { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
        { value: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash Lite' },
        { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' }
      ],
      planner: [
        { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
        { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
        { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' }
      ]
    }
  }

  const [settings, setSettings] = useState({
    voice: {
      enabled: true,
      provider: 'openai',
      model: 'whisper-1'
    },
    providers: {
      chat: {
        provider: 'anthropic',
        model: 'claude-sonnet-4-5-20250929'
      },
      code: {
        provider: 'anthropic',
        model: 'claude-sonnet-4-5-20250929'
      },
      planner: {
        provider: 'anthropic',
        model: 'claude-sonnet-4-5-20250929'
      }
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

  const handleProviderChange = (role, key, value) => {
    setSettings(prev => ({
      ...prev,
      providers: {
        ...prev.providers,
        [role]: {
          ...prev.providers[role],
          [key]: value
        }
      }
    }))
  }

  const handleSave = () => {
    console.log('Saving settings:', settings)

    // Send provider settings to backend via IPC
    if (window.nerion && window.nerion.send) {
      window.nerion.send('save-settings', {
        providers: settings.providers,
        voice: settings.voice,
        network: settings.network,
        learning: settings.learning,
        immune: settings.immune,
        ui: settings.ui
      })
    }

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

          {/* Chat Provider Settings */}
          <section className="settings-section">
            <h3>💬 Chat Provider</h3>
            <p className="section-description">For Genesis conversations and voice interactions</p>
            <div className="setting-item">
              <label>Provider</label>
              <select
                value={settings.providers.chat.provider}
                onChange={(e) => {
                  const newProvider = e.target.value
                  const firstModel = modelsByProvider[newProvider]?.chat[0]?.value || ''
                  handleProviderChange('chat', 'provider', newProvider)
                  if (firstModel) {
                    handleProviderChange('chat', 'model', firstModel)
                  }
                }}
              >
                <option value="anthropic">Anthropic Claude</option>
                <option value="openai">OpenAI GPT</option>
                <option value="google">Google Gemini</option>
              </select>
            </div>
            <div className="setting-item">
              <label>Model</label>
              <select
                value={settings.providers.chat.model}
                onChange={(e) => handleProviderChange('chat', 'model', e.target.value)}
              >
                {modelsByProvider[settings.providers.chat.provider]?.chat.map(model => (
                  <option key={model.value} value={model.value}>{model.label}</option>
                ))}
              </select>
            </div>
          </section>

          {/* Code Provider Settings */}
          <section className="settings-section">
            <h3>💻 Code Provider</h3>
            <p className="section-description">For code generation and analysis</p>
            <div className="setting-item">
              <label>Provider</label>
              <select
                value={settings.providers.code.provider}
                onChange={(e) => {
                  const newProvider = e.target.value
                  const firstModel = modelsByProvider[newProvider]?.code[0]?.value || ''
                  handleProviderChange('code', 'provider', newProvider)
                  if (firstModel) {
                    handleProviderChange('code', 'model', firstModel)
                  }
                }}
              >
                <option value="anthropic">Anthropic Claude</option>
                <option value="openai">OpenAI GPT</option>
                <option value="google">Google Gemini</option>
              </select>
            </div>
            <div className="setting-item">
              <label>Model</label>
              <select
                value={settings.providers.code.model}
                onChange={(e) => handleProviderChange('code', 'model', e.target.value)}
              >
                {modelsByProvider[settings.providers.code.provider]?.code.map(model => (
                  <option key={model.value} value={model.value}>{model.label}</option>
                ))}
              </select>
            </div>
          </section>

          {/* Planner Provider Settings */}
          <section className="settings-section">
            <h3>🎯 Planner Provider</h3>
            <p className="section-description">For task planning and decomposition</p>
            <div className="setting-item">
              <label>Provider</label>
              <select
                value={settings.providers.planner.provider}
                onChange={(e) => {
                  const newProvider = e.target.value
                  const firstModel = modelsByProvider[newProvider]?.planner[0]?.value || ''
                  handleProviderChange('planner', 'provider', newProvider)
                  if (firstModel) {
                    handleProviderChange('planner', 'model', firstModel)
                  }
                }}
              >
                <option value="anthropic">Anthropic Claude</option>
                <option value="google">Google Gemini</option>
              </select>
            </div>
            <div className="setting-item">
              <label>Model</label>
              <select
                value={settings.providers.planner.model}
                onChange={(e) => handleProviderChange('planner', 'model', e.target.value)}
                disabled={modelsByProvider[settings.providers.planner.provider]?.planner.length === 0}
              >
                {modelsByProvider[settings.providers.planner.provider]?.planner.length > 0 ? (
                  modelsByProvider[settings.providers.planner.provider]?.planner.map(model => (
                    <option key={model.value} value={model.value}>{model.label}</option>
                  ))
                ) : (
                  <option value="">No models available for this provider</option>
                )}
              </select>
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
