(function () {
  const qs = (sel) => document.querySelector(sel);
  const halo = qs('#halo');
  const haloState = qs('#haloState');
  const haloPulse = qs('#haloPulse');
  const chatForm = qs('#chatForm');
  const chatInput = qs('#chatInput');
  const sendButton = qs('#sendButton');
  const timeline = qs('#actionTimeline');
  const thoughtRibbon = qs('#thoughtRibbon');
  const resetThoughtsButton = qs('#resetThoughts');
  const toggleThoughtDetailsButton = qs('#toggleThoughtDetails');
  const explainList = qs('#explainList');
  const toast = qs('#toast');
  const ambientGradient = qs('#ambientGradient');
  const ambientGlimmer = qs('#ambientGlimmer');
  const dataCapsule = qs('#dataCapsule');
  const capsuleSubtitle = qs('#capsuleSubtitle');
  const confidenceValue = qs('#confidenceValue');
  const metricList = qs('#metricList');
  const suggestBar = qs('#suggestBar');
  const suggestPrimary = qs('#suggestPrimary');
  const suggestAlt = qs('#suggestAlt');
  const conversationScroll = qs('#conversationScroll');
  const conversationList = qs('#conversationList');
  const conversationEmpty = qs('#conversationEmpty');
  const modeToggle = qs('#modeToggle');
  const modeButtons = modeToggle ? Array.from(modeToggle.querySelectorAll('.mode-toggle__button')) : [];
  const llmSelector = qs('#llmSelector');
  const artifactsPanel = qs('#artifactsPanel');
  const artifactGrid = qs('#artifactGrid');
  const artifactCount = qs('#artifactCount');
  const artifactDetailTitle = qs('#artifactDetailTitle');
  const artifactDetailSummary = qs('#artifactDetailSummary');
  const artifactDetailMetrics = qs('#artifactDetailMetrics');
  const artifactDetailCitations = qs('#artifactDetailCitations');
  const artifactSpeakButton = qs('#artifactSpeak');
  const artifactRefreshButton = qs('#artifactRefresh');
  const patchPanel = qs('#patchPanel');
  const patchSummary = qs('#patchSummary');
  const patchMeta = qs('#patchMeta');
  const patchTabs = patchPanel ? Array.from(patchPanel.querySelectorAll('.patch-panel__tab')) : [];
  const diffLeft = qs('#diffLeft');
  const diffRight = qs('#diffRight');
  const diffFile = qs('#diffFile');
  const diffToggles = patchPanel ? Array.from(patchPanel.querySelectorAll('.diff-view__toggle')) : [];
  const hunkList = qs('#hunkList');
  const findingsList = qs('#findingsList');
  const gateRiskLabel = qs('#gateRisk');
  const upgradePanel = qs('#upgradePanel');
  const upgradeOffersEl = qs('#upgradeOffers');
  const upgradePlanEl = qs('#upgradePlan');
  const upgradePlanTitle = qs('#upgradePlanTitle');
  const upgradePlanMeta = qs('#upgradePlanMeta');
  const upgradePlanSteps = qs('#upgradePlanSteps');
  const upgradePlanCandidates = qs('#upgradePlanCandidates');
  const upgradeRefreshButton = qs('#upgradeRefresh');
  const upgradeClearButton = qs('#upgradeClear');
  const memoryPanel = qs('#memoryPanel');
  const memoryChips = qs('#memoryChips');
  const memoryEmpty = qs('#memoryEmpty');
  const memoryDrawer = qs('#memoryDrawer');
  const memoryDrawerOpen = qs('#memoryDrawerOpen');
  const memoryDrawerClose = qs('#memoryDrawerClose');
  const memoryDrawerBackdrop = qs('#memoryDrawerBackdrop');
  const memoryDrawerFilters = qs('#memoryDrawerFilters');
  const memoryDrawerSearch = qs('#memoryDrawerSearch');
  const memoryDrawerList = qs('#memoryDrawerList');
  const healthTilesRoot = qs('#healthTiles');
  const healthLog = qs('#healthLog');
  const runHealthcheckButton = qs('#runHealthcheck');
  const clearHealthLogButton = qs('#clearHealthLog');
  const settingsPanel = qs('#settingsPanel');
  const settingsApplyButton = qs('#settingsApply');
  const settingsResetButton = qs('#settingsReset');
  const settingsVoiceBackend = qs('#settingsVoiceBackend');
  const settingsVoiceRate = qs('#settingsVoiceRate');
  const settingsDevice = qs('#settingsDevice');
  const settingsHotkey = qs('#settingsHotkey');
  const settingsOffline = qs('#settingsOffline');
  const settingsAutospeak = qs('#settingsAutospeak');
  const learningPanel = qs('#learningPanel');
  const learningTimelineEl = qs('#learningTimeline');
  const learningDetailEl = qs('#learningDetail');
  const learningDetailTitle = qs('#learningDetailTitle');
  const learningDetailSummary = qs('#learningDetailSummary');
  const learningDetailMeta = qs('#learningDetailMeta');
  const learningDetailDiff = qs('#learningDetailDiff');
  const learningRefreshButton = qs('#learningRefresh');
  const learningClearButton = qs('#learningClear');

  const prefersContrast = window.matchMedia('(prefers-contrast: more)');
  const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
  const prefersReducedTransparency = window.matchMedia('(prefers-reduced-transparency: reduce)');

  artifactSpeakButton?.setAttribute('disabled', '');
  diffToggles.forEach((btn) => {
    btn.classList.toggle('is-active', btn.dataset.toggle === 'side-by-side');
  });

  const state = {
    phase: 'standby',
    confidence: 0,
    palette: 'analytical',
    listening: false,
    thoughtOrder: [],
    thoughts: new Map(),
    thoughtMeta: new Map(),
    pendingConfidence: null,
    confidenceDrivers: [],
    interactionMode: 'talk',
    detailsPinned: false,
    artifacts: [],
    selectedArtifactId: null,
    patchVisible: false,
    patchTab: 'overview',
    patchDiffMode: 'side-by-side',
    patchFile: null,
    patchHunks: [],
    patchSelectedHunks: new Set(),
    memorySession: [],
    memoryDrawer: {
      open: false,
      facts: [],
      filters: ['session', 'long', 'pinned', 'expiring'],
      activeFilter: 'session',
      search: '',
    },
    upgradeOffers: [],
    upgradePlan: null,
    upgradeCandidates: {},
    upgradeSelectedId: null,
    learningEvents: [],
    learningSelectedId: null,
    learningDetails: {},
    healthTiles: {},
    healthLogs: [],
    settingsOptions: {
      voiceBackends: [],
      devices: [],
    },
    settingsValues: {
      voiceBackend: '',
      voiceRate: '',
      device: '',
      hotkey: '',
      offline: false,
      autospeak: false,
    },
    pendingUserEcho: null,
    llm: {
      roles: [],
    },
    telemetry: null,
  };
  let standbyTimer = null;

  const paletteMap = {
    analytical: {
      gradient: 'radial-gradient(1400px 1100px at 46% 36%, rgba(91, 226, 255, 0.26), transparent 72%), radial-gradient(980px 860px at 72% 66%, rgba(42, 138, 210, 0.3), transparent 78%), radial-gradient(760px 640px at 32% 70%, rgba(32, 120, 190, 0.26), transparent 80%), linear-gradient(165deg, rgba(14, 46, 74, 0.6), transparent 66%)',
      accent: '#57e8ff',
      accentSoft: 'rgba(87, 232, 255, 0.48)',
      accentAlt: '#00ffe1',
    },
    creative: {
      gradient: 'radial-gradient(1400px 1100px at 48% 38%, rgba(255, 182, 140, 0.32), transparent 70%), radial-gradient(960px 820px at 74% 68%, rgba(196, 82, 48, 0.26), transparent 78%), radial-gradient(780px 660px at 28% 72%, rgba(136, 54, 68, 0.24), transparent 80%), linear-gradient(165deg, rgba(70, 28, 28, 0.55), transparent 66%)',
      accent: '#ff9f6b',
      accentSoft: 'rgba(255, 159, 107, 0.48)',
      accentAlt: '#ffad90',
    },
    ops: {
      gradient: 'radial-gradient(1400px 1100px at 46% 36%, rgba(84, 236, 198, 0.28), transparent 68%), radial-gradient(960px 840px at 70% 68%, rgba(42, 152, 148, 0.26), transparent 78%), radial-gradient(760px 660px at 28% 72%, rgba(32, 112, 118, 0.24), transparent 80%), linear-gradient(165deg, rgba(20, 72, 68, 0.55), transparent 64%)',
      accent: '#4ef0c8',
      accentSoft: 'rgba(78, 240, 200, 0.45)',
      accentAlt: '#67ffd7',
    },
  };

  function escapeHtml(str) {
    return (str || '').replace(/[&<>"']/g, (c) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    })[c]);
  }

  function sendCommand(type, payload) {
    if (window.nerion && typeof window.nerion.send === 'function') {
      window.nerion.send(type, payload || {});
    }
  }

  function addMediaListener(query, handler) {
    if (!query) {
      return;
    }
    if (typeof query.addEventListener === 'function') {
      query.addEventListener('change', handler);
    } else if (typeof query.addListener === 'function') {
      query.addListener(handler);
    }
  }

  function applyEnvironmentPreferences() {
    document.body.dataset.contrast = prefersContrast.matches ? 'high' : 'normal';
    document.body.dataset.scheme = prefersDarkScheme.matches ? 'dark' : 'light';
    document.body.dataset.transparency = prefersReducedTransparency.matches ? 'reduced' : 'full';
  }

  function updateLayoutDensity() {
    const { innerWidth: width, innerHeight: height } = window;
    const density = width >= 1680 && height <= 960 ? 'compact' : 'standard';
    document.body.dataset.density = density;
  }

  function scheduleStandby(delay = 8000) {
    clearTimeout(standbyTimer);
    if (!delay || delay <= 0) {
      return;
    }
    standbyTimer = setTimeout(() => {
      if (!state.listening) {
        setPhase('standby');
      }
    }, delay);
  }

  function setPhase(phase) {
    state.phase = phase;
    if (phase) {
      document.body.dataset.phase = phase;
    }
    if (haloState) {
      haloState.textContent = (phase || 'listening').replace(/_/g, ' ').toUpperCase();
    }
    if (halo) {
      halo.classList.toggle('halo--thinking', phase === 'thinking' || phase === 'acting');
    }
    if (haloPulse) {
      haloPulse.style.animationDuration = phase === 'listening' ? '2.4s' : '3.4s';
    }
    if (phase && phase !== 'standby') {
      scheduleStandby(phase === 'explaining' ? 6000 : 8000);
    } else if (phase === 'standby') {
      clearTimeout(standbyTimer);
    }
  }

  function setInteractionMode(mode) {
    if (!mode || mode === state.interactionMode) {
      return;
    }
    state.interactionMode = mode;
    document.body.dataset.interaction = mode;
    modeButtons.forEach((btn) => {
      const isActive = btn.dataset.mode === mode;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-selected', String(isActive));
    });
    if (chatInput) {
      chatInput.placeholder = mode === 'chat'
        ? 'Type your instruction and press Enter'
        : 'Summarize the meeting notes';
    }
    if (mode === 'chat') {
      endListening();
      if (chatInput) {
        chatInput.focus();
      }
      if (state.phase === 'standby') {
        setPhase('standby');
      }
    } else if (mode === 'talk') {
      if (chatInput) {
        chatInput.blur();
      }
      setPhase('standby');
    }
  }

  function setConfidence(value, drivers) {
    const clamped = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
    state.confidence = clamped;
    state.pendingConfidence = clamped;
    state.confidenceDrivers = Array.isArray(drivers) ? drivers.slice() : [];
    if (confidenceValue) {
      confidenceValue.textContent = clamped > 0 ? `${Math.round(clamped * 100)}%` : '—';
    }
    updateExplainability(drivers);
  }

  function setPalette(kind) {
    if (!kind || state.palette === kind) {
      return;
    }
    const preset = paletteMap[kind];
    if (preset && ambientGradient) {
      ambientGradient.style.background = preset.gradient;
      document.body.dataset.palette = kind;
      if (preset.accent) {
        document.documentElement.style.setProperty('--accent', preset.accent);
      }
      if (preset.accentSoft) {
        document.documentElement.style.setProperty('--accent-soft', preset.accentSoft);
      }
      if (preset.accentAlt) {
        document.documentElement.style.setProperty('--accent-alt', preset.accentAlt);
      }
    } else {
      document.body.dataset.palette = 'analytical';
      document.documentElement.style.setProperty('--accent', paletteMap.analytical.accent);
      document.documentElement.style.setProperty('--accent-soft', paletteMap.analytical.accentSoft);
      document.documentElement.style.setProperty('--accent-alt', paletteMap.analytical.accentAlt);
    }
    state.palette = kind;
  }

  function appendTimeline(event) {
    if (!timeline) {
      return;
    }
    const entry = document.createElement('div');
    entry.className = 'rail__item';
    const time = document.createElement('div');
    time.className = 'rail__item-time';
    const when = new Date();
    time.textContent = when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const dot = document.createElement('div');
    dot.className = 'rail__item-dot';
    const body = document.createElement('div');
    body.innerHTML = `<strong>${escapeHtml(event.title || event.kind || '')}</strong><br>${escapeHtml(event.summary || event.detail || '')}`;
    entry.append(time, dot, body);
    timeline.prepend(entry);
    const items = timeline.querySelectorAll('.rail__item');
    if (items.length > 24) {
      items[items.length - 1].remove();
    }
  }

  function ensureThought(id) {
    if (!id) {
      return null;
    }
    if (state.thoughts.has(id)) {
      return state.thoughts.get(id);
    }
    const node = document.createElement('li');
    node.className = 'thought-node';
    node.dataset.id = id;
    node.innerHTML = '<div class="thought-node__orb"></div><h4 class="thought-node__title"></h4><p class="thought-node__detail"></p><div class="thought-node__card"><p class="thought-node__card-title"></p><p class="thought-node__card-body"></p></div>';
    state.thoughts.set(id, node);
    state.thoughtOrder.push(id);
    thoughtRibbon?.appendChild(node);
    return node;
  }

  function updateThoughtNode(payload) {
    const id = payload.id || payload.step_id;
    const node = ensureThought(id);
    if (!node) {
      return;
    }
    const title = node.querySelector('.thought-node__title');
    const detail = node.querySelector('.thought-node__detail');
    const cardTitle = node.querySelector('.thought-node__card-title');
    const cardBody = node.querySelector('.thought-node__card-body');
    if (title) {
      title.textContent = payload.label || payload.title || '…';
    }
    if (detail) {
      detail.textContent = payload.why || payload.detail || '';
    }
    if (cardTitle) {
      cardTitle.textContent = (payload.label || payload.title || 'Step').toUpperCase();
    }
    if (cardBody) {
      cardBody.textContent = payload.detail || payload.why || (payload.status ? `Status: ${payload.status}` : '');
    }
    state.thoughtMeta.set(id, {
      label: payload.label || payload.title || id,
      status: payload.status,
      reason: payload.why || payload.detail || '',
    });
    node.classList.toggle('thought-node--active', payload.status === 'running' || payload.status === 'active');
    node.classList.toggle('thought-node--done', payload.status === 'done' || payload.status === 'complete');
    if (state.detailsPinned) {
      const lines = state.thoughtOrder.map((tid) => {
        const meta = state.thoughtMeta.get(tid);
        const status = meta?.status ? ` · ${meta.status}` : '';
        return meta ? `${meta.label}${status}: ${meta.reason || '—'}` : tid;
      });
      updateExplainability(lines, { force: true });
    }
  }

  function clearThoughts() {
    state.thoughtOrder = [];
    state.thoughts.clear();
    state.thoughtMeta.clear();
    state.detailsPinned = false;
    if (thoughtRibbon) {
      thoughtRibbon.innerHTML = '';
    }
  }

  function updateExplainability(items, options = {}) {
    if (!explainList) {
      return;
    }
    const force = options.force === true;
    const isArray = Array.isArray(items);
    if (!force && (!isArray || items.length === 0) && explainList.children.length > 0) {
      return;
    }
    const list = isArray ? items : [];
    if (state.detailsPinned && !force) {
      return;
    }
    explainList.innerHTML = '';
    list.forEach((item) => {
      const li = document.createElement('li');
      li.textContent = item;
      explainList.appendChild(li);
    });
  }

  function updateMetrics(metrics) {
    if (!metricList) {
      return;
    }
    metricList.innerHTML = '';
    (metrics || []).forEach((metric) => {
      const dt = document.createElement('dt');
      dt.textContent = metric.label || '';
      const dd = document.createElement('dd');
      dd.textContent = metric.value || '';
      metricList.append(dt, dd);
    });
  }

  function renderLlmSelector() {
    if (!llmSelector) {
      return;
    }
    const roles = state.llm && Array.isArray(state.llm.roles) ? state.llm.roles : [];
    if (!roles.length) {
      llmSelector.innerHTML = '';
      llmSelector.setAttribute('hidden', '');
      return;
    }
    llmSelector.removeAttribute('hidden');
    llmSelector.innerHTML = '';
    roles.forEach((role) => {
      const roleId = role.role || 'chat';
      const container = document.createElement('div');
      container.className = 'llm-selector__role';
      container.dataset.role = roleId;
      if (role.overridden) {
        container.classList.add('llm-selector__role--overridden');
      }

      const labelEl = document.createElement('label');
      const selectId = `llmSelect-${roleId}`;
      labelEl.className = 'llm-selector__label';
      labelEl.htmlFor = selectId;
      labelEl.textContent = role.label || roleId;
      container.appendChild(labelEl);

      const select = document.createElement('select');
      select.className = 'llm-selector__select';
      select.id = selectId;
      select.dataset.role = roleId;

      const autoOption = document.createElement('option');
      autoOption.value = '';
      const autoLabel = role.default_label || role.default || 'default';
      autoOption.textContent = autoLabel && autoLabel !== 'Auto' ? `Auto — ${autoLabel}` : 'Auto';
      select.appendChild(autoOption);

      (role.options || []).forEach((opt) => {
        if (!opt || !opt.id) {
          return;
        }
        const optionEl = document.createElement('option');
        optionEl.value = opt.id;
        optionEl.textContent = opt.label || opt.id;
        if (opt.note) {
          optionEl.title = opt.note;
        }
        select.appendChild(optionEl);
      });

      if (role.overridden && role.active) {
        select.value = role.active;
      } else {
        select.value = '';
      }

      container.appendChild(select);

      const status = document.createElement('p');
      status.className = 'llm-selector__status';
      if (role.overridden && role.active_label) {
        status.textContent = `Using ${role.active_label}`;
      } else {
        status.textContent = `Using default (${role.default_label || role.default || '—'})`;
      }
      container.appendChild(status);

      llmSelector.appendChild(container);
    });
  }

  function appendConversationEntry(role, text) {
    if (!conversationList) {
      return null;
    }
    const clean = escapeHtml(text || '').replace(/\n/g, '<br />');
    const entry = document.createElement('div');
    entry.className = `conversation__entry conversation__entry--${role}`;

    const wrapper = document.createElement('div');
    const meta = document.createElement('div');
    meta.className = 'conversation__entry-meta';
    meta.textContent = role === 'user' ? 'You' : 'Nerion';
    const body = document.createElement('p');
    body.className = 'conversation__entry-text';
    body.innerHTML = clean;
    wrapper.append(meta, body);

    if (role === 'assistant') {
      const haloNode = document.createElement('div');
      haloNode.className = 'conversation__halo';
      const score = state.pendingConfidence ?? state.confidence;
      const ratio = Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : 0;
      haloNode.style.setProperty('--confidence', ratio.toString());
      const label = document.createElement('span');
      label.textContent = ratio > 0 ? `${Math.round(ratio * 100)}%` : '—';
      if (state.confidenceDrivers.length) {
        const tooltip = document.createElement('div');
        tooltip.className = 'conversation__halo-tooltip';
        tooltip.innerHTML = state.confidenceDrivers.map((d) => `<div>${escapeHtml(d)}</div>`).join('');
        haloNode.appendChild(tooltip);
      }
      haloNode.append(label);
      entry.append(haloNode, wrapper);
      state.pendingConfidence = null;
      state.confidenceDrivers = [];
    } else {
      entry.append(wrapper);
    }

    addMessageAndScroll(conversationScroll || conversationList, conversationList, entry);
    return entry;
  }

  function addMessageAndScroll(scroller, list, entry) {
    if (!list || !scroller) {
      return;
    }
    const threshold = 10;
    const pinned = Math.abs((scroller.scrollHeight - scroller.scrollTop) - scroller.clientHeight) < threshold;

    list.append(entry);
    const nodes = list.children;
    while (nodes.length > 300) {
      list.removeChild(list.firstChild);
    }

    if (conversationEmpty) {
      conversationEmpty.classList.toggle('hide', list.children.length > 0);
    }

    if (pinned) {
      scroller.scrollTop = scroller.scrollHeight;
    }
  }

  function selectArtifact(id, emit = true) {
    state.selectedArtifactId = id;
    renderArtifacts();
    if (emit) {
      sendCommand('artifact', { action: 'select', id });
    }
  }

  function renderArtifactDetail(artifact) {
    if (!artifactDetailTitle || !artifactDetailSummary || !artifactDetailMetrics || !artifactDetailCitations) {
      return;
    }
    if (!artifact) {
      artifactDetailTitle.textContent = 'No artifact selected';
      artifactDetailSummary.textContent = 'Incoming research, code diffs, and notes will appear here.';
      artifactDetailMetrics.innerHTML = '';
      artifactDetailCitations.innerHTML = '';
      artifactSpeakButton?.setAttribute('disabled', '');
      return;
    }
    artifactDetailTitle.textContent = artifact.title || artifact.id || 'Artifact';
    artifactDetailSummary.textContent = artifact.summary || artifact.description || 'No summary provided.';
    artifactSpeakButton?.removeAttribute('disabled');

    const metrics = Array.isArray(artifact.metrics) ? artifact.metrics : [];
    artifactDetailMetrics.innerHTML = '';
    metrics.forEach((metric) => {
      const chip = document.createElement('span');
      chip.className = 'artifact-detail__metric';
      const label = document.createElement('span');
      label.textContent = metric.label || 'metric';
      const bar = document.createElement('span');
      bar.className = 'artifact-detail__bar';
      const value = Number(metric.value);
      if (Number.isFinite(value)) {
        bar.style.setProperty('--progress', String(Math.max(0, Math.min(1, value))));
      }
      chip.append(label, bar);
      artifactDetailMetrics.appendChild(chip);
    });

    const citations = Array.isArray(artifact.citations) ? artifact.citations : [];
    artifactDetailCitations.innerHTML = '';
    if (!citations.length) {
      const item = document.createElement('li');
      item.textContent = 'No citations available.';
      artifactDetailCitations.appendChild(item);
    } else {
      citations.forEach((cite) => {
        const item = document.createElement('li');
        item.textContent = cite;
        artifactDetailCitations.appendChild(item);
      });
    }
  }

  function renderArtifacts() {
    if (!artifactGrid || !artifactsPanel) {
      return;
    }
    const items = Array.isArray(state.artifacts) ? state.artifacts : [];
    artifactsPanel.hidden = items.length === 0;
    if (artifactCount) {
      artifactCount.textContent = `${items.length} item${items.length === 1 ? '' : 's'}`;
    }
    artifactGrid.innerHTML = '';
    let selectedArtifact = null;
    items.forEach((artifact) => {
      const id = artifact.id || artifact.title || String(Math.random());
      const card = document.createElement('article');
      card.className = 'artifact-card';
      card.dataset.id = id;
      if (state.selectedArtifactId && state.selectedArtifactId === id) {
        card.classList.add('is-selected');
        selectedArtifact = artifact;
      }
      const kind = document.createElement('span');
      kind.className = 'artifact-card__kind';
      kind.textContent = (artifact.kind || 'artifact').toUpperCase();
      const title = document.createElement('h4');
      title.className = 'artifact-card__title';
      title.textContent = artifact.title || 'Untitled artifact';
      const summary = document.createElement('p');
      summary.className = 'artifact-card__summary';
      summary.textContent = artifact.summary || artifact.description || '—';
      const metrics = document.createElement('div');
      metrics.className = 'artifact-card__metrics';
      const chips = Array.isArray(artifact.metrics) ? artifact.metrics.slice(0, 2) : [];
      chips.forEach((metric) => {
        const chip = document.createElement('span');
        chip.className = 'artifact-chip';
        const spark = document.createElement('span');
        spark.className = 'artifact-chip__spark';
        if (metric.gradient) {
          spark.style.setProperty('--spark-gradient', metric.gradient);
        } else if (Number.isFinite(metric.value)) {
          const pct = Math.max(0, Math.min(1, Number(metric.value)));
          spark.style.setProperty('--spark-gradient', `linear-gradient(90deg, rgba(87, 232, 255, 0.9) ${pct * 100}%, rgba(87, 232, 255, 0) ${pct * 100}%)`);
        }
        const text = document.createElement('span');
        text.textContent = metric.label || String(metric.value ?? 'metric');
        chip.append(spark, text);
        metrics.appendChild(chip);
      });
      card.append(kind, title, summary, metrics);
      card.addEventListener('click', () => selectArtifact(id));
      card.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectArtifact(id);
        }
      });
      card.setAttribute('tabindex', '0');
      artifactGrid.appendChild(card);
    });
    if (!state.selectedArtifactId && items.length) {
      state.selectedArtifactId = items[0].id || items[0].title;
      selectedArtifact = items[0];
    }
    renderArtifactDetail(selectedArtifact);
  }

  function setPatchTab(tab) {
    state.patchTab = tab;
    patchTabs.forEach((btn) => {
      const isActive = btn.dataset.tab === tab;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-selected', String(isActive));
    });
    if (!patchPanel) {
      return;
    }
    patchPanel.querySelectorAll('[data-panel]').forEach((panel) => {
      const el = panel;
      const name = el.getAttribute('data-panel');
      el.classList.toggle('is-active', name === tab);
    });
  }

  function showPatchPanel(show) {
    if (!patchPanel) {
      return;
    }
    state.patchVisible = !!show;
    patchPanel.hidden = !show;
  }

  function updatePatchOverview(payload) {
    if (!patchSummary || !patchMeta) {
      return;
    }
    patchSummary.textContent = payload.summary || 'No patch summary available.';
    patchMeta.innerHTML = '';
    const metaItems = Array.isArray(payload.meta) ? payload.meta : [];
    metaItems.forEach((item) => {
      const chip = document.createElement('span');
      chip.className = 'patch-panel__meta-chip';
      chip.textContent = item;
      patchMeta.appendChild(chip);
    });
  }

  function updatePatchDiff(payload) {
    if (!diffLeft || !diffRight || !diffFile) {
      return;
    }
    const file = payload.file || '—';
    diffFile.textContent = file;
    diffLeft.textContent = payload.left || '';
    diffRight.textContent = payload.right || '';
    state.patchFile = file;
    diffToggles.forEach((btn) => btn.classList.toggle('is-active', btn.dataset.toggle === state.patchDiffMode));
    const hunks = Array.isArray(payload.hunks) ? payload.hunks : [];
    state.patchHunks = hunks;
    if (hunkList) {
      hunkList.innerHTML = '';
      hunks.forEach((hunk) => {
        const id = hunk.id || hunk.range || String(hunk.start || Math.random());
        const chip = document.createElement('button');
        chip.className = 'hunk-chip';
        chip.dataset.hunkId = id;
        chip.textContent = hunk.label || id;
        chip.addEventListener('click', () => {
          const selected = state.patchSelectedHunks.has(id);
          if (selected) {
            state.patchSelectedHunks.delete(id);
          } else {
            state.patchSelectedHunks.add(id);
          }
          chip.classList.toggle('is-selected', !selected);
          sendCommand('patch', {
            action: 'toggle_hunk',
            hunk_id: id,
            file: state.patchFile,
            selected: !selected,
          });
        });
        hunkList.appendChild(chip);
      });
    }
  }

  function updatePatchFindings(payload) {
    if (!findingsList || !gateRiskLabel) {
      return;
    }
    const findings = Array.isArray(payload.findings) ? payload.findings : [];
    findingsList.innerHTML = '';
    findings.forEach((entry) => {
      const li = document.createElement('li');
      li.className = 'findings-list__item';
      li.textContent = entry; // entry should already be readable text
      findingsList.appendChild(li);
    });
    gateRiskLabel.textContent = payload.risk || 'No risk';
  }

  function renderUpgradeOffers() {
    if (!upgradeOffersEl) {
      return;
    }
    const offers = Array.isArray(state.upgradeOffers) ? state.upgradeOffers : [];
    upgradeOffersEl.innerHTML = '';
    if (!offers.length) {
      const empty = document.createElement('p');
      empty.className = 'memory__empty';
      empty.textContent = 'No upgrade offers pending.';
      upgradeOffersEl.appendChild(empty);
      return;
    }
    offers.forEach((offer) => {
      const id = offer.id || Math.random().toString(36).slice(2);
      const card = document.createElement('article');
      card.className = 'upgrade-offer';
      card.dataset.offerId = id;
      if (state.upgradePlan && state.upgradePlan.id === id) {
        card.classList.add('is-selected');
      }
      const score = offer.score !== undefined ? `${Math.round(Number(offer.score) * 100)}% score` : '— score';
      card.innerHTML = `
        <div class="upgrade-offer__header">
          <h4 class="upgrade-offer__title">${offer.title || offer.summary || 'Upgrade offer'}</h4>
          <span class="upgrade-offer__score">${score}</span>
        </div>
        <p class="upgrade-offer__why">${offer.why || 'The agent flagged a candidate self-improvement.'}</p>
        <div class="upgrade-offer__actions">
          <button class="upgrade-offer__action" data-action="preview">Preview</button>
          <button class="upgrade-offer__action" data-action="safe_apply">Safe apply</button>
          <button class="upgrade-offer__action" data-action="defer">Defer</button>
          <button class="upgrade-offer__action" data-action="dismiss">Dismiss</button>
        </div>
      `;
      upgradeOffersEl.appendChild(card);
    });
  }

  function renderUpgradePlan() {
    if (!upgradePlanEl || !upgradePlanTitle || !upgradePlanMeta || !upgradePlanSteps || !upgradePlanCandidates) {
      return;
    }
    const plan = state.upgradePlan;
    if (!plan) {
      upgradePlanEl.hidden = true;
      return;
    }
    upgradePlanEl.hidden = false;
    upgradePlanTitle.textContent = plan.summary || 'Plan summary';
    const meta = [`Source: ${plan.source || 'LLM'}`, plan.files ? `${plan.files.length} files` : null, plan.estimate ? `ETA ${plan.estimate}` : null].filter(Boolean).join(' · ');
    upgradePlanMeta.textContent = meta || 'Review steps and candidate runs.';
    upgradePlanSteps.innerHTML = '';
    const steps = Array.isArray(plan.steps) ? plan.steps : [];
    steps.forEach((step) => {
      const li = document.createElement('li');
      li.className = 'upgrade-step';
      if (step.status === 'active') li.classList.add('is-active');
      if (step.status === 'done') li.classList.add('is-done');
      li.innerHTML = `
        <span class="upgrade-step__label">${step.label || step.action || 'Step'}</span>
        <span class="upgrade-step__status">${(step.status || 'pending').toUpperCase()}</span>
      `;
      upgradePlanSteps.appendChild(li);
    });
    upgradePlanCandidates.innerHTML = '';
    const candidates = state.upgradeCandidates[plan.id] || [];
    candidates.forEach((candidate) => {
      const div = document.createElement('div');
      div.className = 'upgrade-candidate';
      div.innerHTML = `
        <span class="upgrade-candidate__label">${candidate.label || candidate.id || 'candidate'}</span>
        <span class="upgrade-candidate__status">${candidate.status || (candidate.passed ? 'PASS' : 'PENDING')}</span>
      `;
      upgradePlanCandidates.appendChild(div);
    });
  }

  function openMemoryDrawer() {
    state.memoryDrawer.open = true;
    renderMemoryDrawer();
    sendCommand('memory', { action: 'drawer', filter: state.memoryDrawer.activeFilter });
  }

  function closeMemoryDrawer() {
    state.memoryDrawer.open = false;
    renderMemoryDrawer();
  }

  function sendMemoryCommand(action, factId, value) {
    if (!factId) {
      return;
    }
    sendCommand('memory', {
      action,
      fact_id: factId,
      value,
      filter: state.memoryDrawer.activeFilter,
    });
  }

  function renderMemoryChips() {
    if (!memoryChips || !memoryEmpty) {
      return;
    }
    const items = Array.isArray(state.memorySession) ? state.memorySession : [];
    memoryChips.innerHTML = '';
    memoryEmpty.hidden = items.length > 0;
    items.forEach((fact) => {
      const id = fact.id || fact.fact || Math.random().toString(36).slice(2);
      const chip = document.createElement('article');
      chip.className = 'memory__chip';
      chip.dataset.factId = id;
      chip.dataset.pinned = fact.pinned ? '1' : '';
      const confidence = fact.confidence !== undefined ? `${Math.round(Number(fact.confidence) * 100)}% confidence` : '';
      const lastUsed = fact.last_used ? `last used ${fact.last_used}` : 'new';
      chip.innerHTML = `
        <div class="memory__chip-fact">${escapeHtml(fact.fact || fact.text || '—')}</div>
        <div class="memory__chip-meta">${(fact.scope || 'session').toUpperCase()} · ${confidence} · ${lastUsed}</div>
        <div class="memory__chip-actions">
          <button class="memory__chip-button" data-action="${fact.pinned ? 'unpin' : 'pin'}">${fact.pinned ? 'Unpin' : 'Pin'}</button>
          <button class="memory__chip-button" data-action="forget">Forget</button>
          <button class="memory__chip-button" data-action="edit">Edit</button>
        </div>
      `;
      memoryChips.appendChild(chip);
    });
  }

  function filterDrawerFacts() {
    const facts = Array.isArray(state.memoryDrawer.facts) ? state.memoryDrawer.facts : [];
    const scope = state.memoryDrawer.activeFilter;
    const query = (state.memoryDrawer.search || '').toLowerCase();
    return facts.filter((fact) => {
      const scopeValue = (fact.scope || '').toLowerCase();
      const matchesScope = !scope || scope === 'all' || scopeValue.includes(scope);
      const searchable = `${fact.fact || fact.text || ''} ${(fact.tags || []).join(' ')}`.toLowerCase();
      const matchesSearch = !query || searchable.includes(query);
      return matchesScope && matchesSearch;
    });
  }

  function renderMemoryDrawer() {
    if (!memoryDrawer) {
      return;
    }
    if (memoryDrawerFilters) {
      memoryDrawerFilters.querySelectorAll('.memory-drawer__filter').forEach((btn) => {
        const scope = btn.dataset.scope;
        btn.classList.toggle('is-active', scope === state.memoryDrawer.activeFilter);
      });
    }
    if (memoryDrawerSearch && memoryDrawerSearch.value !== state.memoryDrawer.search) {
      memoryDrawerSearch.value = state.memoryDrawer.search || '';
    }
    if (memoryDrawerList) {
      memoryDrawerList.innerHTML = '';
      const items = filterDrawerFacts();
      if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'memory-drawer__item';
        empty.textContent = 'No memories match the current filters.';
        memoryDrawerList.appendChild(empty);
      } else {
        items.forEach((fact) => {
          const id = fact.id || fact.fact || Math.random().toString(36).slice(2);
          const tags = Array.isArray(fact.tags) ? fact.tags.join(', ') : '';
          const lastUsed = fact.last_used ? `Last used ${fact.last_used}` : 'New';
          const entry = document.createElement('div');
          entry.className = 'memory-drawer__item';
          entry.dataset.factId = id;
          entry.dataset.pinned = fact.pinned ? '1' : '';
          entry.innerHTML = `
            <div class="memory-drawer__item-header">
              <div class="memory-drawer__item-title">${(fact.scope || 'session').toUpperCase()} · ${tags || 'untagged'}</div>
              <div class="memory-drawer__item-tags">${lastUsed}</div>
            </div>
            <div class="memory-drawer__item-body">${escapeHtml(fact.fact || fact.text || '—')}</div>
            <div class="memory-drawer__item-actions">
              <button class="memory-drawer__item-button" data-action="${fact.pinned ? 'unpin' : 'pin'}">${fact.pinned ? 'Unpin' : 'Pin'}</button>
              <button class="memory-drawer__item-button" data-action="forget">Forget</button>
              <button class="memory-drawer__item-button" data-action="edit">Edit</button>
            </div>
          `;
          memoryDrawerList.appendChild(entry);
        });
      }
    }
    if (state.memoryDrawer.open) {
      memoryDrawer?.removeAttribute('hidden');
      document.body.dataset.memoryDrawer = 'open';
    } else {
      if (memoryDrawer) {
        memoryDrawer.hidden = true;
      }
      delete document.body.dataset.memoryDrawer;
    }
  }

  function selectUpgradeOffer(offerId, emit = true) {
    const offers = Array.isArray(state.upgradeOffers) ? state.upgradeOffers : [];
    const offer = offers.find((item) => (item.id || item.offer_id) === offerId);
    if (!offer) {
      return;
    }
    state.upgradeSelectedId = offerId;
    if (!state.upgradePlan || state.upgradePlan.id !== offerId) {
      state.upgradePlan = null;
    }
    renderUpgradeOffers();
    renderUpgradePlan();
    if (emit) {
      sendCommand('upgrade', { action: 'preview', offer_id: offerId });
    }
  }

  function renderUpgradePlan() {
    if (!upgradePlanEl || !upgradePlanTitle || !upgradePlanMeta || !upgradePlanSteps || !upgradePlanCandidates) {
      return;
    }
    const plan = state.upgradePlan;
    if (!plan) {
      upgradePlanEl.hidden = true;
      return;
    }
    upgradePlanEl.hidden = false;
    state.upgradeSelectedId = plan.id;
    upgradePlanTitle.textContent = plan.summary || plan.title || 'Self-code plan';
    const metaParts = [];
    if (plan.source) metaParts.push(`Source: ${plan.source}`);
    if (Array.isArray(plan.files)) metaParts.push(`${plan.files.length} files`);
    if (plan.estimate) metaParts.push(`ETA ${plan.estimate}`);
    upgradePlanMeta.textContent = metaParts.join(' · ') || 'Review steps and candidate runs.';
    upgradePlanSteps.innerHTML = '';
    const steps = Array.isArray(plan.steps) ? plan.steps : [];
    steps.forEach((step) => {
      const li = document.createElement('li');
      li.className = 'upgrade-step';
      if (step.status === 'active') li.classList.add('is-active');
      if (step.status === 'done') li.classList.add('is-done');
      li.innerHTML = `
        <span class="upgrade-step__label">${step.label || step.action || 'Step'}</span>
        <span class="upgrade-step__status">${(step.status || 'pending').toUpperCase()}</span>
      `;
      upgradePlanSteps.appendChild(li);
    });
    upgradePlanCandidates.innerHTML = '';
    const results = state.upgradeCandidates[plan.id] || [];
    results.forEach((candidate) => {
      const div = document.createElement('div');
      div.className = 'upgrade-candidate';
      const status = candidate.status || (candidate.passed === true ? 'PASS' : candidate.passed === false ? 'FAIL' : 'PENDING');
      div.innerHTML = `
        <span class="upgrade-candidate__label">${candidate.label || candidate.id || 'candidate'}</span>
        <span class="upgrade-candidate__status">${status.toUpperCase()}</span>
      `;
      upgradePlanCandidates.appendChild(div);
    });
  }

  function renderLearningTimeline() {
    if (!learningTimelineEl) {
      return;
    }
    const events = Array.isArray(state.learningEvents) ? state.learningEvents : [];
    learningTimelineEl.innerHTML = '';
    if (!events.length) {
      const empty = document.createElement('p');
      empty.className = 'memory__empty';
      empty.textContent = 'No learning events recorded yet.';
      learningTimelineEl.appendChild(empty);
      learningDetailEl?.setAttribute('hidden', '');
      return;
    }
    if (!state.learningSelectedId) {
      state.learningSelectedId = events[0].id || events[0].key || events[0].timestamp;
    }
    events.forEach((event) => {
      const id = event.id || event.key || event.timestamp || Math.random().toString(36).slice(2);
      const card = document.createElement('article');
      card.className = 'learning-event';
      card.dataset.eventId = id;
      if (state.learningSelectedId === id) {
        card.classList.add('is-selected');
      }
      const scope = (event.scope || 'session').toUpperCase();
      const source = (event.source || 'agent').toUpperCase();
      const value = event.value || event.new_value || '—';
      const oldValue = event.old_value || '';
      card.innerHTML = `
        <div class="learning-event__header">
          <span>${scope}</span>
          <span>${event.timestamp || ''}</span>
        </div>
        <div class="learning-event__body">${escapeHtml(event.summary || `${event.key || 'preference'} → ${value}`)}</div>
        <div class="learning-event__meta">
          <span>${source}</span>
          ${event.confidence !== undefined ? `<span>${Math.round(Number(event.confidence) * 100)}% confidence</span>` : ''}
        </div>
      `;
      learningTimelineEl.appendChild(card);
    });
    renderLearningDetail();
  }

  function renderLearningDetail() {
    if (!learningDetailEl || !state.learningSelectedId) {
      learningDetailEl?.setAttribute('hidden', '');
      return;
    }
    const events = Array.isArray(state.learningEvents) ? state.learningEvents : [];
    const event = events.find((e) => (e.id || e.key || e.timestamp) === state.learningSelectedId);
    if (!event) {
      learningDetailEl.setAttribute('hidden', '');
      return;
    }
    learningDetailEl.removeAttribute('hidden');
    if (learningDetailTitle) {
      learningDetailTitle.textContent = event.summary || event.key || 'Preference change';
    }
    if (learningDetailSummary) {
      learningDetailSummary.textContent = event.details || event.description || `${event.key || 'value'} → ${event.value || event.new_value}`;
    }
    if (learningDetailMeta) {
      learningDetailMeta.innerHTML = '';
      const metaParts = [
        event.scope ? `Scope: ${event.scope}` : null,
        event.source ? `Source: ${event.source}` : null,
        event.timestamp ? `Timestamp: ${event.timestamp}` : null,
        event.confidence !== undefined ? `Confidence: ${Math.round(Number(event.confidence) * 100)}%` : null,
      ].filter(Boolean);
      metaParts.forEach((line) => {
        const li = document.createElement('li');
        li.textContent = line;
        learningDetailMeta.appendChild(li);
      });
    }
    if (learningDetailDiff) {
      const detail = state.learningDetails[state.learningSelectedId];
      if (detail && detail.diff) {
        learningDetailDiff.textContent = detail.diff;
      } else if (event.old_value || event.value) {
        learningDetailDiff.textContent = `- ${event.old_value || '—'}\n+ ${event.value || event.new_value || '—'}`;
      } else {
        learningDetailDiff.textContent = 'No diff available.';
      }
    }
  }

  function selectLearningEvent(eventId, emit = true) {
    state.learningSelectedId = eventId;
    renderLearningTimeline();
    renderLearningDetail();
    if (emit && eventId) {
      sendCommand('learning', { action: 'select', event_id: eventId });
    }
  }

  function renderUpgradeOffers() {
    if (!upgradeOffersEl) {
      return;
    }
    const offers = Array.isArray(state.upgradeOffers) ? state.upgradeOffers : [];
    upgradeOffersEl.innerHTML = '';
    if (!offers.length) {
      const empty = document.createElement('p');
      empty.className = 'memory__empty';
      empty.textContent = 'No upgrade offers pending.';
      upgradeOffersEl.appendChild(empty);
      upgradePlanEl?.setAttribute('hidden', '');
      return;
    }
    offers.forEach((offer) => {
      const id = offer.id || offer.offer_id || Math.random().toString(36).slice(2);
      const card = document.createElement('article');
      card.className = 'upgrade-offer';
      card.dataset.offerId = id;
      if (state.upgradePlan && state.upgradePlan.id === id) {
        card.classList.add('is-selected');
      }
      const score = offer.score !== undefined ? `${Math.round(Number(offer.score) * 100)}% score` : (offer.risk ? offer.risk.toUpperCase() : '—');
      card.innerHTML = `
        <div class="upgrade-offer__header">
          <h4 class="upgrade-offer__title">${offer.title || offer.summary || 'Upgrade offer'}</h4>
          <span class="upgrade-offer__score">${score}</span>
        </div>
        <p class="upgrade-offer__why">${offer.why || offer.description || 'The agent flagged a candidate self-improvement.'}</p>
        <div class="upgrade-offer__actions">
          <button class="upgrade-offer__action" data-action="preview">Preview</button>
          <button class="upgrade-offer__action" data-action="safe_apply">Safe apply</button>
          <button class="upgrade-offer__action" data-action="defer">Defer</button>
          <button class="upgrade-offer__action" data-action="dismiss">Dismiss</button>
        </div>
      `;
      upgradeOffersEl.appendChild(card);
    });
  }

  function updateHealthTiles(payload = {}) {
    if (!healthTilesRoot) {
      return;
    }
    const updates = payload.tiles || payload;
    state.healthTiles = { ...state.healthTiles, ...updates };
    Object.entries(state.healthTiles).forEach(([key, data]) => {
      const tile = healthTilesRoot.querySelector(`[data-tile="${key}"]`);
      if (!tile || !data) {
        return;
      }
      const statusEl = tile.querySelector('.health-tile__status');
      const valueEl = tile.querySelector('.health-tile__value');
      const noteEl = tile.querySelector('.health-tile__note');
      if (statusEl) {
        statusEl.textContent = data.status || '—';
      }
      if (valueEl) {
        valueEl.textContent = data.value || data.metric || '—';
      }
      if (noteEl) {
        noteEl.textContent = data.note || data.summary || '';
      }
    });
  }

  function appendHealthLog(entry) {
    if (!healthLog) {
      return;
    }
    const text = typeof entry === 'string' ? entry : (entry.message || JSON.stringify(entry));
    state.healthLogs.push(text);
    const div = document.createElement('div');
    div.className = 'health__log-entry';
    div.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
    healthLog.appendChild(div);
    healthLog.scrollTop = healthLog.scrollHeight;
  }

  function renderSettingsOptions(payload = {}) {
    state.settingsOptions = { ...state.settingsOptions, ...payload };
    if (settingsVoiceBackend && Array.isArray(state.settingsOptions.voiceBackends)) {
      settingsVoiceBackend.innerHTML = '';
      state.settingsOptions.voiceBackends.forEach((item) => {
        const option = document.createElement('option');
        option.value = item.value || item.id || item;
        option.textContent = item.label || item.name || item.value || item;
        settingsVoiceBackend.appendChild(option);
      });
    }
    if (settingsDevice && Array.isArray(state.settingsOptions.devices)) {
      settingsDevice.innerHTML = '';
      state.settingsOptions.devices.forEach((item) => {
        const option = document.createElement('option');
        option.value = item.value || item.id || item;
        option.textContent = item.label || item.name || item.value || item;
        settingsDevice.appendChild(option);
      });
    }
  }

  function renderSettingsValues(values = {}) {
    state.settingsValues = { ...state.settingsValues, ...values };
    if (settingsVoiceBackend && state.settingsValues.voiceBackend) {
      settingsVoiceBackend.value = state.settingsValues.voiceBackend;
    }
    if (settingsVoiceRate && state.settingsValues.voiceRate) {
      settingsVoiceRate.value = state.settingsValues.voiceRate;
    }
    if (settingsDevice && state.settingsValues.device) {
      settingsDevice.value = state.settingsValues.device;
    }
    if (settingsHotkey && state.settingsValues.hotkey !== undefined) {
      settingsHotkey.textContent = state.settingsValues.hotkey || 'Capture';
    }
    if (settingsOffline) {
      settingsOffline.checked = !!state.settingsValues.offline;
    }
    if (settingsAutospeak) {
      settingsAutospeak.checked = !!state.settingsValues.autospeak;
    }
  }

  function updateSuggestions(primary, secondary) {
    if (primary && suggestPrimary) {
      suggestPrimary.textContent = primary;
      suggestPrimary.classList.remove('hide');
    } else if (suggestPrimary) {
      suggestPrimary.classList.add('hide');
    }
    if (secondary && suggestAlt) {
      suggestAlt.innerHTML = '';
      secondary.forEach((label) => {
        const chip = document.createElement('span');
        chip.className = 'suggest__chip';
        chip.textContent = label;
        suggestAlt.appendChild(chip);
      });
    }
  }

  function showToast(message) {
    if (!toast) {
      return;
    }
    toast.textContent = message;
    toast.classList.remove('hide');
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => toast.classList.add('hide'), 2400);
  }
  showToast._timer = null;

  function handleEvent(event) {
    if (!event) {
      return;
    }
    const { type } = event;
    const payload = event.payload !== undefined ? event.payload : event;
    switch (type) {
      case 'state': {
        setPhase(payload.phase || payload.mode);
        if (payload.intent_palette) {
          setPalette(payload.intent_palette);
        }
        if (payload.interaction_mode) {
          setInteractionMode(payload.interaction_mode);
        }
        if (payload.reset_thoughts) {
          clearThoughts();
        }
        break;
      }
      case 'thought_step': {
        updateThoughtNode(payload);
        break;
      }
      case 'confidence': {
        setConfidence(Number(payload.value ?? payload.score), payload.drivers || payload.explain || []);
        break;
      }
      case 'action_log': {
        appendTimeline({
          title: payload.kind,
          summary: payload.summary || payload.detail,
        });
        break;
      }
      case 'chat_turn': {
        const role = payload.role === 'user' ? 'user' : 'assistant';
        const text = payload.text || '';
        if (role === 'user') {
          const pending = state.pendingUserEcho;
          const isMatch = pending && pending.text === text;
          if (isMatch && pending.entry) {
            pending.entry.dataset.pending = 'false';
            delete pending.entry.dataset.pending;
            delete pending.entry.dataset.textSignature;
          } else if (!text) {
            // nothing to render
          } else {
            appendConversationEntry('user', text);
          }
          state.pendingUserEcho = null;
        } else {
          appendConversationEntry('assistant', text);
        }
        appendTimeline({ title: role === 'user' ? 'User' : 'Agent', summary: text });
        if (role === 'user') {
          setPhase('thinking');
        } else {
          setPhase('explaining');
          scheduleStandby(6000);
        }
        break;
      }
      case 'chat_ack': {
        appendConversationEntry('assistant', payload.text || '');
        setPhase('explaining');
        scheduleStandby(6000);
        break;
      }
      case 'suggestion': {
        updateSuggestions(payload.primary, payload.secondary || []);
        break;
      }
      case 'metrics': {
        updateMetrics(payload.items || []);
        if (payload.subtitle && capsuleSubtitle) {
          capsuleSubtitle.textContent = payload.subtitle;
        }
        if (payload.telemetry) {
          state.telemetry = payload.telemetry;
          const anomalyLines = Array.isArray(payload.telemetry.anomalies) ? payload.telemetry.anomalies : [];
          if (!state.detailsPinned && anomalyLines.length) {
            updateExplainability(anomalyLines);
          }
        }
        break;
      }
      case 'llm_options': {
        state.llm = {
          roles: Array.isArray(payload.roles) ? payload.roles : [],
        };
        renderLlmSelector();
        break;
      }
      case 'memory_session': {
        state.memorySession = Array.isArray(payload.items) ? payload.items : [];
        renderMemoryChips();
        break;
      }
      case 'memory_drawer': {
        state.memoryDrawer = {
          ...state.memoryDrawer,
          facts: Array.isArray(payload.facts) ? payload.facts : [],
          filters: Array.isArray(payload.filters) && payload.filters.length ? payload.filters : state.memoryDrawer.filters,
          activeFilter: payload.active_filter || state.memoryDrawer.activeFilter,
          search: payload.search !== undefined ? payload.search : state.memoryDrawer.search,
        };
        if (payload.open !== undefined) {
          state.memoryDrawer.open = !!payload.open;
        }
        renderMemoryDrawer();
        break;
      }
      case 'memory_update': {
        renderMemoryChips();
        appendHealthLog(`Memory ${payload.action || 'update'}: ${payload.fact || ''}`.trim());
        break;
      }
      case 'upgrade_offer': {
        const offers = Array.isArray(payload.offers)
          ? payload.offers
          : Array.isArray(payload)
            ? payload
            : [];
        state.upgradeOffers = offers;
        renderUpgradeOffers();
        if (!offers.length) {
          state.upgradePlan = null;
          renderUpgradePlan();
        }
        break;
      }
      case 'selfcode_plan': {
        const plan = payload.plan || payload;
        if (!plan || !plan.id) {
          break;
        }
        if (!Array.isArray(state.upgradeOffers) || !state.upgradeOffers.some((offer) => offer.id === plan.id)) {
          state.upgradeOffers = [...(state.upgradeOffers || []), { id: plan.id, title: plan.summary || plan.title, why: plan.why || '', score: plan.score }];
        }
        state.upgradePlan = plan;
        state.upgradeCandidates[plan.id] = plan.candidates || state.upgradeCandidates[plan.id] || [];
        renderUpgradeOffers();
        renderUpgradePlan();
        break;
      }
      case 'selfcode_candidate_result': {
        const planId = payload.plan_id || payload.id;
        if (!planId) {
          break;
        }
        const list = state.upgradeCandidates[planId] || [];
        let entry = list.find((item) => item.id === payload.candidate_id);
        if (!entry) {
          entry = { id: payload.candidate_id };
          list.push(entry);
        }
        entry.label = payload.label || entry.label || payload.candidate_id;
        entry.status = payload.status || (payload.passed === true ? 'PASS' : payload.passed === false ? 'FAIL' : 'PENDING');
        entry.passed = payload.passed;
        state.upgradeCandidates[planId] = list;
        if (state.upgradePlan && state.upgradePlan.id === planId) {
          renderUpgradePlan();
        }
        break;
      }
      case 'upgrade_clear': {
        state.upgradeOffers = [];
        state.upgradePlan = null;
        renderUpgradeOffers();
        renderUpgradePlan();
        break;
      }
      case 'learning_timeline': {
        state.learningEvents = Array.isArray(payload.events) ? payload.events : [];
        if (payload.selected_id) {
          state.learningSelectedId = payload.selected_id;
        }
        state.learningDetails = {};
        renderLearningTimeline();
        break;
      }
      case 'learning_event': {
        const events = Array.isArray(state.learningEvents) ? state.learningEvents : [];
        const updated = payload.event || payload;
        if (updated && (updated.id || updated.key)) {
          const id = updated.id || updated.key;
          const idx = events.findIndex((e) => (e.id || e.key) === id);
          if (idx >= 0) {
            events[idx] = { ...events[idx], ...updated };
          } else {
            events.unshift(updated);
          }
          state.learningEvents = events;
          state.learningSelectedId = id;
        }
        renderLearningTimeline();
        break;
      }
      case 'learning_diff': {
        if (payload && payload.event_id) {
          state.learningDetails[payload.event_id] = payload;
          state.learningSelectedId = state.learningSelectedId || payload.event_id;
          renderLearningDetail();
        }
        break;
      }
      case 'learning_clear': {
        state.learningEvents = [];
        state.learningSelectedId = null;
        state.learningDetails = {};
        renderLearningTimeline();
        renderLearningDetail();
        break;
      }
      case 'health_status': {
        updateHealthTiles(payload);
        break;
      }
      case 'health_log': {
        appendHealthLog(payload.message || payload);
        break;
      }
      case 'settings_options': {
        renderSettingsOptions(payload);
        break;
      }
      case 'settings_values': {
        renderSettingsValues(payload);
        break;
      }
      case 'settings_hotkey': {
        renderSettingsValues({ hotkey: payload.hotkey });
        break;
      }
      case 'artifact_list': {
        const items = Array.isArray(payload.items) ? payload.items : [];
        state.artifacts = items;
        if (items.length) {
          const first = items[0];
          state.selectedArtifactId = first.id || first.title || null;
        } else {
          state.selectedArtifactId = null;
        }
        renderArtifacts();
        break;
      }
      case 'artifact_select': {
        const id = payload.id || payload.artifact_id;
        if (id) {
          selectArtifact(id, false);
        }
        break;
      }
      case 'artifact_detail': {
        const artifact = payload.artifact;
        if (artifact) {
          state.selectedArtifactId = artifact.id || artifact.title || state.selectedArtifactId;
          const list = state.artifacts.slice();
          const idx = list.findIndex((item) => (item.id || item.title) === state.selectedArtifactId);
          if (idx >= 0) {
            list[idx] = { ...list[idx], ...artifact };
            state.artifacts = list;
          }
          renderArtifacts();
        }
        break;
      }
      case 'patch_overview': {
        showPatchPanel(true);
        setPatchTab('overview');
        state.patchSelectedHunks.clear();
        updatePatchOverview(payload);
        if (payload.findings) {
          updatePatchFindings(payload);
        }
        break;
      }
      case 'patch_diff': {
        showPatchPanel(true);
        setPatchTab('diff');
        state.patchSelectedHunks.clear();
        updatePatchDiff(payload);
        break;
      }
      case 'patch_findings': {
        showPatchPanel(true);
        setPatchTab('findings');
        updatePatchFindings(payload);
        break;
      }
      case 'patch_clear': {
        state.patchSelectedHunks.clear();
        showPatchPanel(false);
        break;
      }
      case 'speak_start': {
        setPhase('acting');
        break;
      }
      case 'speak_stop': {
        setPhase('explaining');
        break;
      }
      case 'audio_state': {
        if (!payload) {
          break;
        }
        if (payload.status === 'speaking') {
          setPhase('acting');
        } else if (payload.status === 'idle') {
          if (state.interactionMode === 'talk') {
            setPhase('explaining');
            scheduleStandby(payload.delay || 5000);
          }
        }
        break;
      }
      case 'error': {
        showToast(`${payload.code || 'Error'}: ${payload.message || ''}`);
        break;
      }
      default:
        break;
    }
  }

  function beginListening() {
    if (state.interactionMode !== 'talk') {
      return;
    }
    if (state.listening) {
      return;
    }
    state.listening = true;
    setPhase('listening');
    sendCommand('ptt', { state: 'pressed', source: 'ui' });
  }

  function endListening() {
    if (!state.listening) {
      return;
    }
    state.listening = false;
    sendCommand('ptt', { state: 'released', source: 'ui' });
    if (state.interactionMode === 'talk') {
      setPhase('thinking');
    }
  }

  function submitChat(event) {
    if (event) {
      event.preventDefault();
    }
    const value = (chatInput && chatInput.value.trim()) || '';
    if (!value) {
      return;
    }
    const entry = appendConversationEntry('user', value);
    if (entry) {
      entry.dataset.pending = 'true';
      entry.dataset.textSignature = value;
    }
    state.pendingUserEcho = entry ? { text: value, entry } : { text: value, entry: null };
    sendCommand('chat', { text: value, speak: false });
    chatInput.value = '';
    setPhase('thinking');
  }

  function wireInteractions() {
    if (halo) {
      halo.addEventListener('mousedown', beginListening);
      halo.addEventListener('mouseup', endListening);
      halo.addEventListener('mouseleave', endListening);
      halo.addEventListener('touchstart', (ev) => {
        ev.preventDefault();
        beginListening();
      }, { passive: false });
      halo.addEventListener('touchend', (ev) => {
        ev.preventDefault();
        endListening();
      });
    }

    window.addEventListener('keydown', (ev) => {
      if (ev.code === 'Space' && state.interactionMode === 'talk' && !state.listening) {
        ev.preventDefault();
        beginListening();
      }
      if (ev.code === 'Escape') {
        endListening();
        if (state.memoryDrawer.open) {
          closeMemoryDrawer();
        }
      }
    });
    window.addEventListener('keyup', (ev) => {
      if (ev.code === 'Space') {
        ev.preventDefault();
        endListening();
      }
    });

    chatForm?.addEventListener('submit', submitChat);
    sendButton?.addEventListener('click', submitChat);
    resetThoughtsButton?.addEventListener('click', () => clearThoughts());
    resetThoughtsButton?.addEventListener('click', () => {
      if (toggleThoughtDetailsButton) {
        toggleThoughtDetailsButton.dataset.state = '';
        toggleThoughtDetailsButton.textContent = 'Details';
      }
      updateExplainability([]);
    });
    toggleThoughtDetailsButton?.addEventListener('click', () => {
      const expanded = toggleThoughtDetailsButton.dataset.state === 'expanded';
      toggleThoughtDetailsButton.dataset.state = expanded ? '' : 'expanded';
      toggleThoughtDetailsButton.textContent = expanded ? 'Details' : 'Collapse';
      state.detailsPinned = !expanded;
      if (expanded) {
        updateExplainability([], { force: true });
        return;
      }
      const lines = state.thoughtOrder.map((id) => {
        const meta = state.thoughtMeta.get(id);
        const status = meta?.status ? ` · ${meta.status}` : '';
        return meta ? `${meta.label}${status}: ${meta.reason || '—'}` : id;
      });
      updateExplainability(lines, { force: true });
    });
    suggestPrimary?.addEventListener('click', () => {
      const label = suggestPrimary.textContent || '';
      sendCommand('override', { action: 'accept_suggestion', label });
      showToast(`Accepted: ${label}`);
    });
    llmSelector?.addEventListener('change', (event) => {
      const select = event.target.closest('.llm-selector__select');
      if (!select || !(select instanceof HTMLSelectElement)) {
        return;
      }
      const role = select.dataset.role;
      if (!role) {
        return;
      }
      const provider = select.value;
      sendCommand('llm', { action: 'set', role, provider: provider || null });
    });
    modeButtons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const next = btn.dataset.mode === 'chat' ? 'chat' : 'talk';
        setInteractionMode(next);
        sendCommand('mode', { mode: next });
      });
    });
    artifactSpeakButton?.addEventListener('click', () => {
      if (!state.selectedArtifactId) {
        return;
      }
      sendCommand('artifact', { action: 'speak', id: state.selectedArtifactId });
    });
    artifactRefreshButton?.addEventListener('click', () => {
      sendCommand('artifact', { action: 'refresh' });
    });
    learningRefreshButton?.addEventListener('click', () => {
      sendCommand('learning', { action: 'refresh' });
    });
    learningClearButton?.addEventListener('click', () => {
      state.learningEvents = [];
      state.learningSelectedId = null;
      renderLearningTimeline();
      renderLearningDetail();
      sendCommand('learning', { action: 'clear' });
    });
    learningTimelineEl?.addEventListener('click', (event) => {
      const card = event.target.closest('.learning-event');
      if (!card) {
        return;
      }
      const id = card.dataset.eventId;
      if (id) {
        selectLearningEvent(id);
      }
    });
    learningDetailEl?.addEventListener('click', (event) => {
      const button = event.target.closest('.learning-detail__action');
      if (!button || !state.learningSelectedId) {
        return;
      }
      const action = button.dataset.action;
      if (!action) {
        return;
      }
      if (action === 'scope') {
        const newScope = window.prompt('Set scope (session/long/pinned)', 'session');
        if (!newScope) {
          return;
        }
        sendCommand('learning', { action: 'set_scope', event_id: state.learningSelectedId, scope: newScope });
      } else {
        sendCommand('learning', { action, event_id: state.learningSelectedId });
      }
    });
    upgradeRefreshButton?.addEventListener('click', () => {
      sendCommand('upgrade', { action: 'refresh' });
    });
    upgradeClearButton?.addEventListener('click', () => {
      sendCommand('upgrade', { action: 'clear' });
      state.upgradeOffers = [];
      state.upgradePlan = null;
      renderUpgradeOffers();
      renderUpgradePlan();
    });
    upgradeOffersEl?.addEventListener('click', (event) => {
      const actionBtn = event.target.closest('.upgrade-offer__action');
      const card = event.target.closest('.upgrade-offer');
      if (!card) {
        return;
      }
      const offerId = card.dataset.offerId;
      if (!offerId) {
        return;
      }
      if (actionBtn) {
        const action = actionBtn.dataset.action;
        if (!action) {
          return;
        }
        if (action === 'preview') {
          selectUpgradeOffer(offerId);
        } else {
          sendCommand('upgrade', { action, offer_id: offerId });
        }
      } else {
        selectUpgradeOffer(offerId);
      }
    });
    upgradePlanEl?.addEventListener('click', (event) => {
      const button = event.target.closest('.upgrade-plan__action');
      if (!button || !state.upgradePlan) {
        return;
      }
      const action = button.dataset.action;
      if (!action) {
        return;
      }
      sendCommand('selfcode', { action, plan_id: state.upgradePlan.id });
    });
    memoryDrawerOpen?.addEventListener('click', () => openMemoryDrawer());
    memoryDrawerClose?.addEventListener('click', () => closeMemoryDrawer());
    memoryDrawerBackdrop?.addEventListener('click', () => closeMemoryDrawer());
    memoryDrawerFilters?.addEventListener('click', (event) => {
      const btn = event.target.closest('.memory-drawer__filter');
      if (!btn) {
        return;
      }
      const scope = btn.dataset.scope || 'session';
      state.memoryDrawer.activeFilter = scope;
      renderMemoryDrawer();
    });
    memoryDrawerSearch?.addEventListener('input', (event) => {
      state.memoryDrawer.search = event.target.value;
      renderMemoryDrawer();
    });
    memoryDrawerList?.addEventListener('click', (event) => {
      const button = event.target.closest('[data-action]');
      if (!button) {
        return;
      }
      const item = event.target.closest('.memory-drawer__item');
      const factId = item?.dataset.factId;
      const action = button.dataset.action;
      if (!factId || !action) {
        return;
      }
      if (action === 'edit') {
        const current = item.querySelector('.memory-drawer__item-body')?.textContent || '';
        const updated = window.prompt('Edit memory', current);
        if (updated && updated !== current) {
          sendMemoryCommand('edit', factId, updated);
        }
        return;
      }
      sendMemoryCommand(action, factId);
    });
    memoryChips?.addEventListener('click', (event) => {
      const button = event.target.closest('.memory__chip-button');
      if (!button) {
        return;
      }
      const chip = event.target.closest('.memory__chip');
      const factId = chip?.dataset.factId;
      const action = button.dataset.action;
      if (!factId || !action) {
        return;
      }
      if (action === 'edit') {
        const current = chip.querySelector('.memory__chip-fact')?.textContent || '';
        const updated = window.prompt('Edit memory', current);
        if (updated && updated !== current) {
          sendMemoryCommand('edit', factId, updated);
        }
        return;
      }
      sendMemoryCommand(action, factId);
    });
    runHealthcheckButton?.addEventListener('click', () => {
      sendCommand('health', { action: 'run' });
      appendHealthLog('Healthcheck requested…');
    });
    clearHealthLogButton?.addEventListener('click', () => {
      state.healthLogs = [];
      if (healthLog) {
        healthLog.innerHTML = '';
      }
    });
    thoughtRibbon?.addEventListener('click', (event) => {
      const node = event.target.closest('.thought-node');
      if (!node) {
        return;
      }
      const id = node.dataset.id;
      if (!id) {
        return;
      }
      node.classList.toggle('thought-node--expanded');
      const meta = state.thoughtMeta.get(id);
      if (meta) {
        updateExplainability([
          `${meta.label}: ${meta.reason || meta.status || 'pending'}`,
        ]);
      }
    });
    patchTabs.forEach((tabBtn) => {
      tabBtn.addEventListener('click', () => {
        const tab = tabBtn.dataset.tab || 'overview';
        setPatchTab(tab);
      });
    });
    diffToggles.forEach((toggle) => {
      toggle.addEventListener('click', () => {
        const mode = toggle.dataset.toggle || 'side-by-side';
        state.patchDiffMode = mode;
        diffToggles.forEach((btn) => btn.classList.toggle('is-active', btn.dataset.toggle === mode));
        patchPanel?.setAttribute('data-diff-mode', mode);
        sendCommand('patch', { action: 'set_diff_mode', mode });
      });
    });
    patchPanel?.addEventListener('click', (event) => {
      const actionBtn = event.target.closest('.patch-panel__action');
      if (actionBtn) {
        const action = actionBtn.dataset.action;
        if (action) {
          sendCommand('patch', { action });
        }
      }
    });
    settingsApplyButton?.addEventListener('click', () => {
      const payload = {
        voice_backend: settingsVoiceBackend?.value,
        voice_rate: settingsVoiceRate?.value,
        device: settingsDevice?.value,
        hotkey: settingsHotkey?.textContent,
        offline: settingsOffline?.checked,
        autospeak: settingsAutospeak?.checked,
      };
      sendCommand('settings', { action: 'apply', values: payload });
      showToast('Settings updated');
    });
    settingsResetButton?.addEventListener('click', () => {
      sendCommand('settings', { action: 'reset' });
    });
    settingsHotkey?.addEventListener('click', () => {
      settingsHotkey.textContent = 'Listening…';
      sendCommand('settings', { action: 'capture_hotkey' });
    });
  }

  function wireBridge() {
    if (window.nerion && typeof window.nerion.onEvent === 'function') {
      window.nerion.onEvent(handleEvent);
    }
    if (window.nerion && typeof window.nerion.onStatus === 'function') {
      window.nerion.onStatus((status) => {
        if (status && status.type === 'python_exit') {
          showToast(`Runtime exited (${status.code ?? '0'})`);
        }
      });
    }
    if (window.nerion && typeof window.nerion.ready === 'function') {
      window.nerion.ready();
    }
    if (window.nerion && typeof window.nerion.send === 'function') {
      window.nerion.send('memory', { action: 'refresh' });
      window.nerion.send('learning', { action: 'refresh' });
      window.nerion.send('upgrade', { action: 'refresh' });
      window.nerion.send('artifact', { action: 'refresh' });
      window.nerion.send('health', { action: 'status' });
      window.nerion.send('settings', { action: 'refresh' });
      window.nerion.send('llm', { action: 'refresh' });
    }
    window.Nerion = { emit: handleEvent }; // allow manual testing
  }

  document.addEventListener('DOMContentLoaded', () => {
    wireInteractions();
    wireBridge();
    setInteractionMode('talk');
    setPhase('standby');
    setPalette('analytical');
    setConfidence(0);
    renderArtifacts();
    renderUpgradeOffers();
    renderUpgradePlan();
    renderLearningTimeline();
    renderLearningDetail();
    showPatchPanel(false);
    updateSuggestions('Send summary', ['Schedule deep dive', 'Open action log']);
    renderMemoryChips();
    renderMemoryDrawer();
    if (conversationEmpty) {
      conversationEmpty.classList.toggle('hide', !!(conversationList && conversationList.children.length));
    }
    applyEnvironmentPreferences();
    updateLayoutDensity();
    addMediaListener(prefersContrast, applyEnvironmentPreferences);
    addMediaListener(prefersDarkScheme, applyEnvironmentPreferences);
    addMediaListener(prefersReducedTransparency, applyEnvironmentPreferences);
    window.addEventListener('resize', updateLayoutDensity);
    if (!window.nerion) {
      console.warn('Nerion preload bridge missing; UI will remain idle until backend connects.');
    }
  });
})();
