# Nerion Project - Complete System Documentation

**Last Updated:** October 25, 2025 (Database: 1004 agent-generated lessons across 10 languages)

---

## 📋 Documentation & Change Tracking

**This file (CLAUDE.md):** Single source of truth for understanding Nerion's architecture, goals, and current state.

**Change History:** See [CHANGELOG.md](./CHANGELOG.md) for detailed, timestamped history of all confirmed, tested, and verified changes.

**Important:** CHANGELOG.md only contains changes that have been tested and confirmed working. Experimental changes, failed attempts, or in-progress work are NOT logged until verified.

---

## 🎯 What is Nerion? (Complete Vision)

**Nerion is not a code quality tool—it's a biological immune system for software that operates autonomously at PhD-level.**

### The Core Metaphor

Imagine if your codebase had a **living immune system** like your body:
- **Permanently integrated** - Not an external tool you run, but a resident organism
- **Continuously learning** - Gets smarter every day about YOUR specific code
- **Proactive & autonomous** - Monitors 24/7, not on-demand
- **Self-healing** - Detects and fixes issues automatically
- **Evolves** - Recursively improves its own capabilities

That's Nerion.

### End Goals (The North Star)

1. **90% Code Quality Classification** - Distinguish good code from bad with PhD-level accuracy
2. **Autonomous Bug Prevention** - Catch issues before they reach production (git hooks)
3. **Institutional Knowledge Capture** - Permanently preserve senior developer expertise
4. **Continuous Evolution** - Self-improve through meta-learning
5. **Zero-Latency Prevention** - Real-time feedback at commit time
6. **Codebase-Specific Mastery** - Day 1 (generic) → Day 365 (expert in YOUR code)

### How Nerion is Different

| Traditional AI Coding Tools | Nerion (Biological Immune System) |
|----------------------------|----------------------------------|
| External plugin/tool | Permanent resident in codebase |
| On-demand (you ask) | Proactive (monitors 24/7) |
| Generic training | Codebase-specific learning |
| One-shot answers | Continuous improvement |
| Text-based only | Voice-first + GUI + CLI |
| Static capabilities | Self-improving |

---

## 🏗️ System Architecture (All Components)

Nerion is a **multi-tiered autonomous developer agent** with 10+ major subsystems:

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NERION ECOSYSTEM                              │
│         "A Biological Immune System for Software"                    │
└─────────────────────────────────────────────────────────────────────┘

USER INTERFACES
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────┐
│  Mission Control GUI │  │  Voice Interface     │  │  CLI/Chat    │
│  (Electron + React)  │  │  (STT/TTS/PTT)      │  │  (Terminal)  │
│  - Training dash     │  │  - Push-to-talk     │  │  - Commands  │
│  - Status panels     │  │  - Wake word        │  │  - Intents   │
│  - Terminal PTY      │  │  - Hands-free mode  │  │  - Memory    │
└──────────┬───────────┘  └──────────┬───────────┘  └──────┬───────┘
           │                         │                      │
           └─────────────────────────┴──────────────────────┘
                                     │
                          IPC / Unix Socket / HTTP
                                     │
┌────────────────────────────────────┴─────────────────────────────────┐
│                    IMMUNE SYSTEM DAEMON (24/7)                       │
│  - Watches codebase for changes (file system monitoring)            │
│  - Runs GNN training in background                                  │
│  - Detects threats and auto-fixes (high confidence)                 │
│  - Serves real-time metrics to GUI                                  │
│  - Heartbeat: Health (5m), Learn (1h), Evolve (1d), Memory (7d)    │
└────────────────────────────────────┬─────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
          ┌─────────▼────────┐            ┌──────────▼──────────┐
          │  CHAT ENGINE     │            │  PARENT/PLANNER     │
          │  - Intent router │            │  - Task decompose   │
          │  - Command parse │            │  - Tool orchestrate │
          │  - Memory bridge │            │  - Safety policies  │
          └─────────┬────────┘            └──────────┬──────────┘
                    │                                │
                    └────────────────┬───────────────┘
                                     │
              ┌──────────────────────┴───────────────────────┐
              │                                               │
    ┌─────────▼──────────┐                       ┌──────────▼─────────┐
    │  EXECUTION LAYER   │                       │  WEB RESEARCH      │
    │                    │                       │  - Site query      │
    │  ┌──────────────┐  │                       │  - Web search      │
    │  │ DIGITAL      │  │                       │  - Doc assimilate  │
    │  │ PHYSICIST    │  │                       └────────────────────┘
    │  │ (GNN Brain)  │  │
    │  │ - 816-dim    │  │         ┌─────────────────────────────┐
    │  │ - SAGE/GCN   │  │         │  SELF-CODER ENGINE          │
    │  │ - 58.9% acc  │  │         │  - AST-based editing        │
    │  └──────────────┘  │         │  - Plan generation          │
    │                    │         │  - Safety verification      │
    │  ┌──────────────┐  │         │  - Test execution           │
    │  │ BEHAVIORAL   │  │         │  - Git integration          │
    │  │ COACH        │  │         └─────────────────────────────┘
    │  │ (Fast Path)  │  │
    │  │ - Heuristics │  │         ┌─────────────────────────────┐
    │  │ - 80% tasks  │  │         │  LEARNING SYSTEMS           │
    │  └──────────────┘  │         │  - Curriculum generation    │
    │                    │         │  - Meta-learning            │
    └────────────────────┘         │  - Shadow replay            │
                                   │  - Upgrade agent            │
                                   └─────────────────────────────┘

                    ┌──────────────────────────┐
                    │  STORAGE & MEMORY        │
                    │  - curriculum.sqlite     │
                    │  - GNN weights (.pt)     │
                    │  - Knowledge index       │
                    │  - Experience logs       │
                    │  - Session cache         │
                    └──────────────────────────┘
```

---

## 🧠 Major Components Deep Dive

### 1. Digital Physicist (Deep Learning Brain)
**Location:** `nerion_digital_physicist/`
**Purpose:** Learn the "physics" of code—understand what makes code good vs bad

**What it does:**
- Converts code → AST → Graph representation
- Classifies as "before/bad" (0) or "after/good" (1)
- Learns from 973 curriculum lessons (483 GitHub-scraped + 490 generated)
- Uses Graph Neural Networks (not sequential transformers)

**Architectures tested:**
- **GraphSAGE:** 58.9% accuracy (WINNER)
- GCN: 55.2%
- GAT: 54.8%
- GIN: 47.4%

**Current features:**
- 48-dimensional structural features (node types, edges, metrics)
- 768-dimensional semantic features (CodeBERT embeddings) - IN PROGRESS
- Total: 816 dimensions

**Training:**
- Dataset: 1,800 graphs (Oct 17 dataset, production-quality code)
- Validation: 15% split
- Early stopping with patience
- Metrics: Accuracy, ROC-AUC, F1-score

**Roadmap to 90%:**
- Phase 1: Semantic embeddings → 75-80% (IN PROGRESS)
- Phase 2: Hierarchical pooling → +5-8%
- Phase 3: Control/data flow → +3-5%
- Phase 4: Graph transformers → +2-4%
- Phase 5: Contrastive learning → +2-3%
- Phase 6: More training data → +1-3%

### 2. Self-Coder Engine
**Location:** `selfcoder/`
**Purpose:** Autonomous code modification with safety guarantees

**Workflow:**
1. User request → Intent detection
2. Generate improvement plan (LLM)
3. Policy evaluation (governor checks safety)
4. AST-based modifications (not regex)
5. Run tests automatically
6. Create git commit (if approved)

**Safety features:**
- Never modifies without approval (unless auto-fix mode + confidence >95%)
- Policy files define what's allowed (`policy/`)
- Governor tracks execution frequency
- All changes versioned in git
- Malicious code detection (`security/`)

**Components:**
- `planner/` - Multi-step plan generation
- `actions/` - Code modification primitives
- `ast_editor/` - AST-based editing
- `policy/` - Safety policies and approval gates
- `governor.py` - Rate limiting, risk assessment
- `tester/` - Auto-run tests after changes
- `vcs/` - Git integration

### 3. Behavioral Coach (Fast Path)
**Location:** `selfcoder/learning/behavioral.py`
**Purpose:** Handle 80% of common tasks with heuristics

**What it handles:**
- Variable naming conventions
- Import sorting
- Docstring formatting
- Simple refactors (list comprehensions, f-strings)
- Code style enforcement

**Why it exists:**
- GNN is "deep path" (slow, for novel problems)
- Behavioral Coach is "fast path" (instant, for common patterns)
- Best of both worlds: Speed + Intelligence

### 4. Voice Interface
**Location:** `voice/`
**Purpose:** Hands-free, voice-first development

**Components:**
- **STT** (Speech-to-Text): Whisper, Vosk, cloud providers
- **TTS** (Text-to-Speech): pyttsx3, cloud TTS
- **PTT** (Push-to-Talk): Keyboard shortcut triggering
- **Wake word:** "Hey Nerion" detection

**Usage:**
```
User: [Press PTT] "Nerion, add try-except to parse_config function"
Nerion: [TTS] "I'll add exception handling to that function"
[Generates plan, modifies code]
Nerion: [TTS] "Done. Added try-except with logging on IOError"
```

**Profiles:**
- `whisper_fast` - Local, fast, good accuracy
- `whisper_accurate` - Local, slower, best accuracy
- `cloud_stt` - Google/Azure (requires API key)
- `vosk` - Offline, lightweight

### 5. Mission Control GUI (Desktop App)
**Location:** `app/ui/holo-app/`
**Purpose:** Professional desktop interface for monitoring and control

**Technology:**
- Electron (cross-platform desktop app)
- React (UI components)
- Vite (build system)
- xterm.js (terminal emulator)
- WebSocket (real-time communication)

**Features:**
1. **Terminal** - Full bash PTY with reconnection
2. **Genesis** - Neural network visualization
3. **Training Dashboard** (5 tabs):
   - Overview - High-level metrics
   - Training Data - Dataset inspection
   - Episode History - Training logs
   - Memory Explorer - Replay buffer visualization
   - Training Logs - Real-time output
4. **Status Panels:**
   - Immune System Vitals
   - Signal Health
   - Memory Snapshot
5. **Thought Process** - AI reasoning visualization
6. **Settings** - Configuration management

**Distribution:**
- Build: `npm run build` → `Nerion.app`
- Ready for macOS .dmg and Windows .exe packaging

### 6. Immune System Daemon
**Location:** `daemon/nerion_daemon.py`
**Purpose:** 24/7 background monitoring and healing

**Biological Heartbeat:**
- **Health check:** Every 5 minutes (vital signs)
- **Learn from commits:** Every hour (new knowledge)
- **Brain evolution:** Daily (retrain GNN)
- **Memory consolidation:** Weekly (cleanup, archive)

**Responsibilities:**
- Watch codebase for changes (file system monitoring)
- Run GNN training in background
- Detect threats and auto-fix (high confidence >95%)
- Serve metrics to GUI via Unix socket (`~/.nerion/daemon.sock`)
- Learn from new commits (post-commit hook)
- Auto-restart on crash

**Communication:**
- Unix socket: JSON protocol
- GUI subscribes to updates
- LaunchAgent integration (macOS) for auto-start

### 7. Chat Engine & Parent System
**Location:** `app/chat/` and `app/parent/`
**Purpose:** Natural language interface and task orchestration

**Intent routing:**
- "health" → Run diagnostics
- "upgrade" → Self-improvement
- "search web for..." → Web research
- "fix the bug in..." → Self-coding
- "what did I work on yesterday" → Memory recall

**Parent/Planner:**
- Decomposes complex requests into steps
- Routes to appropriate subsystem (GNN, self-coder, web, etc.)
- Manages tool execution
- Enforces safety policies

**Multi-provider LLM support:**
- DeepSeek (fast, cheap)
- Claude 3.5 Sonnet (high quality)
- Gemini 2.0/2.5 (balanced)
- GPT-4 (fallback)

### 8. Web Research Tools
**Location:** `app/chat/routes_web.py`
**Purpose:** Autonomous information gathering

**Capabilities:**
- Site query (extract info from specific URLs)
- Web search (Google/Bing/DuckDuckGo)
- Documentation assimilation
- Local knowledge indexing

**Use cases:**
- "What's the latest Django version?"
- "Search for solutions to this error message"
- "Extract API docs from this URL"

### 9. Curriculum Generation
**Location:** `.claude/agents/` (6 CERF agents: A1, A2, B1, B2, C1, C2)
**Purpose:** Autonomous creation of training lessons across 10 programming languages

**CEFR Framework (A1 → C2):**
- **A1/A2** (Beginner) - Variables, loops, basic data structures
- **B1/B2** (Intermediate) - OOP, comprehensions, exceptions
- **C1** (Professional) - Frameworks, async, testing, APIs
- **C2** (Expert/PhD) - Metaclasses, threading, algorithms, security

**Current status:**
- **1004 total lessons** (as of Oct 25, 2025)
- **100% agent-generated** - GitHub scraper removed, all lessons created by 6 CERF agents
- **Multi-language support:** 10 languages (Python, Java, SQL, JavaScript, TypeScript, C++, C#, Go, Rust, Ruby)
- **Language distribution (production-ready):**
  - **Tier 1 (20% each):** Python, Java, SQL (critical infrastructure)
  - **Tier 2:** JS/TS 15%, C++ 8%, C# 5%, Go 4%, PHP 3%, Rust 3%, Ruby 2%
- **CERF Distribution:**
  - A1 (Beginner): ~12%
  - A2 (Elementary): ~20%
  - B1 (Intermediate): ~41%
  - B2 (Upper-Int): ~19%
  - C1 (Professional): ~7%
  - C2 (Mastery): ~1% (growing)

**Agent Generation Process (YOLO Mode Enabled):**
1. Agent queries database for duplicates
2. LLM generates before/after code pairs for specific CERF level
3. Self-vetting (both snippets must execute, tests must pass)
4. Language field explicitly set (e.g., `language="python"`)
5. Store in `agent_generated_curriculum.sqlite` for review
6. Quality review (11 validation checks: 5 technical + 6 subjective)
7. Approved lessons merged to production `curriculum.sqlite`

**6 CERF Agents (Fully Autonomous):**
- All agents have `auto_approve_tools: true` (YOLO mode)
- Global permissions: `Write(*)`, `Edit(*)`, `Bash(rm:*)` enabled
- Each agent generates lessons appropriate to its CERF level
- Agents automatically clean up temporary files after generation
- Production database protected (agents write to separate workspace DB)

### 10. Memory Systems
**Location:** `app/chat/memory_bridge.py`, `out/memory/`
**Purpose:** Long-term knowledge retention

**Types:**
1. **Curriculum database** - 1004 code lessons (100% agent-generated across 10 languages)
2. **Experience logs** - All interactions, successes, failures
3. **Session cache** - Current conversation context
4. **Knowledge index** - Learned patterns and solutions
5. **Telemetry** - Training metrics, performance data

**Capabilities:**
- "What did I work on yesterday?" → Session recall
- "How did we solve X last time?" → Pattern lookup
- "Show me all bugs in authentication" → Category search

---

## 🎯 Use Cases (Real-World Applications)

### For Individual Developers

**1. Code Quality Mentor**
```
Scenario: Junior developer writes nested loops
Nerion: "I notice nested loops here. Consider NumPy vectorization (C1-level pattern).
         I can show you how—would you like me to refactor this?"
```

**2. Bug Prevention**
```
Scenario: Developer uses dict.get() without default
Nerion: "Warning: dict.get('key') without default caused KeyError in auth.py last week.
         Add default value: dict.get('key', None)"
```

**3. Voice-Driven Development**
```
Developer: [PTT] "Add type hints to all functions in utils.py"
Nerion: [TTS] "I'll add type hints based on usage patterns I've learned"
[Analyzes code, adds hints, runs tests]
Nerion: [TTS] "Done. Added type hints to 12 functions, all tests pass"
```

### For Teams

**1. Institutional Knowledge Capture**
```
Senior dev fixes subtle async bug
→ Nerion creates lesson automatically
→ Junior dev encounters similar pattern
→ Nerion: "This looks like the race condition Sarah fixed last month"
```

**2. Consistent Quality Enforcement**
```
Team standard: Always use context managers for file I/O
Developer writes: file = open(...)
Nerion: "Use context manager (with statement) per team standards"
```

**3. Onboarding Acceleration**
```
New developer joins team
→ Nerion has learned team's patterns for 6 months
→ Provides instant feedback on team conventions
→ Suggests improvements in team's style
```

### For Organizations

**1. Security Hardening**
```
Nerion detects: SQL string concatenation
Alert: "SQL injection risk detected in query_builder.py:42"
Auto-fix: Suggest parameterized queries
```

**2. Performance Optimization**
```
Nerion identifies: Nested loops over large datasets
Suggestion: "This O(n²) loop processes 10K items. Consider NumPy (50x faster)"
```

**3. Compliance Enforcement**
```
Organization policy: All API calls must have timeout
Nerion pre-commit hook: "requests.get() missing timeout parameter (policy violation)"
```

---

## 📊 Current State (What's Working NOW)

### ✅ Fully Operational

- **Voice-first chat interface** - Push-to-talk, TTS, STT working
- **Multi-provider LLM support** - DeepSeek, Claude, Gemini, GPT
- **Intent routing** - Automatic detection of user requests
- **Web research** - Site query and web search
- **Self-coding** - Can modify its own codebase
- **Healthcheck & diagnostics** - System monitoring
- **Memory system** - Long-term and session memory
- **GNN training pipeline** - Full training/validation workflow
- **Curriculum database** - 1004 lessons (100% agent-generated)
- **Mission Control GUI** - Professional Electron app
- **Daemon process** - Background immune system
- **Git hooks integration** - Pre-commit/post-commit (ready)
- **Multiple GNN architectures** - GCN, GraphSAGE, GIN, GAT
- **Multi-language support** - 10 languages (Python, Java, SQL, JS, TS, C++, C#, Go, Rust, Ruby)
- **6 CERF Agents** - Fully autonomous (YOLO mode) lesson generation across all levels

### 🔄 Partially Working

- **GNN accuracy** - 58.9% (target: 90%)
- **Auto-fix mode** - Training, not production yet
- **CodeBERT embeddings** - Integration complete, dataset generation ongoing (51%)

### 🚧 In Development

- **Phase 1 semantic features** - CodeBERT dataset 51% complete
- **Category population** - Tag NULL category lessons for filtering
- **Meta-learning** - Prompt improvement via LLM feedback
- **Framework lessons** - Agents can generate, need to scale up production (NumPy, Pandas, Flask, FastAPI, etc.)
- **C2 expansion** - Scale up mastery-level lesson generation

---

## 🗂️ File Structure (Key Locations)

```
/Users/ed/Nerion-V2/
├── app/                           # Application layer
│   ├── chat/                      # Chat engine, voice, intents
│   │   ├── engine.py              # Main conversation loop
│   │   ├── voice_io.py            # TTS/STT integration
│   │   ├── intents.py             # Intent detection
│   │   └── parent_exec.py         # Parent LLM executor
│   ├── parent/                    # Task planning system
│   │   ├── driver.py              # Parent coordinator
│   │   └── executor.py            # Tool execution
│   ├── ui/holo-app/               # Mission Control Electron app
│   │   ├── src/main.js            # Electron main process
│   │   └── src/mission-control/   # React UI components
│   ├── api/                       # REST API, terminal server
│   │   └── terminal_server.py     # PTY server for GUI
│   └── learning/                  # Upgrade agent
│       └── upgrade.py             # Self-improvement system
│
├── selfcoder/                     # Self-modification engine
│   ├── planner/                   # Plan generation
│   ├── actions/                   # Code modification primitives
│   ├── ast_editor/                # AST-based editing
│   ├── policy/                    # Safety policies
│   ├── governor.py                # Rate limiting, risk assessment
│   ├── security/                  # Malicious code detection
│   ├── tester/                    # Auto-test runner
│   ├── vcs/                       # Git integration
│   └── learning/                  # Continuous learning
│       ├── behavioral.py          # Fast heuristics (Behavioral Coach)
│       └── curriculum_gen.py      # Lesson generation
│
├── nerion_digital_physicist/      # GNN deep learning
│   ├── agent/
│   │   ├── brain.py               # GNN architectures (GCN, SAGE, GIN, GAT)
│   │   ├── data.py                # AST → Graph, feature extraction
│   │   └── semantics.py           # CodeBERT embeddings
│   ├── training/
│   │   ├── run_training.py        # Training loop
│   │   └── dataset_builder.py    # Curriculum → PyTorch Geometric
│   ├── db/
│   │   └── safe_curriculum.py     # Curriculum database schema
│   └── generation/                # Lesson generation
│       └── lesson_generator.py    # LLM-based lesson creation
│
├── voice/                         # Voice interface
│   ├── stt/                       # Speech-to-text (Whisper, Vosk)
│   ├── tts/                       # Text-to-speech (pyttsx3, cloud)
│   ├── wake_word/                 # Wake word detection
│   └── io_bus/                    # Audio I/O management
│
├── daemon/                        # Immune system daemon
│   └── nerion_daemon.py           # 24/7 background process
│
├── config/                        # Configuration
│   ├── tools.yaml                 # Tool manifest
│   ├── intents.yaml               # Intent definitions
│   └── model_catalog.yaml         # LLM provider configs
│
├── out/                           # Runtime data
│   ├── learning/
│   │   └── curriculum.sqlite      # 973 code lessons (483 GitHub + 490 generated)
│   ├── training_runs/             # GNN training history
│   │   └── oct17_comparison/      # Recent 4-arch comparison
│   └── memory/                    # Long-term memory storage
│
├── experiments/                   # Research artifacts
│   └── datasets/gnn/
│       ├── final_complete/supervised/20251017T202703Z/  # Baseline (48-dim)
│       └── codebert_semantic/     # CodeBERT dataset (816-dim, IN PROGRESS)
│
├── digital_physicist_brain.pt     # Current GNN weights
├── digital_physicist_brain.meta.json  # Model metadata
├── Nerion.app                     # macOS application (symlink)
├── start_nerion.sh                # Unified startup script
├── .env                           # API keys (Gemini, Claude, etc.)
└── CLAUDE.md                      # This file
```

---

## 🚀 Roadmap (Where We're Going)

### Current Phase: MVP → Production (Q4 2025)
**Focus:** GNN accuracy 90%, curriculum completion, immune system deployment

**Immediate Priorities (1-2 weeks):**
1. ✅ Complete CodeBERT dataset generation (51% done, ETA 1-2 days)
2. ✅ Train SAGE with semantic features (target: 75-80% accuracy)
3. ✅ Measure Phase 1 improvement vs 58.9% baseline
4. ⚠️ Populate category metadata for 930 NULL lessons

**Short-term (1-2 months):**
1. Scale up lesson generation (target 5000+ lessons across all 10 languages)
2. Expand C2 mastery content (scale up agent generation)
3. Deploy daemon on Nerion codebase (dogfooding - monitor itself)
4. Create macOS .dmg installer
5. Implement git pre-commit hooks (block bad code at commit time)

**Medium-term (3-6 months):**
1. Reach 90% GNN accuracy (Phase 1-6 complete)
2. Auto-fix mode with confidence thresholds (>95% = auto, <95% = suggest)
3. Community lesson marketplace (share/download curriculum)
4. Balance curriculum across 10 languages (maintain 20/20/20/15/8/5/4/3/3/2 distribution)

### V1 (6-12 months): Production Ready
- ✅ Multi-language support (COMPLETE - 10 languages operational)
- CI/CD integrations (GitHub Actions, GitLab CI, CircleCI)
- Cloud-hosted brain option (SaaS for pre-trained models)
- Auto-fix mode (confidence >95%)
- Lesson marketplace (community sharing)
- Desktop app distribution (macOS .dmg, Windows .exe)
- IDE plugins (VS Code, PyCharm, IntelliJ)

### V2 (12-24 months): Autonomous Evolution
- Recursive self-improvement (Nerion improves its own prompts)
- Runtime monitoring integration (Sentry/Datadog)
- Multi-agent architecture (specialized agents for security, performance, docs)
- Evolutionary category discovery (auto-discover new lesson types)
- Cross-language learning (Python lessons → JavaScript)

### V3+ (Year 2+): Biological Ecosystem
- Enterprise deployment (on-premise, air-gapped)
- Multi-team collaboration (shared organizational brain)
- Autonomous refactoring (architectural improvements)
- True recursive self-improvement (improve core GNN architecture)
- Codebase swarm intelligence (multiple codebases share knowledge)

---

## 🔧 Current Work (IN PROGRESS)

### CodeBERT Dataset Generation (Background Process)
- **Status:** 450/973 lessons processed (46.3% complete)
- **Started:** October 19, 2025
- **ETA:** 1-2 days remaining (slow due to CPU-only processing)
- **Process:** PID 47891 (confirmed working - cache actively updated)
- **Output:** `experiments/datasets/gnn/codebert_semantic/supervised/*/dataset.pt`
- **Log:** `/tmp/dataset_gen.log`

**Why so slow?**
- CodeBERT is 125M parameter transformer
- Running on M2 Max CPU (no GPU acceleration configured)
- ~0.28 lessons/minute (each lesson = 2 code snippets)
- Cache helps, but initial embeddings are expensive

**Monitoring:**
```bash
# Check progress
tail -50 /tmp/dataset_gen.log

# Verify process working
ps aux | grep "[4]7891"

# Check cache updates (confirms active work)
ls -lh nerion_digital_physicist/agent/.semantic_cache.json
```

### GNN Training Phase 1 Plan
Once dataset completes:
1. **Verify features:** Check dataset has 816 dimensions (48 + 768)
2. **Train SAGE:** Use same hyperparameters, new dataset
3. **Compare results:** Baseline 58.9% vs semantic (target 75-80%)
4. **Analyze features:** Which semantic features contribute most?
5. **Document findings:** Update CLAUDE.md with Phase 1 results

---

## 📝 Technical Details

### GNN Architecture Specifications
All models use consistent hyperparameters for fair comparison:
- **Hidden channels:** 256
- **Num layers:** 4
- **Dropout:** 0.2
- **Pooling:** mean (global aggregation)
- **Attention heads:** 4 (for GAT only)
- **Optimizer:** Adam
- **Learning rate:** 1e-3
- **Batch size:** 32
- **Validation split:** 15%
- **Early stopping:** Patience ~20% of epochs

### Dataset Evolution (Understanding Accuracy Trends)

| Dataset | Date | Samples | Avg Nodes/Graph | Accuracy | Notes |
|---------|------|---------|-----------------|----------|-------|
| Sept 29 | 2025-09-29 | 200 | 2.8 | 70-75% | Simple test code |
| Oct 10 | 2025-10-10 | 889 | 2.5 | 63-68% | Expanded set |
| Oct 17 | 2025-10-17 | 1,800 | 111 | 47-59% | Production code (40x more complex) |

**Key Insight:** Accuracy "decline" (75% → 59%) is NOT model degradation—it's increased task difficulty. Oct 17 dataset represents real production-quality code with complex structures, which is much harder to classify correctly.

### CodeBERT Details
- **Model:** microsoft/codebert-base
- **Parameters:** 125M
- **Pre-training:** 6.4M GitHub code-docstring pairs
- **Languages:** Python, Java, JavaScript, PHP, Ruby, Go
- **Output:** 768-dimensional vector per code snippet
- **Performance:**
  - CPU (M2 Max): ~0.28 lessons/minute
  - GPU (if configured): Would be 50-100x faster
  - Gemini API (alternative): 100x faster, slightly lower quality

---

## 💡 Key Decisions & Context

### Why CodeBERT vs Other Embeddings?
**CodeBERT chosen for:**
- Code-specific pre-training (6.4M GitHub samples)
- Understands programming semantics (variable naming, function patterns)
- 768-dimensional rich features
- Proven effectiveness on code understanding tasks

**Trade-offs accepted:**
- Very slow on CPU (4-5 days total)
- Could use Gemini API (100x faster, slightly lower quality)
- User prioritized quality over speed: "i dont mind slow"

### Why GraphSAGE Architecture?
- Best performance in Oct 17 comparison (58.9% vs 55.2% GCN, 54.8% GAT, 47.4% GIN)
- Good balance of expressiveness and stability
- Handles variable-sized graphs well (inductive learning)
- Scales to large graphs efficiently

### Dataset Complexity
Oct 17 dataset is 40x more complex (111 nodes/graph vs 2.8 nodes/graph):
- Represents real production code (not toy examples)
- Lower accuracy is expected for harder task
- More realistic evaluation of model capability
- Better test of production readiness

### Two-Tiered Learning Philosophy
**Why have both Behavioral Coach + Digital Physicist?**
- 80/20 rule: 80% of tasks are common patterns (fast path)
- 20% of tasks are novel problems (deep path)
- Behavioral Coach handles routine (instant response)
- Digital Physicist handles complexity (slower, but learns)
- Best of both worlds: Speed + Intelligence

---

## 🎮 Quick Start Commands

### Check Dataset Generation Status
```bash
tail -50 /tmp/dataset_gen.log
ps aux | grep python | grep dataset_builder
stat -f "%Sm" nerion_digital_physicist/agent/.semantic_cache.json
```

### Train SAGE (When Dataset Ready)
```bash
python3 -m nerion_digital_physicist.training.run_training \
  --dataset experiments/datasets/gnn/codebert_semantic/supervised/*/dataset.pt \
  --output-dir out/training_runs/codebert_sage \
  --architecture sage \
  --epochs 50 \
  --batch-size 32 \
  --hidden-channels 256 \
  --num-layers 4 \
  --dropout 0.2
```

### Generate Dataset with Gemini (Fast Alternative)
```bash
# Kill CodeBERT process first: kill -9 47891
export NERION_V2_GEMINI_KEY="your_key_here"
NERION_SEMANTIC_PROVIDER=gemini python3 -m nerion_digital_physicist.training.dataset_builder \
  --db out/learning/curriculum.sqlite \
  --output-dir experiments/datasets/gnn/gemini_semantic \
  --name "oct17_with_gemini" \
  --mode supervised
```

### Start Mission Control GUI
```bash
cd app/ui/holo-app
npm run dev  # Development mode
# OR
npm run build && open ../../Nerion.app  # Production build
```

### Start Immune System Daemon
```bash
python3 daemon/nerion_daemon.py --codebase /path/to/your/repo
```

### Voice Interface
```bash
python3 -m app.chat.engine --voice --profile whisper_fast
# Press Ctrl+Space for push-to-talk
```

---

## 📖 Context for Next Session

When restarting Claude Code:

### First Actions
1. **Check dataset completion:** `ls experiments/datasets/gnn/codebert_semantic/supervised/*/dataset.pt`
2. **If complete:** Train SAGE with semantic features, compare to 58.9% baseline
3. **If incomplete:** Check progress and verify process:
   ```bash
   # Find log file location
   lsof -p 47891 2>/dev/null | grep "\.log"

   # Read actual progress (look for "Progress: X/973")
   tail -50 /private/tmp/dataset_gen.log

   # Verify process health
   ps -p 47891 -o pid,state,pcpu,etime,command

   # Check cache is updating (should change every few seconds)
   stat -f "%Sm" -t "%H:%M:%S" nerion_digital_physicist/agent/.semantic_cache.json
   ```

### Current Focus
- **Phase 1 semantic embeddings** to reach 75-80% accuracy (from 58.9%)
- CodeBERT dataset generation: **500/973 lessons (51.4%)** - ETA ~34 hours (Wednesday evening)
- PID 47891 running since Oct 18 10:57 PM, actively processing at ~14 lessons/hour
- End goal: 90% accuracy for production deployment

### Remember
- Nerion is NOT just a GNN training tool—it's a **biological immune system for software**
- GNN is ONE component of 10+ major subsystems
- Voice interface, daemon, GUI, self-coder all equally important
- End goal: Autonomous code health at PhD-level

---

## 🔍 Monitoring & Maintenance

### Check Progress Commands
```bash
# Dataset generation progress
tail -50 /tmp/dataset_gen.log

# Process status (PID 47891)
ps aux | grep "[4]7891"

# Cache updates (confirms active work)
ls -lh nerion_digital_physicist/agent/.semantic_cache.json

# Verify process working (not stuck)
stat -f "%Sm" nerion_digital_physicist/agent/.semantic_cache.json
```

### Training Progress
Training runs output to `out/training_runs/` with:
- `digital_physicist_brain.pt` - Model weights
- `metrics.json` - Full training history (accuracy, loss, AUC, F1 per epoch)
- Console logs showing epoch-by-epoch progress

### Daemon Health
```bash
# Check if daemon running
ps aux | grep nerion_daemon

# View daemon logs
tail -f ~/.nerion/daemon.log

# Check socket connection
ls -la ~/.nerion/daemon.sock
```

---

## ⚙️ Important Configuration

### Environment Variables (.env)
```bash
NERION_V2_GEMINI_KEY=<your_gemini_key>        # For embeddings/chat
CLAUDE_API_KEY=<your_claude_key>              # For chat
DEEPSEEK_API_KEY=<your_deepseek_key>          # For chat
NERION_SEMANTIC_PROVIDER=codebert             # or "gemini"
NERION_SEMANTIC_TIMEOUT=120                   # seconds
```

### Model Catalog (config/model_catalog.yaml)
Defines available LLM providers and models for:
- Chat completions
- Embeddings
- Function calling
- Streaming

---

## 🎯 Todo List (Current Phase)

1. ✅ Examine current feature extraction pipeline
2. ✅ Install transformers library for CodeBERT
3. ✅ Add CodeBERT to SemanticEmbedder
4. ✅ Test CodeBERT embedding on sample code
5. ✅ Check dataset builder for regeneration
6. 🔄 **Wait for CodeBERT dataset generation (450/973, ~1-2 days remaining)**
7. ⏳ Verify dataset has 816-dim features (48 structural + 768 semantic)
8. ⏳ Train SAGE with CodeBERT embeddings
9. ⏳ Compare baseline (58.9%) vs semantic (target 75%+)
10. ⏳ Update CLAUDE.md with Phase 1 results

---

## 📚 Additional Resources

### Documentation
- README.md - Project overview
- ARCHITECTURE.md - System design (if exists)
- config/tools.yaml - Available tools and capabilities

### Training Artifacts
- `out/training_runs/oct17_comparison/` - Recent 4-architecture comparison
- `digital_physicist_brain.meta.json` - Current model metadata

### Lesson Categories (22 current)
Exception handling, async patterns, security, performance, type hints, comprehensions, generators, context managers, logging, testing, optimization, error messages, decorators, class design, function design, code smells, documentation, naming, imports, f-strings, pathlib, subprocess

### Missing Categories (58 needed)
NumPy, Pandas, Flask, Django, FastAPI, SQLAlchemy, pytest fixtures, basics (variables, loops, conditionals, if/else, functions), HTTP requests, JSON handling, CSV processing, regex, datetime, collections, itertools, functools, dataclasses, enums, and more

---

## 🏁 Summary

**Nerion in One Sentence:**
A biological immune system for software that lives in your codebase, learns from your team's patterns, prevents bugs 24/7, and evolves itself to become a PhD-level code quality expert.

**Three Key Points:**
1. **Not a Tool, an Organism** - Permanent resident, not external plugin
2. **Multi-Tiered Intelligence** - Fast heuristics + Deep learning
3. **Self-Improving** - Gets smarter every day about YOUR code

**Current State:**
- 10+ subsystems operational (voice, GUI, daemon, GNN, self-coder, memory)
- GNN at 58.9% accuracy, targeting 90%
- Phase 1 (semantic embeddings) 46% complete
- Production-ready infrastructure, needs accuracy improvement

**Next Milestone:**
Complete Phase 1, reach 75-80% accuracy, deploy daemon on Nerion codebase (dogfooding)

---

## 📝 Maintenance Guidelines

### When to Update CLAUDE.md
- Architecture changes (new components, significant refactors)
- New major features (voice interface, GUI updates, etc.)
- Current state changes (accuracy milestones, phase completions)
- Roadmap updates (shifting priorities, new timelines)

### When to Update CHANGELOG.md
- ✅ **ONLY after changes are tested and confirmed working**
- New features successfully integrated
- Bugs fixed and verified
- Performance improvements measured
- Configuration changes deployed
- **Never leave "In Progress" entries in CHANGELOG.md** - those belong in CLAUDE.md

### What NOT to Log in CHANGELOG.md
- ❌ Experimental changes still being tested (keep in CLAUDE.md)
- ❌ Failed attempts or bugs (document lessons learned if valuable)
- ❌ In-progress work (track in CLAUDE.md "Current Work" section)
- ❌ Speculative future plans (document in CLAUDE.md "Roadmap" section)
- ❌ Stale status updates (update or remove immediately when status changes)

### Quality Control Rules
- **CHANGELOG.md = Factual history** - Only confirmed changes
- **CLAUDE.md = Current state** - Includes in-progress work
- **If status changes** - Update CLAUDE.md immediately, only add to CHANGELOG.md when confirmed
- **No ambiguity** - Reader should never wonder "Did this actually happen?"

---

*CLAUDE.md is the single source of truth for understanding Nerion.*
*CHANGELOG.md is the authoritative history of confirmed changes.*
- remember, whenever creating a one time migration script/test-file for one time use, or any other one time use files, whenever the job is done make sure not to leave the one time use file in the directory, always delete that file.
- make sure not to overflow the directory with MD files, double check if there is a md file present for current implemantation update that one instead of creating new ones. and again make sure to follow claude.md rules and log all verified changes in changelog.md