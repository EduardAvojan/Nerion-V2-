<a name="top"></a>
# Nerion - A Biological Immune System for Software

<div align="center">

![Nerion Logo](https://img.shields.io/badge/Nerion-Immune_System-blue?style=for-the-badge)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=flat-square)](https://github.com/EduardAvojan/Nerion-V2-)

**Autonomous Code Quality | PhD-Level Intelligence | 24/7 Protection**

[Quick Start](#quick-start) • [Documentation](#documentation) • [Features](#key-features) • [Architecture](#architecture) • [Contributing](#contributing)

</div>

---

## 🎯 What is Nerion?

**Nerion is not a code quality tool—it's a biological immune system for software that operates autonomously at PhD-level.**

Imagine if your codebase had a **living immune system** like your body:
- **Permanently integrated** - Not an external tool you run, but a resident organism
- **Continuously learning** - Gets smarter every day about YOUR specific code
- **Proactive & autonomous** - Monitors 24/7, not on-demand
- **Self-healing** - Detects and fixes issues automatically
- **Evolves** - Recursively improves its own capabilities

### The Vision

Nerion learns the "physics" of code quality using Graph Neural Networks, generates its own curriculum, and autonomously improves your codebase while you sleep. It's designed to reach **90% code quality classification accuracy** and prevent bugs before they reach production.

---

## ✨ Key Features

### 🧠 Deep Learning Brain
- **Graph Neural Networks (GNN)** - Understands code structure, not just syntax
- **58.9% accuracy baseline** (GraphSAGE architecture) - targeting 90%
- **1,004 training lessons** across 10 programming languages
- **Self-improving** - Generates its own curriculum

### 🌍 Multi-Language Support
Protects production systems in **10 languages**:
- **Tier 1** (20% each): Python, Java, SQL
- **Tier 2**: JavaScript, TypeScript, C++, C#, Go, Rust, Ruby

### 🎤 Voice-First Interface
- **Push-to-talk** or wake word ("Hey Nerion")
- **Hands-free development** - code while walking, thinking, or resting
- **Multiple STT/TTS providers** - Whisper, Vosk, cloud APIs

### 🖥️ Mission Control GUI
- **Professional Electron app** - Cross-platform desktop interface
- **Real-time monitoring** - Training dashboards, system vitals
- **Integrated terminal** - Full bash PTY with reconnection
- **Neural visualization** - Watch the brain learn

### 🛡️ Autonomous Protection
- **24/7 daemon** - Background monitoring and healing
- **Git hooks integration** - Pre-commit bug prevention
- **Auto-fix mode** - High-confidence fixes applied automatically
- **Safety policies** - Configurable approval gates

### 🔄 Self-Coding Engine
- **AST-based editing** - Never breaks your code with regex
- **Plan generation** - Multi-step improvements with LLM
- **Test-driven** - Runs tests automatically after changes
- **Git integration** - Automatic commits with descriptive messages

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ (for Mission Control GUI)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/EduardAvojan/Nerion-V2-.git
cd Nerion-V2-
```

### 2. Install Dependencies

**Option A: Minimal Installation (Core Agent)**
```bash
pip install -e .
```

**Option B: Full Installation (Voice + Web + GUI)**
```bash
pip install -e ".[voice,web,docs]"
cd app/ui/holo-app
npm install
```

**Option C: Development Installation (Everything)**
```bash
pip install -e ".[dev,voice,web,docs]"
cd app/ui/holo-app
npm install
```

### 3. Configure API Keys

Copy the example environment file and add your API keys:
```bash
cp .env.example .env
# Edit .env and add your keys (Gemini, Claude, etc.)
```

### 4. Launch Nerion

**Start Mission Control GUI:**
```bash
cd app/ui/holo-app
npm run dev
```

**Start Chat Interface:**
```bash
nerion-chat
```

**Run Health Check:**
```bash
nerion healthcheck
```

**Start Voice Interface:**
```bash
nerion-chat --voice
```

---

## 📚 Documentation

### 🎓 Getting Started
- [**Quick Start Guide**](docs/guides/QUICKSTART.md) - Get up and running in 5 minutes
- [**Contributing Guide**](docs/guides/CONTRIBUTING.md) - Join the development effort
- [**Troubleshooting**](docs/guides/troubleshooting.md) - Common issues and solutions

### 🏗️ Architecture & Design
- [**System Overview**](CLAUDE.md) - Complete system documentation (single source of truth)
- [**Mission Control Design**](docs/architecture/MISSION_CONTROL_DESIGN.md) - Electron GUI architecture
- [**Immune System Summary**](docs/architecture/IMMUNE_SYSTEM_SUMMARY.md) - Daemon and monitoring
- [**Evolution Plan**](docs/architecture/evolution_plan.md) - Roadmap to 90% accuracy
- [**API Reference**](docs/architecture/nerion-v2-api.md) - REST API documentation

### 🧩 Components
- [**Electron App**](docs/components/README_ELECTRON.md) - Mission Control GUI
- [**Immune System Daemon**](docs/components/README_IMMUNE_SYSTEM.md) - 24/7 background protection
- [**Voice Interface**](docs/components/voice_and_hot_reload.md) - STT/TTS integration
- [**IDE Bridge**](docs/components/ide_bridge_tui.md) - Terminal UI integration

### 🛠️ Development
- [**Agent Lesson Workflow**](docs/development/AGENT_LESSON_WORKFLOW.md) - How to generate curriculum lessons
- [**Lesson Specification**](docs/development/nerion_lesson_specification.md) - CERF framework details
- [**Cheat Sheet**](docs/development/CHEATSHEET.md) - Quick command reference
- [**Policy System**](docs/development/policy.md) - Safety policies and approval gates
- [**Learning Cycle**](docs/development/learning_cycle.md) - How Nerion learns and improves

### 📖 Reference
- [**Database Protection**](docs/reference/DATABASE_PROTECTION.md) - 7-layer protection system
- [**Duplicate Prevention**](docs/reference/DUPLICATE_PREVENTION.md) - SHA256 content hashing
- [**Project Journal**](docs/reference/project_journal.md) - Development history
- [**Safety Progress**](docs/reference/safety_progress.md) - Safety features timeline

### 📜 Changelog
- [**CHANGELOG.md**](CHANGELOG.md) - Detailed history of all verified changes

---

## 🏛️ Architecture

Nerion is a **multi-tiered autonomous developer agent** with 10+ major subsystems:

```
┌─────────────────────────────────────────────────┐
│           USER INTERFACES                       │
│  Mission Control GUI | Voice | CLI/Chat        │
└─────────────────┬───────────────────────────────┘
                  │
      ┌───────────┴───────────┐
      │   IMMUNE SYSTEM       │
      │   (24/7 Daemon)       │
      └───────────┬───────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
┌─────▼─────┐        ┌───────▼────────┐
│  DIGITAL  │        │  SELF-CODER    │
│ PHYSICIST │        │    ENGINE      │
│ (GNN)     │        │  (AST-based)   │
└───────────┘        └────────────────┘
      │                       │
      └───────────┬───────────┘
                  │
        ┌─────────▼─────────┐
        │  CURRICULUM DB    │
        │  1,004 Lessons    │
        │  10 Languages     │
        └───────────────────┘
```

### Two-Tiered Learning

1. **Behavioral Coach (Fast Path)** - Handles 80% of common tasks with heuristics
   - Variable naming conventions
   - Import sorting
   - Docstring formatting
   - Simple refactors

2. **Digital Physicist (Deep Path)** - Handles 20% of novel problems with GNN
   - Novel bug patterns
   - Complex refactoring
   - Architectural improvements
   - Performance optimization

---

## 🎓 CERF Curriculum Framework

Nerion's training curriculum spans **6 skill levels** (A1 → C2):

| Level | Description | Example Topics |
|-------|-------------|----------------|
| **A1** | Beginner | Variables, loops, basic data structures |
| **A2** | Elementary | Lists, dicts, file I/O, string methods |
| **B1** | Intermediate | Classes, inheritance, decorators, generators |
| **B2** | Upper-Intermediate | Async/await, metaclasses, threading |
| **C1** | Professional | Memory profiling, distributed systems, security |
| **C2** | Mastery/PhD | Compiler internals, JIT, GC algorithms |

**Current Database:**
- 1,004 total lessons (100% agent-generated)
- 10 programming languages
- Production-ready distribution (20% Python/Java/SQL, 40% others)

---

## 🛣️ Roadmap

### Current Phase: MVP → Production (Q4 2025)

**Immediate (1-2 weeks):**
- ✅ Complete CodeBERT dataset generation (51% done)
- ✅ Train SAGE with semantic features (target: 75-80% accuracy)
- ⏳ Populate category metadata for 930 NULL lessons

**Short-term (1-2 months):**
- Add framework lessons (~100-150): NumPy, Pandas, Flask, FastAPI
- Expand C2 mastery content (40-50 more lessons)
- Deploy daemon on Nerion codebase (dogfooding)
- Create macOS .dmg installer

**Medium-term (3-6 months):**
- Reach 90% GNN accuracy (Phase 1-6 complete)
- Auto-fix mode with confidence thresholds
- Community lesson marketplace

### V1 (6-12 months): Production Ready
- CI/CD integrations (GitHub Actions, GitLab CI)
- Cloud-hosted brain option (SaaS)
- Desktop app distribution (macOS .dmg, Windows .exe)
- IDE plugins (VS Code, PyCharm)

### V2 (12-24 months): Autonomous Evolution
- Recursive self-improvement
- Runtime monitoring integration
- Multi-agent architecture
- Cross-language learning

---

## 🤝 Contributing

We welcome contributions! Nerion is an ambitious project to create a truly autonomous code quality system.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** (follow the coding standards)
4. **Run tests** (`pytest`)
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to your fork** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Areas We Need Help

- **Curriculum Generation** - Create high-quality code lessons (A1-C2 levels)
- **GNN Improvements** - Experiment with architectures (Graph Transformers, etc.)
- **Multi-Language Support** - Add more language-specific patterns
- **Testing** - Increase test coverage (currently ~60%)
- **Documentation** - Improve guides and tutorials
- **UI/UX** - Enhance Mission Control GUI

See [CONTRIBUTING.md](docs/guides/CONTRIBUTING.md) for detailed guidelines.

---

## 📊 Current Status

### ✅ Fully Operational
- Voice-first chat interface (PTT, TTS, STT)
- Multi-provider LLM support (DeepSeek, Claude, Gemini, GPT)
- Intent routing and command execution
- Web research capabilities
- Self-coding with AST-based editing
- Memory system (long-term + session)
- GNN training pipeline (4 architectures tested)
- Mission Control GUI (Electron app)
- Daemon process (background immune system)
- Multi-language support (10 languages)
- 6 CERF Agents (fully autonomous)

### 🔄 In Development
- GNN accuracy improvement (58.9% → 90%)
- CodeBERT semantic embeddings (Phase 1)
- Framework lesson expansion
- C2 mastery content scaling

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Claude Code** - Development assistant
- **PyTorch Geometric** - GNN framework
- **CodeBERT** - Semantic code embeddings
- **Electron** - Desktop app framework
- **All contributors** - Thank you for your support!

---

## 📧 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/EduardAvojan/Nerion-V2-/issues)
- **Discussions**: [Join the conversation](https://github.com/EduardAvojan/Nerion-V2-/discussions)

---

<div align="center">

**Built with ❤️ by the Nerion Team**

⭐ **Star this repo** if you find it useful!

[⬆ Back to Top](#top)

</div>
