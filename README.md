# Nerion V2: The Self-Improving Developer Agent

> For complete and detailed documentation, please visit the [documentation](./documentation/README.md) directory.


> For complete and detailed documentation, please visit the [documentation](./documentation/README.md) directory.


Nerion V2 is a professional-grade, open-source platform that unites a voice-first assistant with a self-improving software engineer. It keeps the agent, planner, and safety loops on your local machine while delegating language generation to the API providers you configure.

## Core Architecture: A Two-Tiered Mind

Nerion's autonomy is powered by a two-tiered learning architecture, allowing it to handle both everyday tasks and complex, novel problems.

1.  **The Behavioral Coach (Fast, Instinctual Learning):** Handles the 80% of common tasks: fixing simple code smells, learning which tools to use, and performing routine maintenance. It is fast, efficient, and heuristic.

2.  **The Digital Physicist (Deep, Analytical Reasoning):** This is Nerion's deep-thinking GNN engine. It learns by generating its own curriculum and training itself on the outcomes, allowing it to master the complex "physics" of the codebase. It is invoked for novel bugs, major refactoring, and proactive optimization.

---

## Installation

This project uses a modern, flexible dependency management system. You can choose to install only the components you need.

### 1. Clone the Repository

```bash
git clone https://github.com/EduardAvojan/Nerion-V2-.git
cd Nerion-V2-
```

### 2. Install Dependencies

Choose one of the following installation methods:

**A) Minimal Installation (Core Agent)**

This is recommended for headless environments or for users who only need the core AI and automation logic.

```bash
pip install -e .
```

**B) Full User Installation (with Voice and Web)**

This installs the core agent plus all dependencies required for voice interaction and web browsing capabilities.

```bash
pip install -e ".[voice,web,docs]"
```

**C) Development Installation**

This installs all dependencies, including the tools required for testing and development.

```bash
pip install -e ".[dev,voice,web,docs]"
```

### 3. Provide API Credentials

Create a `.env` file for your API keys by copying the example template.

```bash
cp .env.example .env
```

Now, edit the `.env` file and add your API keys (e.g., `NERION_V2_GEMINI_KEY`).

---

## Quick Start

Once installed, you can launch the agent's chat interface:

```bash
nerion-chat
```

Or run a full system diagnostic:

```bash
nerion healthcheck
```

### Key Commands

*   **Ask Nerion to self-code a change:**
    ```bash
    nerion plan -i "add a try/except wrapper to the main function" --apply
    ```
*   **Research a topic on the web:**
    ```bash
    nerion docs site-query --query "best laptop for developers 2025" --augment
    ```

> For a more comprehensive command reference, see the Cheat Sheet: `docs/CHEATSHEET.md`
