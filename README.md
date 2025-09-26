# Nerion V2 (API-First, Voice-Forward, Self-Improving)

> **Nerion V2 notice (API-first pivot):** this repository tracks the evolution of Nerion toward hosted LLM providers. Follow `docs/nerion-v2-api.md` for the canonical migration plan and provider defaults.

Nerion V2 is a professional-grade platform that unites a voice-first assistant with a self-improving software engineer. It keeps the agent, planner, and safety loops on your machine while delegating language generation to the API providers you configure.

## What is Nerion?
* **A Self-Hosted Orchestrator:** The agent, safety rails, and data stores live on your hardware. Outbound network traffic is limited to the LLM APIs you explicitly enable.
* **An Autonomous Engineer:** Nerion can write, debug, and refactor its own code. You give it high-level goals, and it generates and safely applies the necessary code changes.
* **A Knowledge Engine:** Nerion can research topics on the web, read documentation, and synthesize information into a personalized knowledge base.

---

## Project Architecture: A Two-Tiered Mind

Nerion's autonomy is powered by a two-tiered learning architecture, allowing it to handle both everyday tasks and complex, novel problems.

### 1. The Behavioral Coach (Fast, Instinctual Learning)
This is Nerion's first line of intelligence, rooted in modules like `selfcoder/self_improve.py`.

*   **How it learns:** It learns from the direct success or failure of its actions. By observing which tools and strategies work best over time, it optimizes its own behavior.
*   **Its Role:** It handles the 80% of common tasks: fixing simple, well-defined code smells from linters, learning which tools to use for which intents, and performing routine maintenance. It is fast, efficient, and heuristic.

### 2. The Digital Physicist (Deep, Analytical Reasoning)
This is Nerion's deep-thinking engine, which is the current focus of development.

*   **How it learns:** It learns via a curiosity-driven approach. The GNN predicts the outcome of a code change and is then rewarded based on the **"surprise"** of the actual result. By prioritizing surprising outcomes (where its predictions were wrong), it focuses its training on what it doesn't understand, making learning highly efficient. This mechanism drives it to explore and master the complex physics of the codebase.
*   **Its Role:** It is invoked for the 20% of complex problems the "Coach" can't solve: novel bugs, major architectural refactoring, and proactive performance optimization.

---

## The Digital Physicist: Development Plan

The evolution of the Digital Physicist is the project's core strategic focus. It follows a clear, three-step plan.

*   **Step 1: Scale Up the GNN Brain (In Progress)**
    *   **Goal:** Evolve the GNN to a "specialist-grade" brain with the capacity to learn complex patterns.
    *   **Status:** The foundational work is complete. We have a stable training loop where the GNN learns from the verified outcomes of its code manipulations. We are now expanding the complexity of tasks it can learn.

*   **Step 2: Build the Orchestrator (Next)**
    *   **Goal:** Create the high-level logic that allows the GNN (Deep Thinker) and the LLM (Creative Brain) to communicate and collaborate, with the Orchestrator dispatching problems to the appropriate system.

*   **Step 3: Implement the "Meta-Policy" (Future)**
    *   **Goal:** Build the high-level executive function that gives Nerion true creative autonomy, allowing it to set its own strategic goals for self-improvement.

---

## Quick Start

1.  **Install in Editable Mode**
    ```bash
    pip install -e .[dev]
    ```

2.  **Provide API Credentials**
    ```bash
    cp .env.example .env
    # edit .env and add your API keys (e.g., NERION_V2_GEMINI_KEY)
    ```

3.  **Launch the Agent**
    ```bash
    nerion-chat
    ```

## Key Commands

*   **Run a full system diagnostic:**
    ```bash
    nerion healthcheck
    ```
*   **Ask Nerion to self-code a change:**
    ```bash
    nerion plan -i "add a try/except wrapper to the main function" --apply
    ```
*   **Research a topic on the web:**
    ```bash
    nerion docs site-query --query "best laptop for developers 2025" --augment
    ```
*   **Inspect the code graph:**
    ```bash
    nerion graph affected --symbol MyClass --depth 2
    ```
*   **View the full command list:**
    ```bash
    nerion --help
    ```

> For a more comprehensive command reference, see the Cheat Sheet: `docs/CHEATSHEET.md`.