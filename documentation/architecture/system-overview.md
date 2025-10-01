# System Overview

Nerion V2 is a professional-grade, open-source platform that unites a voice-first assistant with a self-improving software engineer. It keeps the agent, planner, and safety loops on your local machine while delegating language generation to the API providers you configure.

## Core Architecture: A Two-Tiered Mind

Nerion's autonomy is powered by a two-tiered learning architecture, allowing it to handle both everyday tasks and complex, novel problems.

1.  **The Behavioral Coach (Fast, Instinctual Learning):** Handles the 80% of common tasks: fixing simple code smells, learning which tools to use, and performing routine maintenance. It is fast, efficient, and heuristic.

2.  **The Digital Physicist (Deep, Analytical Reasoning):** This is Nerion's deep-thinking GNN engine. It learns by generating its own curriculum and training itself on the outcomes, allowing it to master the complex "physics" of the codebase. It is invoked for novel bugs, major refactoring, and proactive optimization.
