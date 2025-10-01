# How it Works

The Digital Physicist is a sophisticated system that combines a Graph Neural Network (GNN), a world model, and a learning loop to achieve a deep understanding of the codebase. It operates through several interconnected components: the Agent, the Environment, and the Generation system.

## The Agent: The Brain of the Digital Physicist

The Agent component (`nerion_digital_physicist/agent`) houses the core intelligence of the Digital Physicist.

*   **Brain (`brain.py`)**: This module defines various Graph Neural Network (GNN) architectures (GCN, SAGE, GIN, GAT) that form the "brain" of the Digital Physicist. These GNNs are designed to process graph representations of code and predict the outcome of changes. They are built upon a `_StackedGraphModel` that ensures consistent output dimensionality, supporting residual connections and flexible pooling strategies.
*   **Data (`data.py`)**: This module is crucial for transforming raw code into a structured graph format that the GNN can understand. It parses Python Abstract Syntax Trees (ASTs), extracts structural features (e.g., line count, argument count, cyclomatic complexity) for functions, and constructs a `networkx` graph. This graph is then converted into a PyTorch Geometric `Data` object, with nodes featurized using both structural and semantic (embeddings) information. It also defines various edge roles (e.g., `sequence`, `call`, `shared_symbol`) to capture rich relationships within the code.
*   **Policy (`policy.py`)**: This module defines the agent's learning and decision-making policy. The `AgentV2` class integrates the GNN brain, an optimizer, and a memory system to learn from experiences. It employs a curiosity-driven exploration strategy, predicting action outcomes with uncertainty and adapting its exploration rate (epsilon) based on observed "surprise." This adaptive exploration, along with entropy bonuses, guides the agent towards more informative learning experiences.

### The Surprise Metric

The "surprise metric" is a core mechanism driving the Digital Physicist's curiosity-driven learning. It quantifies how unexpected an observed outcome is, given the agent's current predictions.

**How it's Calculated:** The surprise metric is derived from the agent's prediction error, normalized by its uncertainty. When the agent makes a prediction about the outcome of an action (e.g., whether a code change will be successful), it also estimates its confidence (uncertainty) in that prediction. After the action is executed and the actual outcome is observed, the difference between the predicted and actual outcome constitutes the prediction error.

The formula can be conceptually understood as: `Surprise = Prediction Error / Uncertainty`.

**Purpose and Role:** A high surprise value indicates that the agent's current model of the world was significantly wrong or highly uncertain about a particular outcome. This signals to the agent that there is valuable new information to be learned. By actively seeking out and prioritizing actions that yield high surprise, the Digital Physicist is driven to explore novel situations, refine its understanding of the codebase, and improve its predictive model. This mechanism is crucial for enabling the Digital Physicist's continuous self-improvement and its ability to master complex and novel problems.
*   **Project Graph (`project_graph.py`)**: This module builds a comprehensive, project-wide code graph. It discovers and parses all Python files within a given project root, extracting information about modules, classes, functions, and their import relationships. This information is then used to construct a `networkx` graph, which can be further processed into a PyTorch Geometric `Data` object for GNN analysis.
*   **Semantics (`semantics.py`)**: This module handles the generation of semantic embeddings for code snippets. It uses an LLM-pluggable approach, allowing for integration with external LLM providers for richer semantic understanding. In the absence of an external provider, it falls back to a deterministic hash-based embedding, ensuring the learning loop can continue during development and testing. It also includes caching mechanisms for efficiency.

## The Environment: The Digital Physicist's Sandbox

The Environment component (`nerion_digital_physicist/environment`) provides a controlled sandbox for the Digital Physicist to experiment and learn within.

*   **Core (`core.py`)**: The `EnvironmentV2` class simulates the process of code modification and testing. It applies precise, AST-guided text edits to code, runs tests in an isolated sandbox, and meticulously records metadata about the outcome of each action. This metadata includes details about linting results, validation status, and information from generative LLM providers. It also offers the ability to preview the source code and graph representation of hypothetical actions without altering the actual files, enabling the agent to "imagine" the consequences of its actions.
*   **Actions (`actions.py`)**: This module defines the set of discrete `Action`s that the Digital Physicist can perform within the environment. These actions range from structural changes like `RENAME_LOCAL_VARIABLE_IN_ADD` to more complex operations like `IMPLEMENT_MULTIPLY_DOCSTRING`. It also includes utilities like `StatefulRenameVisitor` for performing AST-based code transformations.
*   **Generative (`generative.py`)**: This module provides the tools for LLM-backed generative code actions. The `GenerativeActionEngine` leverages an LLM to generate function bodies based on provided signatures and docstrings, with robust deterministic fallbacks if the LLM is unavailable. It also includes helper functions to seamlessly apply these generated code snippets back into the source.

## The Generation System: Creating Learning Experiences

The Generation component (`nerion_digital_physicist/generation`) is responsible for autonomously creating the curriculum that the Digital Physicist learns from.

*   **Curriculum Generator (`curriculum_generator.py`)**: This module is the engine for creating new training lessons. It utilizes an LLM to generate `before`, `after`, and `test` code files for a given lesson description. A critical "Self-Vetting" process then verifies the generated lesson's validity and correctness by running tests in a sandbox and analyzing the structural quality of the code change using the GNN. It also incorporates a repair mechanism to automatically fix lessons that fail initial vetting.
*   **Orchestrator (`orchestrator.py`)**: This module manages the entire autonomous learning cycle. It identifies sources of inspiration for new lesson topics, generates concrete lesson ideas using an LLM, and rigorously assesses their potential impact and viability. It integrates meta-policy evaluations and duplicate checking to ensure the quality and novelty of generated lessons before triggering the curriculum generator.

## The Learning Loop: Continuous Improvement

The Digital Physicist is constantly learning and improving its understanding of the codebase. It does this through a continuous learning loop orchestrated by the `learning_orchestrator.py` module, which integrates the Agent, Environment, and Generation components. This loop consists of the following high-level steps:

1.  **Inspiration & Idea Generation**: The Orchestrator identifies strategic focus areas and generates novel lesson ideas using LLMs.
2.  **Impact Assessment & Vetting**: Generated ideas are assessed for impact and clarity, and then rigorously self-vetted by the `curriculum_generator.py` using behavioral (tests) and structural (GNN) checks.
3.  **Experience & Learning**: Validated lessons are used by the Agent within the Environment to perform actions, observe outcomes, and update its GNN brain based on the "surprise" of those outcomes.
4.  **Model Update**: The GNN is continuously refined through training on these experiences, improving its ability to predict the quality of code changes.
