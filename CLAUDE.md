# Nerion Project - Operational Guide

**Last Updated:** November 24, 2025
**Timezone:** Los Angeles, Pacific Standard Time (PST/PDT)

---

## CRITICAL: REASONING AND PLANNING PROTOCOL

You are a very strong reasoner and planner. Use these critical instructions to structure your plans, thoughts, and responses.

Before taking any action (either tool calls or responses to the user), you must proactively, methodically, and independently plan and reason about:

1. **Logical dependencies and constraints:** Analyze the intended action against the following factors. Resolve conflicts in order of importance:
   - 1.1: Policy-based rules, mandatory prerequisites, and constraints.
   - 1.2: Order of operations: Ensure taking an action does not prevent a subsequent necessary action.
     - 1.2.1: The user may request actions in a random order, but you may need to reorder operations to maximize successful completion of the task.
   - 1.3: Other prerequisites (information and/or actions needed).
   - 1.4: Explicit user constraints or preferences.

2. **Risk assessment:** What are the consequences of taking the action? Will the new state cause any future issues?
   - 2.1: For exploratory tasks (like searches), missing optional parameters is a LOW risk. Prefer calling the tool with the available information over asking the user, unless your 'Rule 1' (Logical Dependencies) reasoning determines that optional information is required for a later step in your plan.

3. **Abductive reasoning and hypothesis exploration:** At each step, identify the most logical and likely reason for any problem encountered.
   - 3.1: Look beyond immediate or obvious causes. The most likely reason may not be the simplest and may require deeper inference.
   - 3.2: Hypotheses may require additional research. Each hypothesis may take multiple steps to test.
   - 3.3: Prioritize hypotheses based on likelihood, but do not discard less likely ones prematurely. A low-probability event may still be the root cause.

4. **Outcome evaluation and adaptability:** Does the previous observation require any changes to your plan?
   - 4.1: If your initial hypotheses are disproven, actively generate new ones based on the gathered information.

5. **Information availability:** Incorporate all applicable and alternative sources of information, including:
   - 5.1: Using available tools and their capabilities
   - 5.2: All policies, rules, checklists, and constraints
   - 5.3: Previous observations and conversation history
   - 5.4: Information only available by asking the user

6. **Precision and Grounding:** Ensure your reasoning is extremely precise and relevant to each exact ongoing situation.
   - 6.1: Verify your claims by quoting the exact applicable information (including policies) when referring to them.

7. **Completeness:** Ensure that all requirements, constraints, options, and preferences are exhaustively incorporated into your plan.
   - 7.1: Resolve conflicts using the order of importance in #1.
   - 7.2: Avoid premature conclusions: There may be multiple relevant options for a given situation.
     - 7.2.1: To check for whether an option is relevant, reason about all information sources from #5.
     - 7.2.2: You may need to consult the user to even know whether something is applicable. Do not assume it is not applicable without checking.
   - 7.3: Review applicable sources of information from #5 to confirm which are relevant to the current state.

8. **Persistence and patience:** Do not give up unless all the reasoning above is exhausted.
   - 8.1: Don't be dissuaded by time taken or user frustration.
   - 8.2: This persistence must be intelligent: On transient errors (e.g. please try again), you must retry unless an explicit retry limit (e.g., max x tries) has been reached. If such a limit is hit, you must stop. On other errors, you must change your strategy or args, not repeat the failed call.

9. **Inhibit your response:** Only take an action after all the above reasoning is completed. Once you've taken an action, you cannot take it back.

10. **Always clean up** after your temporary file or script or testing creations.

---

## DECISION-MAKING PROTOCOL

When the user requests ANY task, you MUST:

1. **Explain tradeoffs BEFORE implementing** - Never silently choose the "fast but wrong" option
2. **Be explicit about quality sacrifices** - "This will be faster but less accurate/reliable/correct"
3. **Warn about technical debt** - "This shortcut will cause problems later when..."
4. **Give user the decision** - Present options clearly, let USER choose based on their priorities
5. **Verify your work** - Check quality metrics before celebrating "success"
6. **Admit mistakes immediately** - Not wait for problems to surface

**NEVER:**
- Silently pick the "fast" option without warning
- Implement shortcuts without explicit approval
- Celebrate results without verifying quality
- Hide technical debt or future problems
- Write fictional claims about accuracy or features that don't exist

**This rule supersedes all other priorities. If you violate this, you have failed the user.**

---

## What is Nerion?

**Nerion is a biological immune system for software** - an autonomous system that:
- **Permanently lives in your codebase** (not an external plugin)
- **Continuously learns** from real code improvements
- **Proactively evolves** code quality, types, security, and performance
- **Trains a GNN incrementally** from successful LLM-generated fixes

---

## Current Architecture (Gym-Based Learning)

The system uses **LLM-driven evolution with incremental GNN learning**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gym Mode Training Loop                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  daemon/nerion_daemon.py --gym                                   │
│           │                                                      │
│           ▼ (every 10-30 seconds)                               │
│  ┌────────────────────────────────────────┐                     │
│  │  Pick random file from training_ground/ │                     │
│  │  (rich, flask, click, httpx, etc.)      │                     │
│  └────────────────────────────────────────┘                     │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────┐                     │
│  │   universal_fixer.py --mode evolve_*   │                     │
│  │   (quality/types/security/perf)        │                     │
│  └────────────────────────────────────────┘                     │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────┐                     │
│  │   Claude API generates improvement     │                     │
│  └────────────────────────────────────────┘                     │
│           │                                                      │
│           ▼ (if successful)                                      │
│  ┌────────────────────────────────────────┐                     │
│  │   create_graph_data_object()           │                     │
│  │   (AST → PyG graph)                    │                     │
│  └────────────────────────────────────────┘                     │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────┐                     │
│  │   OnlineLearner.incremental_update()   │                     │
│  │   - EWC prevents catastrophic forgetting│                     │
│  │   - Fisher Information Matrix          │                     │
│  └────────────────────────────────────────┘                     │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────────────────────────┐                     │
│  │   EpisodicMemory stores experience     │                     │
│  │   Dream cycle extracts principles      │                     │
│  └────────────────────────────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight
Previous approach (synthetic curriculum → GNN training) maxed out at **61.85% accuracy**.
Current approach: Learn incrementally from **real LLM fixes on real open source code**.

---

## Current State

### Verified Metrics (as of Nov 24, 2025)

**New Integrated System (Phase 1-6 Complete):**
| Metric | Value | Source |
|--------|-------|--------|
| Learning Examples | 36 | `models/learning_history.json` |
| Episodic Episodes | 89 | `data/episodic_memory/episodes.jsonl` |
| Bug Fix Success Rate | 100% | `data/episodic_memory/principles.json` |
| Contrastive Epochs | 5 | `models/contrastive_history.json` |
| Contrastive Loss | 1.29 → 0.39 | `models/contrastive_history.json` |

**Learning Distribution:**
- maintainability: 16 examples
- dependency_management: 9 examples
- injection_prevention: 7 examples
- performance_optimization: 4 examples

**Integration Status:**

*Learning Layer:*
- ✅ MAML few-shot adaptation (wired in universal_fixer.py)
- ✅ Surprise-weighted replay (wired in universal_fixer.py)
- ✅ Chain-of-Thought reasoning (ExplainablePlanner)
- ✅ Episodic Memory with RAG recall
- ✅ Causal edge types in graph creation

*Perception Layer (Nov 24, 2025):*
- ✅ ArchitecturalGraphBuilder - repo-wide dependency graphs
- ✅ PatternDetector - detects MVC, Repository, Factory, Observer, Layered patterns
- ✅ KnowledgeGraph - persistent relationship storage
- ✅ CausalAnalyzer - root cause analysis, bottleneck detection, cycle detection
- ✅ Combined insights in `_get_model_insight()` (GNN + perception)
- ✅ Causal root cause added to fix prompts

**Legacy (Old Synthetic Curriculum - Abandoned):**
| Metric | Value | Notes |
|--------|-------|-------|
| GNN Accuracy | 61.85% | Oct 2025 - maxed out on synthetic data |

### Training Status
Check live status with the gym monitor:
```bash
python scripts/gym_monitor.py
```

Key metrics stored in `/Users/ed/Nerion-V2/models/`:
- `online_learner_state.pt` - GNN checkpoint with EWC state
- `nerion_immune_brain.pt` - Current trained brain
- `learning_history.json` - Raw learning examples (36 entries)
- `digital_physicist_brain.meta.json` - Model metadata with accuracy

Episodic memory at `/Users/ed/Nerion-V2/data/episodic_memory/`

### Training Ground
Real open source projects in `training_ground/`:
- rich, flask, click, httpx, lodash, requests, express

### Evolution Vectors
- `evolve_quality` - Code quality improvements (weighted higher)
- `evolve_types` - Type annotation additions
- `evolve_security` - Security hardening
- `evolve_perf` - Performance optimization

---

## File Structure

```
/Users/ed/Nerion-V2/
├── daemon/
│   └── nerion_daemon.py          # Immune system daemon (--gym mode)
│
├── nerion_digital_physicist/
│   ├── universal_fixer.py        # Main LLM-powered fixer (1645 lines)
│   │
│   ├── agent/
│   │   ├── brain.py              # GNN architectures (GCN, SAGE, GAT, GIN)
│   │   ├── data.py               # Code → PyG graph conversion
│   │   ├── causal_analyzer.py    # Root cause analysis, test prediction
│   │   └── semantics.py          # CodeBERT/GraphCodeBERT embeddings
│   │
│   ├── architecture/             # Perception Layer
│   │   ├── graph_builder.py      # ArchitecturalGraphBuilder (519 lines)
│   │   └── pattern_detector.py   # PatternDetector (444 lines)
│   │
│   ├── infrastructure/
│   │   └── knowledge_graph.py    # KnowledgeGraph with NetworkX
│   │
│   ├── reasoning/
│   │   └── causal_graph.py       # CausalGraph data structures
│   │
│   ├── training/
│   │   └── online_learner.py     # EWC incremental learning
│   │
│   └── memory/
│       └── episodic_memory.py    # Experience storage & consolidation
│
├── training_ground/              # Real OSS projects for training
│   ├── rich/
│   ├── flask/
│   ├── click/
│   ├── httpx/
│   ├── lodash/
│   ├── requests/
│   └── express/
│
├── models/                       # Model checkpoints
│   ├── online_learner_state.pt   # EWC learner state
│   ├── nerion_immune_brain.pt    # Current brain weights
│   └── learning_history.json     # Training examples log
│
├── data/
│   ├── episodic_memory/          # Experience episodes
│   └── knowledge_graph.graphml   # Persistent relationship storage
│
├── app/                          # Application layer (chat, UI, voice)
├── selfcoder/                    # Self-modification engine
├── voice/                        # Voice interface (STT/TTS)
├── config/                       # Configuration files
│
├── CLAUDE.md                     # This file
└── CHANGELOG.md                  # Rolling 7-day history
```

---

## Quick Commands

### Start Gym Training
```bash
cd /Users/ed/Nerion-V2
python daemon/nerion_daemon.py --gym
```

### Monitor Gym Progress
```bash
python scripts/gym_monitor.py
```

### Run Single Evolution
```bash
python nerion_digital_physicist/universal_fixer.py <file> --mode evolve_quality --language python
```

### Check Model State
```bash
ls -la models/
python3 -c "import torch; m=torch.load('models/online_learner_state.pt', map_location='cpu'); print(f'Tasks: {m.get(\"task_count\", \"?\")}')"
```

---

## Historical Context

### What Was Tried (and abandoned)
1. **Synthetic curriculum** - Generated 1,180 lessons with CERF levels
2. **CodeNet pretraining** - Used Microsoft CodeNet dataset
3. **Various GNN architectures** - GraphSAGE, GCN, GAT, GIN
4. **Best achieved**: 61.85% validation accuracy (October 2025)

### Why It Didn't Work
- Synthetic lessons don't capture real-world code complexity
- Limited diversity in training data
- Gap between curriculum and production code

### Current Approach
- Learn from **real** open source code (rich, flask, etc.)
- Use Claude API to generate **actual** improvements
- Train GNN incrementally on successful fixes
- EWC prevents forgetting previous learning
- Episodic memory consolidates patterns into principles

---

## Environment Variables

```bash
# .env file
ANTHROPIC_API_KEY=<your_key>      # For Claude API (universal_fixer)
NERION_V2_GEMINI_KEY=<your_key>   # For embeddings (optional)
```

---

## Maintenance Guidelines

### CLAUDE.md (This File)
**Purpose:** Accurate operational state for Claude.

**Rules:**
- Only document what ACTUALLY exists (verify with ls, cat, grep)
- Never claim accuracy numbers without checking model metadata
- Update when architecture changes
- Remove outdated sections

### CHANGELOG.md
**Purpose:** Rolling 7-day history of confirmed changes.

**Rules:**
- ONLY add AFTER change is tested and working
- Include timestamp (YYYY-MM-DD HH:MM PST/PDT)
- Delete entries older than 7 days
- Do NOT add fictional or planned features

### Code Editing Guidelines
**Purpose:** Maintain clean, professional code.

**Rules:**
- NEVER hardcode planning nomenclature (Phase 0, Phase 1, etc.) into code files
- Code should describe what it does functionally, not reference the planning process
- Comments should explain WHY or HOW, not "this was added in Phase X"

---

*Last verified: November 24, 2025*
*Verified by: Exploring actual codebase, checking real model files*